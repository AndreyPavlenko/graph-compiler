//===-- GpuUtils.h - DESC ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef GPUUTILS_H
#define GPUUTILS_H

#include "gc/Utils/Log.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

#include <numeric>

using namespace mlir;

namespace mlir::gc {
template <typename DerivedT> struct GpuPass {

  int64_t getGpuPropertyAsInt(Builder &builder, StringRef name,
                              int64_t defaultValue) {
    if (auto mod = static_cast<DerivedT *>(this)
                       ->getOperation()
                       ->template getParentOfType<ModuleOp>()) {
      DataLayout layout(mod);
      if (auto value = layout.getDevicePropertyValue(
              builder.getStringAttr("GPU" /* device ID*/),
              builder.getStringAttr(name))) {
        if (auto attr = dyn_cast<IntegerAttr>(*value)) {
          return attr.getInt();
        }
      }
    }
    return defaultValue;
  }

  int64_t getNumEus(Builder &builder) {
    return getGpuPropertyAsInt(builder, "num_exec_units",
                               static_cast<DerivedT *>(this)->numEus);
  }

  int64_t getNumEusPerSlice(Builder &builder) {
    return getGpuPropertyAsInt(builder, "num_exec_units_per_slice",
                               static_cast<DerivedT *>(this)->numEusPerSlice);
  }

  int64_t getNumThreadsPerEu(Builder &builder) {
    return getGpuPropertyAsInt(builder, "num_threads_per_eu",
                               static_cast<DerivedT *>(this)->numThreadsPerEu);
  }

  int64_t getCacheSize(Builder &builder) {
    return getGpuPropertyAsInt(builder, "L1_cache_size_in_bytes",
                               static_cast<DerivedT *>(this)->cacheSize);
  }

  int64_t getVectorWidth(Builder &builder) {
    return getGpuPropertyAsInt(builder, "max_vector_op_width",
                               static_cast<DerivedT *>(this)->vectorWidth);
  }

  int64_t getWorkGroupSize(Builder &builder) {
    return getGpuPropertyAsInt(builder, "max_work_group_size",
                               static_cast<DerivedT *>(this)->workGroupSize);
  }
};

struct OpRewriter final : IRRewriter {
  Location loc;

  explicit OpRewriter(func::FuncOp &func)
      : IRRewriter(func.getContext()), loc(func.getLoc()), func(func) {}

  template <typename OpTy, typename... Args> OpTy create(Args &&...args) {
    return RewriterBase::create<OpTy>(loc, std::forward<Args>(args)...);
  }

  arith::ConstantIndexOp getConstant(int64_t v) {
    if (func.empty()) {
      func.addEntryBlock();
    } else {
      // Try to find ConstantIndexOp with the same value in the function.
      for (auto &op : func.getBody().getOps()) {
        if (auto constOp = dyn_cast<arith::ConstantIndexOp>(op)) {
          if (constOp.value() == v) {
            return constOp;
          }
        }
      }
    }

    auto ip = saveInsertionPoint();
    setInsertionPointToStart(&func.getBody().front());
    auto op = create<arith::ConstantIndexOp>(v);
    restoreInsertionPoint(ip);
    return op;
  }

private:
  func::FuncOp &func;
};

// Round to the largest power of 2 that is <= value.
template <typename T> static T floorPow2(T value) {
  auto v = static_cast<std::make_unsigned_t<T>>(value);
  return T(1) << llvm::bit_width(v) - 1;
}

// Round to the smallest power of 2 that is >= value.
template <typename T> static T ceilPow2(T value) {
  auto v = static_cast<std::make_unsigned_t<T>>(value);
  return llvm::bit_ceil(v);
}

// Adjust tile sizes that meet the following conditions:
// 1. The product of all tiles is as close to totalSize as possible.
// 2. The new sizes are proportional to the initial sizes.
// 3. The new sizes are powers of 2.
template <typename T>
static void adjustTiles(T totalSize, T *begin, T *end,
                        T minSize = static_cast<T>(1)) {
  assert((minSize & (minSize - 1)) == 0 && "minSize must be a power of 2");
  auto count = end - begin;
  if (count == 0) {
    return;
  }

  T total = ceilPow2(totalSize);

  if (count == 1) {
    *begin = std::min(std::max(ceilPow2(*begin), minSize), total);
    return;
  }

  if (count > 2) {
    // Split the array in two. The first one consists of the 2 elements - the
    // first one and the product of the rest. The second one is the rest.
    T first[] = {*begin, std::accumulate(begin + 2, end, *(begin + 1),
                                         std::multiplies<>())};
    adjustTiles(total, first, first + 2, minSize);
    adjustTiles(total / *first, begin + 1, end, minSize);
    *begin = *first;
    return;
  }

  --end;
  T a = *begin;
  T b = *end;
  bool swap = a < b;
  if (swap) {
    std::swap(a, b);
  }

  a = ceilPow2(a);
  b = floorPow2(b);

  if (a * b <= total) {
    *begin = swap ? b : a;
    *end = swap ? a : b;
    return;
  }

  double ratio = static_cast<double>(a) / static_cast<double>(b);
  T x = static_cast<T>(std::sqrt(total)) * static_cast<T>(std::sqrt(ratio));
  x = std::min(ceilPow2(x), std::min(a, total));
  T y = std::min(floorPow2(total / x), b);

  // Adjust x and y to get the closest ratio
  if (auto diff =
          std::abs(ratio - static_cast<double>(x) / static_cast<double>(y));
      y >= 2 && x * 2 <= a &&
      std::abs(ratio - static_cast<double>(x * 2) /
                           static_cast<double>(y / 2)) < diff) {
    x *= 2;
    y /= 2;
  } else if (x >= 2 && y * 2 <= b &&
             std::abs(ratio - static_cast<double>(x / 2) /
                                  static_cast<double>(y * 2)) < diff) {
    x /= 2;
    y *= 2;
  }

  if (y < minSize && a >= minSize && b >= minSize) {
    if (auto newX = ceilPow2(total / minSize); newX >= minSize) {
      x = std::min(newX, a);
      y = minSize;
    }
  }

  *begin = swap ? y : x;
  *end = swap ? x : y;
}

template <typename T, unsigned N>
static void adjustTiles(T totalSize, SmallVector<T, N> &tiles,
                        T minSize = static_cast<T>(1)) {
  adjustTiles(totalSize, tiles.begin(), tiles.end(), minSize);
}

// Check recursively if the specified operation has an operand that
// depends on a result of a previous operation, matching the predicate.
template <unsigned MaxDepth = std::numeric_limits<unsigned>::max()>
bool isOperandDependsOnOp(bool (*predicate)(Operation *), Operation *operation,
                          unsigned depth = 0) {
  for (auto operand : operation->getOperands()) {
    if (auto op = operand.getDefiningOp();
        op &&
        (predicate(op) || (depth < MaxDepth &&
                           isOperandDependsOnOp(predicate, op, depth + 1)))) {
      return true;
    }
  }
  return false;
}

// Check recursively if there are any operation, matching the predicate, that
// depends on the result of the specified operation.
template <unsigned MaxDepth = std::numeric_limits<unsigned>::max()>
bool isOpDependsOnResult(bool (*predicate)(Operation *), Operation *operation,
                         unsigned depth = 0) {
  for (auto res : operation->getResults()) {
    for (auto u : res.getUsers()) {
      if (predicate(u) ||
          (depth < MaxDepth && isOpDependsOnResult(predicate, u, depth + 1))) {
        return true;
      }
    }
  }
  return false;
}
} // namespace mlir::gc
#endif
