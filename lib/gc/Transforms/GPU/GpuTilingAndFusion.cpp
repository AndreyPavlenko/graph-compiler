//===-- GpuTilingAndFusion.cpp - DESC ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include <mlir/Dialect/Affine/Utils.h>

using namespace mlir;
// using namespace mlir::gc::gpu;

namespace mlir::gc {
#define GEN_PASS_DECL_GPUTILINGANDFUSION
#define GEN_PASS_DEF_GPUTILINGANDFUSION
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

struct GpuTilingAndFusion final
    : gc::impl::GpuTilingAndFusionBase<GpuTilingAndFusion> {
  explicit GpuTilingAndFusion()
      : GpuTilingAndFusion(gc::GpuTilingAndFusionOptions{}) {}
  explicit GpuTilingAndFusion(const gc::GpuTilingAndFusionOptions &opts)
      : GpuTilingAndFusionBase(opts) {}

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    scf::SCFTileAndFuseOptions outOpts;
    scf::SCFTileAndFuseOptions innerOpts;
    outOpts.tilingOptions.setLoopType(
        scf::SCFTilingOptions::LoopType::ForallOp);
    innerOpts.tilingOptions.setLoopType(
        scf::SCFTilingOptions::LoopType::ForallOp);
    // The outer loop is converted to a GPU kernel and the tile sizes are mapped
    // to the grid sizes.
    outOpts.tilingOptions.setTileSizeComputationFunction(
        // The tile sizes calculation is based on the following equation:
        // n * TS0 * TS1 * ... * TSn = euMem
        // where:
        // n - an average number of bytes, processed by each iteration
        // TS0, TS1, ... TSn - the tile sizes for each loop correspondingly
        // euMem - the physical memory (cache) size of the GPU execution unit
        //
        // To calculate the tile size TS, we need to divide the total loop size
        // S by the ratio r:
        //
        // n * (S0/r0) * (S1/r1) * ... * (Sn/rn) = euMem
        // r0 * r1 * ... * rn = (n * S0 * S1 * ... * Sn) / euMem
        // If all sizes are equal, then S0 = ... = Sn = S, r0 = ... = rn = r:
        // r^n = (n * S^n) / euMem
        // r = (n * S^n / euMem)^(1/n)
        [euMem = getEuMem(rewriter)](
            OpBuilder &builder, Operation *op) -> SmallVector<OpFoldResult> {
          auto ti = dyn_cast<TilingInterface>(op);
          if (!ti) {
            return {};
          }

          auto itTypes = ti.getLoopIteratorTypes();
          auto itDomains = ti.getIterationDomain(builder);
          assert(itTypes.size() == itDomains.size());

          // TODO: Add a parameter to the options?
          size_t totalSize = calcOperandsSize(op);
          unsigned loopCount = 0;

          for (auto [t, r] : zip(itTypes, itDomains)) {
            if (t == utils::IteratorType::parallel) {
              if (auto v = getConstantIntValue(r.size)) {
                loopCount++;
                totalSize *= *v;
              } else {
                return calcDynamicSizes(builder, ti, euMem);
              }
            }
          }

          if (loopCount == 0) {
            return {};
          }

          // TODO: In case of different sizes, calculate the ratio for each loop
          double ratio = std::pow(static_cast<double>(totalSize) /
                                      static_cast<double>(euMem),
                                  1.0 / loopCount);
          SmallVector<OpFoldResult> tiles;
          tiles.reserve(itDomains.size());

          for (auto [t, r] : zip(itTypes, itDomains)) {
            if (t != utils::IteratorType::parallel) {
              tiles.emplace_back(builder.getIndexAttr(1));
            } else if (auto v = getConstantIntValue(r.size)) {
              tiles.emplace_back(ceil(builder, *v, ratio));
            } else {
              abort(); // Must never get here
            }
          }

          return tiles;
        });

    // The inner loop tiles are mapped to the thread group sizes.
    innerOpts.tilingOptions.setTileSizeComputationFunction(
        // The tile sizes are calculated by dividing the total loop size by the
        // number of execution unit threads:
        // TS = ceil(S / euThreads)
        [euThreads = getEuThreads(rewriter)](
            OpBuilder &builder, Operation *op) -> SmallVector<OpFoldResult> {
          auto ti = dyn_cast<TilingInterface>(op);
          if (!ti) {
            return {};
          }

          SmallVector<OpFoldResult> tiles;
          auto itTypes = ti.getLoopIteratorTypes();
          auto itDomains = ti.getIterationDomain(builder);
          assert(itTypes.size() == itDomains.size());
          unsigned loopCounter = 0;

          for (auto [t, r] : zip(itTypes, itDomains)) {
            if (t != utils::IteratorType::parallel) {
              tiles.emplace_back(builder.getIndexAttr(1));
              continue;
            }

            loopCounter++;

            if (auto cv = getConstantIntValue(r.size)) {
              tiles.emplace_back(ceil(builder, *cv, euThreads));
              continue;
            }

            if (auto value = getConstIdxValue(r.size.get<Value>())) {
              tiles.emplace_back(ceil(builder, value, euThreads));
              continue;
            }

            if (auto outerLoop = op->getParentOfType<scf::ForallOp>()) {
              if (auto steps = outerLoop.getStep(builder);
                  steps.size() >= loopCounter) {
                auto step = steps[loopCounter - 1];
                if (auto v = getConstantIntValue(step)) {
                  tiles.emplace_back(ceil(builder, *v, euThreads));
                  continue;
                }
                if (auto stepOp = step.getDefiningOp()) {
                  // Move the step calculation out of the loop
                  auto ip = builder.saveInsertionPoint();
                  builder.setInsertionPointAfter(stepOp);
                  auto loc = stepOp->getLoc();
                  auto ratio =
                      builder.create<arith::ConstantIndexOp>(loc, euThreads);
                  tiles.emplace_back(builder.create<arith::CeilDivUIOp>(
                      loc, stepOp->getResult(0), ratio));
                  builder.restoreInsertionPoint(ip);
                  continue;
                }
              }
            }

            return {};
          }

          return tiles;
        });

    tileAndFuse(getOperation(), rewriter, &outOpts, &innerOpts);
  }

private:
  static void tileAndFuse(Operation *op, RewriterBase &rewriter,
                          const scf::SCFTileAndFuseOptions *outOpts,
                          const scf::SCFTileAndFuseOptions *innerOpts) {
    SmallVector<TilingInterface> ops;
    op->walk([&](linalg::LinalgOp linalgOp) {
      if (auto ti = dyn_cast<TilingInterface>(linalgOp.getOperation())) {
        ops.emplace_back(ti);
      }
    });

    SmallVector<Operation *> replaced;
    for (auto ti : llvm::reverse(ops)) {
      if (std::find(replaced.begin(), replaced.end(), ti.getOperation()) !=
          replaced.end()) {
        continue;
      }

      auto result =
          scf::tileConsumerAndFuseProducersUsingSCF(rewriter, ti, *outOpts);

      if (failed(result)) {
        ti.emitError() << "Failed to tile and fuse using SCF";
        return;
      }

      SmallVector<Operation *> opsToReplace{ti.getOperation()};
      append_range(opsToReplace, result->fusedProducers);
      for (Operation *toReplace : opsToReplace) {
        for (OpResult res : toReplace->getResults()) {
          if (auto repl = result->replacements.lookup(res)) {
            if (innerOpts) {
              tileAndFuse(repl.getDefiningOp(), rewriter, innerOpts, nullptr);
            }
            rewriter.replaceAllUsesWith(res, repl);
          }
        }

        // For some reason (probably a bug?) the operation could be
        // referenced by a dead code inside the replacement, that prevents this
        // operation from being erased. Erasing the dead code first.
        for (auto u : toReplace->getUsers()) {
          if (u->use_empty()) {
            rewriter.eraseOp(u);
          }
        }

        if (toReplace->use_empty()) {
          rewriter.eraseOp(toReplace);
          replaced.push_back(toReplace);
        }
      }
    }
  }

  static SmallVector<OpFoldResult>
  calcDynamicSizes(OpBuilder &builder, TilingInterface ti, size_t euMem) {
    auto itTypes = ti.getLoopIteratorTypes();
    auto itDomains = ti.getIterationDomain(builder);
    assert(itTypes.size() == itDomains.size());

    auto loc = ti.getLoc();
    Value dynamicSize;
    size_t staticSize = calcOperandsSize(ti.getOperation());
    unsigned loopCount = 0;

    for (auto [t, r] : zip(itTypes, itDomains)) {
      if (t != utils::IteratorType::parallel) {
        continue;
      }
      loopCount++;
      if (auto v = getConstantIntValue(r.size)) {
        staticSize *= *v;
      } else if (dynamicSize) {
        dynamicSize = builder.create<arith::MulIOp>(loc, dynamicSize,
                                                    r.size.get<Value>());
      } else {
        dynamicSize = r.size.get<Value>();
      }
    }

    assert(loopCount);
    assert(dynamicSize);
    if (staticSize > 1) {
      dynamicSize = builder.create<arith::MulIOp>(
          loc, dynamicSize,
          builder.create<arith::ConstantIndexOp>(loc, staticSize));
    }
    dynamicSize = builder.create<arith::UIToFPOp>(
        loc, builder.getF64Type(),
        builder.create<arith::IndexCastOp>(loc, builder.getI64Type(),
                                           dynamicSize));

    auto memSize = builder.create<arith::ConstantFloatOp>(
        loc, APFloat(static_cast<double>(euMem)), builder.getF64Type());
    auto pow = builder.create<arith::ConstantFloatOp>(
        loc, APFloat(1.0 / loopCount), builder.getF64Type());
    auto ratio = builder.create<math::PowFOp>(
        loc, builder.create<arith::DivFOp>(loc, dynamicSize, memSize), pow);

    SmallVector<OpFoldResult> tiles;
    tiles.reserve(itDomains.size());

    for (auto [t, r] : zip(itTypes, itDomains)) {
      if (t != utils::IteratorType::parallel) {
        tiles.emplace_back(builder.getIndexAttr(1));
      } else {
        Value value;
        if (auto v = getConstantIntValue(r.size)) {
          value = builder.create<arith::ConstantFloatOp>(
              loc, APFloat(static_cast<double>(*v)), builder.getF64Type());
        } else {
          value = builder.create<arith::UIToFPOp>(
              loc, builder.getF64Type(),
              builder.create<arith::IndexCastOp>(loc, builder.getI64Type(),
                                                 r.size.get<Value>()));
        }
        auto ts = builder.create<arith::FPToUIOp>(
            loc, builder.getI64Type(),
            builder.create<math::CeilOp>(
                loc, builder.create<arith::DivFOp>(loc, value, ratio)));
        tiles.emplace_back(builder.create<arith::IndexCastOp>(
            loc, builder.getIndexType(), ts));
      }
    }

    return tiles;
  }

  static size_t calcOperandsSize(Operation *op) {
    size_t size = 0;
    auto typeSize = [](Type t) -> size_t {
      Type et;
      if (auto mt = dyn_cast<MemRefType>(t)) {
        et = mt.getElementType();
      } else if (auto tt = dyn_cast<TensorType>(t)) {
        et = tt.getElementType();
      } else {
        return 0;
      }
      return et.isIntOrFloat() ? et.getIntOrFloatBitWidth() / 8 : 1;
    };
    for (auto operand : op->getOperands()) {
      if (auto defOp = operand.getDefiningOp()) {
        for (auto t : defOp->getResultTypes()) {
          size += typeSize(t);
        }
      } else {
        size += typeSize(operand.getType());
      }
    }
    return size == 0 ? 1 : size;
  }

  static int64_t getConstIdxValue(Value value) {
    if (auto op = value.getDefiningOp<arith::ConstantIndexOp>()) {
      return op.value();
    }
    if (auto minOp = value.getDefiningOp<affine::AffineMinOp>()) {
      for (const AffineExpr &result : minOp.getMap().getResults()) {
        if (auto constExpr = dyn_cast<AffineConstantExpr>(result)) {
          return constExpr.getValue();
        }
      }
    }
    if (auto minOp = value.getDefiningOp<arith::MinSIOp>()) {
      for (Value operand : {minOp.getLhs(), minOp.getRhs()}) {
        if (auto v = getConstIdxValue(operand))
          return v;
      }
    }
    return 0;
  }

  template <typename A, typename B>
  static IntegerAttr ceil(OpBuilder &builder, A a, B b) {
    return builder.getIndexAttr(
        static_cast<int64_t>(std::ceil(static_cast<double>(a) / b)));
  }

  size_t getEuMem(OpBuilder &builder) {
    return getDeviceProperty(builder, "L1_cache_size_in_bytes", euMem);
  }

  size_t getEuThreads(OpBuilder &builder) {
    return getDeviceProperty(builder, "threads_per_eu", euThreads);
  }

  size_t getDeviceProperty(OpBuilder &builder, StringRef name,
                           size_t defaultValue) {
    if (auto mod = getOperation()->getParentOfType<ModuleOp>()) {
      DataLayout layout(mod);
      if (auto value = layout.getDevicePropertyValue(
              builder.getStringAttr("GPU" /* device ID*/),
              builder.getStringAttr(name))) {
        if (auto attr = dyn_cast<IntegerAttr>(*value)) {
          return static_cast<size_t>(attr.getInt());
        }
      }
    }
    return defaultValue;
  }
};
} // namespace
