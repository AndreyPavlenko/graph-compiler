//===-- GpuTilingAndFusion.cpp - DESC ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "./GpuUtils.h"
#include "gc/Dialect/Linalgx/Utils.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace mlir::gc;
using namespace mlir::scf;

namespace mlir::gc {
#define GEN_PASS_DECL_GPUTILINGANDFUSION
#define GEN_PASS_DEF_GPUTILINGANDFUSION
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

struct GpuTilingAndFusion final
    : GpuPass<GpuTilingAndFusion>,
      gc::impl::GpuTilingAndFusionBase<GpuTilingAndFusion> {
  friend struct GpuPass;
  explicit GpuTilingAndFusion()
      : GpuTilingAndFusion(GpuTilingAndFusionOptions{}) {}
  explicit GpuTilingAndFusion(const GpuTilingAndFusionOptions &opts)
      : GpuPass(), GpuTilingAndFusionBase(opts) {}

  void runOnOperation() override {
    auto fn = getOperation();
    if (fn.isExternal()) {
      return;
    }

    OpRewriter rw(fn);
    tileAndFuseLinalgOps(rw, fn);
    tileForallOps(rw, fn);
    if (failed(simplifyRegions(rw, fn->getRegions()))) {
      // Not simplified
    }
  }

private:
  void tileAndFuseLinalgOps(OpRewriter &rw, func::FuncOp &fn) {
    auto numEus = getNumEus(rw);
    auto numEusPerSlice = getNumEusPerSlice(rw);
    auto numThreadsPerEu = getNumThreadsPerEu(rw);
    auto cacheSize = getCacheSize(rw);
    auto vectorWidth = getVectorWidth(rw);
    auto cachePerThread =
        std::max(cacheSize / numEusPerSlice / numThreadsPerEu, vectorWidth);
    SCFTileAndFuseOptions opts;
    opts.tilingOptions.setTileSizeComputationFunction(
        [&rw, cachePerThread, vectorWidth,
         numThreads = numEus * numThreadsPerEu](
            OpBuilder &builder, Operation *op) -> SmallVector<OpFoldResult> {
          auto ti = dyn_cast<TilingInterface>(op);
          if (!ti) {
            return {};
          }

          rw.loc = op->getLoc();
          rw.setInsertionPoint(op);
          auto itTypes = ti.getLoopIteratorTypes();
          auto itDomains = ti.getIterationDomain(builder);
          assert(itTypes.size() == itDomains.size());

          SmallVector<int64_t> sizes;
          int64_t maxSize = 0;
          int64_t numIterations = 1;
          for (auto [t, r] : zip(itTypes, itDomains)) {
            if (t == utils::IteratorType::parallel) {
              if (auto v = getConstantIntValue(r.size)) {
                numIterations *= *v;
                sizes.emplace_back(*v);
                maxSize = std::max(maxSize, *v);
              } else {
                return computeDynamicTiles(rw, ti, numThreads, cachePerThread);
              }
            }
          }

          assert(!sizes.empty());
          auto elementSize = getElementSize(op);
          auto sizePerThread = numIterations / numThreads * elementSize;
          auto totalSize = std::max(sizePerThread, cachePerThread);
          totalSize = std::max(totalSize / elementSize, 64L);
          int64_t minTileSize = 1;

          // If the operation could be lowered to XeGPU, make the tiles
          // multiple of the vector width and the minimum tile size 8.
          if (canLowerToXeGPU(op)) {
            minTileSize = 8;
            totalSize = std::max(totalSize / vectorWidth, 1L) * vectorWidth;
          }

          SmallVector<int64_t> tiles = sizes;
          adjustTiles(totalSize, tiles, minTileSize);

          // If the tiles are not less than the sizes, split the largest tile
          // into two to avoid loops elimination by the canonicalizer pass.
          if (auto pairs = zip(sizes, tiles);
              std::all_of(pairs.begin(), pairs.end(), [](auto p) {
                return std::get<0>(p) <= std::get<1>(p);
              })) {
            for (auto &t : tiles) {
              if (t >= maxSize) {
                t = std::max(1L, floorPow2(maxSize / 2));
                break;
              }
            }
          }

          unsigned counter = 0;
          SmallVector<OpFoldResult> result;
          result.reserve(itDomains.size());

          for (auto [t, r] : zip(itTypes, itDomains)) {
            if (t == utils::IteratorType::parallel) {
              result.emplace_back(rw.getConstant(tiles[counter++]));
            } else {
              result.emplace_back(rw.getConstant(0));
            }
          }

          return result;
        });
    opts.setFusionControlFn(
        [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
            bool) -> std::optional<SCFTileAndFuseOptions::ControlFnResult> {
          Operation *op = originalProducer.getOwner();
          if (!op) {
            return std::nullopt;
          }
          if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
            if (!linalgOp.hasOnlyProjectedPermutations()) {
              return std::nullopt;
            }
          }

          // If the result of this slice is used by a MatmulOp and the slice has
          // an operand produced by a previous MatmulOp, do not fuse.
          if (isOpDependsOnResult<0>(linalgx::isMatmulOp, candidateSliceOp) &&
              isOperandDependsOnOp(linalgx::isMatmulOp, candidateSliceOp)) {
            return std::nullopt;
          }

          return SCFTileAndFuseOptions::ControlFnResult{};
        });
    opts.tilingOptions.setLoopType(SCFTilingOptions::LoopType::ForallOp);

    for (auto ti = findTi(rw, fn); ti; ti = findTi(rw, fn)) {
      auto result = tileConsumerAndFuseProducersUsingSCF(rw, *ti, opts);

      if (failed(result)) {
        ti->emitError() << "Failed to tile and fuse using SCF";
        return;
      }

      SmallVector<Operation *> opsToReplace{ti->getOperation()};
      append_range(opsToReplace, result->fusedProducers);
      for (Operation *toReplace : opsToReplace) {
        for (OpResult res : toReplace->getResults()) {
          if (auto repl = result->replacements.lookup(res)) {
            rw.replaceAllUsesWith(res, repl);
          }
        }
      }

      if (failed(simplifyRegions(rw, fn->getRegions()))) {
        // Not simplified
      }
    }
  }

  static std::optional<TilingInterface> findTi(OpBuilder &b, Operation *op) {
    std::optional<TilingInterface> last;
    op->walk<WalkOrder::PreOrder>([&](linalg::LinalgOp linalgOp) {
      if (linalgOp.hasOnlyProjectedPermutations() &&
          !linalgOp->getParentOfType<ForallOp>()) {
        if (auto ti = dyn_cast<TilingInterface>(linalgOp.getOperation())) {
          int64_t numTiles = 0;
          int64_t numIterations = 1;
          for (auto [t, r] :
               zip(ti.getLoopIteratorTypes(), ti.getIterationDomain(b))) {
            if (t == utils::IteratorType::parallel) {
              numTiles++;
              if (auto v = getConstantIntValue(r.size)) {
                numIterations *= *v;
              }
            }
          }
          if (numTiles > 0 && numIterations >= 32) {
            last = ti;
          }
        }
      }
      return WalkResult::skip();
    });
    return last;
  }

  static SmallVector<OpFoldResult> computeDynamicTiles(OpRewriter &rw,
                                                       TilingInterface ti,
                                                       int64_t numThreads,
                                                       int64_t cachePerThread) {
    auto itTypes = ti.getLoopIteratorTypes();
    auto itDomains = ti.getIterationDomain(rw);
    assert(itTypes.size() == itDomains.size());
    rw.loc = ti.getLoc();
    rw.setInsertionPoint(ti.getOperation());

    Value dynamicSize;
    auto staticSize = getElementSize(ti.getOperation());
    unsigned loopCount = 0;

    for (auto [t, r] : zip(itTypes, itDomains)) {
      if (t != utils::IteratorType::parallel) {
        continue;
      }
      loopCount++;
      if (auto v = getConstantIntValue(r.size)) {
        staticSize *= *v;
      } else if (dynamicSize) {
        dynamicSize =
            rw.create<arith::MulIOp>(dynamicSize, r.size.get<Value>());
      } else {
        dynamicSize = r.size.get<Value>();
      }
    }

    assert(loopCount);
    assert(dynamicSize);
    if (staticSize > 1) {
      dynamicSize =
          rw.create<arith::MulIOp>(dynamicSize, rw.getConstant(staticSize));
    }
    auto i64Type = rw.getI64Type();
    dynamicSize = rw.create<arith::UIToFPOp>(
        rw.getF64Type(), rw.create<arith::IndexCastOp>(i64Type, dynamicSize));

    // TODO: Call the adjustTiles() function for the tiles calculation.

    auto nt = rw.create<arith::ConstantFloatOp>(
        APFloat(static_cast<double>(numThreads)), rw.getF64Type());
    auto cpt = rw.create<arith::ConstantFloatOp>(
        APFloat(static_cast<double>(cachePerThread)), rw.getF64Type());
    Value totalSize = rw.create<arith::MaximumFOp>(
        rw.getF64Type(), rw.create<arith::DivFOp>(dynamicSize, nt), cpt);
    auto pow = rw.create<arith::ConstantFloatOp>(APFloat(1.0 / loopCount),
                                                 rw.getF64Type());
    // The average tile size is totalSize^(1 / loopCount)
    Value avgTileSize = rw.create<math::PowFOp>(totalSize, pow);
    avgTileSize = rw.create<arith::MaximumFOp>(
        rw.getF64Type(),
        rw.create<arith::ConstantFloatOp>(APFloat(1.0), rw.getF64Type()),
        avgTileSize);
    avgTileSize = rw.create<arith::FPToSIOp>(i64Type, avgTileSize);

    SmallVector<OpFoldResult> tiles;
    tiles.reserve(itDomains.size());

    for (auto [t, r] : zip(itTypes, itDomains)) {
      if (t != utils::IteratorType::parallel) {
        tiles.emplace_back(rw.getIndexAttr(1));
      } else {
        Value value;
        if (auto v = getConstantIntValue(r.size)) {
          value = rw.create<arith::ConstantIntOp>(*v, i64Type);
        } else {
          value = rw.create<arith::IndexCastOp>(i64Type, r.size.get<Value>());
        }
        value = rw.create<arith::MinSIOp>(i64Type, value, avgTileSize);
        tiles.emplace_back(
            rw.create<arith::IndexCastOp>(rw.getIndexType(), value));
      }
    }

    return tiles;
  }

  static int64_t getElementSize(Operation *op) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      if (auto inits = linalgOp.getDpsInits(); !inits.empty()) {
        if (auto t = getElementTypeOrSelf(inits[0].getType());
            t.isIntOrFloat()) {
          return std::max(1L, t.getIntOrFloatBitWidth() / 8L);
        }
      }
    }
    return 1L;
  }

  // TODO: Add more checks
  static bool canLowerToXeGPU(Operation *operation) {
    auto op = dyn_cast<linalg::LinalgOp>(operation);
    if (!op) {
      return false;
    }
    if (op.hasDynamicShape()) {
      return false;
    }

    auto checkOperand = [&](Value operand, bool isOutput = false) {
      ShapedType type;
      if (auto memref = dyn_cast<MemRefType>(operand.getType())) {
        type = memref;
      } else if (auto tensor = dyn_cast<RankedTensorType>(operand.getType())) {
        type = tensor;
      } else {
        return false;
      }

      auto shape = type.getShape();
      if (isOutput) {
        if (shape.size() != 2 || shape[0] * shape[1] < 16) {
          return false;
        }
      } else if (shape.size() > 2) {
        return false;
      }

      return true;
    };

    if (auto inits = op.getDpsInits();
        !inits.empty() && !checkOperand(inits[0], true)) {
      return false;
    }

    if (auto inputs = op.getDpsInputs();
        !std::all_of(inputs.begin(), inputs.end(),
                     [&](Value v) { return checkOperand(v); })) {
      return false;
    }

    return true;
  }

  void tileForallOps(OpRewriter &rw, func::FuncOp &fn) {
    auto wgSize = getWorkGroupSize(rw);
    fn.walk<WalkOrder::PreOrder>([&rw, wgSize](ForallOp loop) {
      if (!loop->getParentOfType<ForallOp>()) {
        tileForallOp(rw, loop, wgSize);
      }
      return WalkResult::skip();
    });
  }

  static void tileForallOp(OpRewriter &rw, ForallOp op, int64_t wgSize) {
    rw.loc = op.getLoc();
    rw.setInsertionPoint(op);
    OpFoldResult zero{rw.getConstant(0)};
    OpFoldResult one{rw.getConstant(1)};
    auto innerSteps = op.getMixedStep();
    auto outerBounds = op.getMixedUpperBound();
    SmallVector<OpFoldResult> outerSteps;
    auto count = innerSteps.size();

    { // Calculate outer steps
      SmallVector<int64_t> tiles;
      tiles.reserve(count);
      for (auto s : innerSteps) {
        if (auto v = getConstantIntValue(s)) {
          tiles.emplace_back(*v);
        } else {
          // TODO: Add support for dynamic sizes
          tiles.emplace_back(32);
        }
      }
      adjustTiles(wgSize, tiles);
      outerSteps.reserve(count);
      for (auto [s, b, t] : zip(innerSteps, outerBounds, tiles)) {
        if (auto sv = getConstantIntValue(s)) {
          auto step = *sv * t;
          if (auto bv = getConstantIntValue(b)) {
            step = std::min(step, *bv);
          }
          outerSteps.emplace_back(rw.getConstant(step));
        } else {
          outerSteps.emplace_back(
              rw.create<arith::MulIOp>(s.get<Value>(), rw.getConstant(t)));
        }
      }
    }

    auto outerLoop =
        rw.create<ForallOp>(op.getMixedLowerBound(), outerBounds, outerSteps,
                            op.getOutputs(), std::nullopt);
    rw.setInsertionPointToStart(outerLoop.getBody());
    SmallVector<OpFoldResult> innerBounds;
    SmallVector<Range> ranges;

    {
      auto idxType = rw.getIndexType();
      auto ctx = rw.getContext();
      auto minMap = AffineMap::get(
          /*dimCount=*/3, /*symbolCount=*/0,
          {getAffineDimExpr(0, ctx),
           getAffineDimExpr(1, ctx) - getAffineDimExpr(2, ctx)},
          rw.getContext());
      innerBounds.reserve(count);
      ranges.reserve(count);
      for (auto [i, u, s] : zip(outerLoop.getInductionVars(),
                                outerLoop.getMixedUpperBound(), outerSteps)) {
        OpFoldResult iub;
        auto cu = getConstantIntValue(u);
        auto cs = getConstantIntValue(s);
        if (cu && cs && (*cu % *cs == 0)) {
          iub = s;
        } else {
          Value vub = cu ? rw.getConstant(*cu) : u.get<Value>();
          Value vs = cs ? rw.getConstant(*cs) : s.get<Value>();
          iub = OpFoldResult(rw.create<affine::AffineMinOp>(
              idxType, minMap, ValueRange{vs, vub, i}));
        }
        innerBounds.emplace_back(iub);
        ranges.emplace_back(Range{i, iub, one});
      }
    }

    SmallVector<Value> innerOutputs;
    for (auto o : outerLoop.getRegionIterArgs()) {
      innerOutputs.emplace_back(rw.create<tensor::ExtractSliceOp>(o, ranges));
    }

    auto innerLoop =
        rw.create<ForallOp>(SmallVector<OpFoldResult>(count, zero), innerBounds,
                            innerSteps, innerOutputs, op.getMapping());
    SmallVector<Type> argTypes{innerLoop.getBody()->getArgumentTypes()};
    innerLoop.getRegion().takeBody(op.getRegion());
    for (auto [arg, type] :
         zip(innerLoop.getBody()->getArguments(), argTypes)) {
      arg.setType(type);
    }

    // Collect all users of the inner loop outputs
    llvm::SmallSet<Operation *, 4> outUsers;
    for (auto out : innerLoop.getRegionIterArgs()) {
      for (auto user : out.getUsers()) {
        outUsers.insert(user);
      }
    }

    // Replace the induction variables of the inner loop with the sum of the
    // outer and inner induction variables, but only in the operations, that
    // are not using the inner loop outputs, which are already sliced.
    rw.setInsertionPointToStart(innerLoop.getBody());
    for (auto [inIdx, outIdx] : llvm::zip(innerLoop.getInductionVars(),
                                          outerLoop.getInductionVars())) {
      auto newIdx = rw.create<arith::AddIOp>(inIdx, outIdx);
      outUsers.insert(newIdx);
      inIdx.replaceAllUsesExcept(newIdx, outUsers);
    }

    rw.setInsertionPointToStart(outerLoop.getTerminator().getBody());
    for (auto [i, o] :
         llvm::zip(innerLoop.getResults(), outerLoop.getRegionIterArgs())) {
      rw.create<tensor::ParallelInsertSliceOp>(i, o, ranges);
    }

    rw.replaceOp(op, outerLoop);
  }
};
} // namespace
