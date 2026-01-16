//===-- GpuTilingAndFusion.cpp - DESC ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "./GpuUtils.h"
#include "gc/Utils/Transform.h"

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

#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace mlir::gc;
using namespace mlir::scf;

namespace mlir::gc {
#define GEN_PASS_DECL_GPUTILINGANDFUSION
#define GEN_PASS_DEF_GPUTILINGANDFUSION
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

// struct TileAndFuseLinalgOpsPattern : public RewritePattern {
//   explicit TileAndFuseLinalgOpsPattern(GpuTilingAndFusion *pass)
//       : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1,
//                        &pass->getContext()) {}

//   LogicalResult matchAndRewrite(Operation *op,
//                                 PatternRewriter &rewriter) const override {
//     return success();
//   }

// private:
//   SCFTileAndFuseOptions opts;

//   SmallVector<OpFoldResult> computeTileSizes(OpBuilder &builder,
//                                              Operation *op) {
//     auto ti = dyn_cast<TilingInterface>(op);
//     if (!ti) {
//       return {};
//     }

//     IRRewriter rw(builder);
//     auto vectorWidth = getVectorWidth(rw);
//     auto sgSize = getSubGroupSize(rw);
//     auto wgSize = getWorkGroupSize(rw);
//     auto lanesAttr = rw.getStringAttr(GC_LANES_ATTR);
//     auto threadsAttr = rw.getStringAttr(GC_THREADS_ATTR);

//     rw.setInsertionPoint(op);
//     auto itTypes = ti.getLoopIteratorTypes();
//     auto itDomains = ti.getIterationDomain(builder);
//     assert(itTypes.size() == itDomains.size());

//     SmallVector<int64_t> sizes;
//     int64_t maxSize = 0;
//     int64_t numIterations = 1;
//     for (auto [t, r] : zip(itTypes, itDomains)) {
//       if (t == utils::IteratorType::parallel) {
//         if (auto v = getConstantIntValue(r.size)) {
//           numIterations *= *v;
//           sizes.emplace_back(*v);
//           maxSize = std::max(maxSize, *v);
//         } else {
//           gcLogE("Dynamic tiles are not supported!");
//           return {};
//         }
//       }
//     }

//     // TODO: Analyse the graph of suppliers to be fused and adjust the
//     // value.
//     int64_t opsPerLane = 4;
//     int64_t totalSize = vectorWidth * sgSize * opsPerLane;
//     if (totalSize > numIterations) {
//       totalSize = std::max(numIterations / vectorWidth * vectorWidth, 1L);
//     }

//     SmallVector<int64_t> tiles = sizes;
//     adjustTiles(totalSize, tiles);

//     // If the tiles are equal to the sizes, split the largest tile.
//     if (tiles == sizes) {
//       auto tile = findFactor(maxSize, maxSize / 2);

//       if (tile == maxSize) {
//         // Find another size, that can be split
//         auto another = maxSize;
//         sort(sizes, std::greater<>());
//         for (auto s : sizes) {
//           if (s != maxSize && (tile = findFactor(s, s / 2)) != s) {
//             another = s;
//             break;
//           }
//         }
//         if (another == maxSize) {
//           tile = 1;
//           // Find the smallest size that is not 1
//           for (auto s : reverse(sizes)) {
//             if (s != 1) {
//               maxSize = s;
//               break;
//             }
//           }
//         } else {
//           maxSize = another;
//         }
//       }

//       for (auto &t : tiles) {
//         if (t == maxSize) {
//           t = tile;
//           break;
//         }
//       }
//     }

//     unsigned counter = 0;
//     int64_t numThreads = numIterations * sgSize / vectorWidth / opsPerLane;
//     SmallVector<OpFoldResult> result;
//     result.reserve(itDomains.size());

//     for (auto [t, r] : zip(itTypes, itDomains)) {
//       if (t == utils::IteratorType::parallel) {
//         int64_t s = tiles[counter++];
//         numThreads /= s;
//         result.emplace_back(rw.createConstant(s));
//       } else {
//         result.emplace_back(rw.createConstant(0L));
//       }
//     }

//     numThreads = std::min(std::max(numThreads, 1L), wgSize);
//     SmallVector<int64_t> threads = sizes;
//     adjustTiles(numThreads, threads, false);
//     op->setDiscardableAttr(lanesAttr, rw.getI64IntegerAttr(sgSize));
//     op->setDiscardableAttr(threadsAttr, rw.getI64ArrayAttr(threads));
//     return result;
//   }
// };

struct GpuTilingAndFusion final
    : gc::impl::GpuTilingAndFusionBase<GpuTilingAndFusion> {
  friend struct TileAndFuseLinalgOpsPattern;
  explicit GpuTilingAndFusion()
      : GpuTilingAndFusion(GpuTilingAndFusionOptions{}) {}
  explicit GpuTilingAndFusion(const GpuTilingAndFusionOptions &opts)
      : GpuTilingAndFusionBase(opts) {}

  void runOnOperation() override {
    auto fn = getOperation();
    if (fn.isExternal()) {
      return;
    }

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    OpRewriter rw(fn);
    tileAndFuseLinalgOps(rw, fn);
  }

private:
  void tileAndFuseLinalgOps(OpRewriter &rw, func::FuncOp fn) {
    int nameCounter = 0;
    auto kernelNameBase = fn.getName().str() + "_kernel";
    DevAttrs devAttrs(fn);
    int64_t vectorWidth = devAttrs.getVectorWidth().value_or(16);
    int64_t wgSize = devAttrs.getMaxWgSize().value_or(1024);
    auto sgSizes = devAttrs.getSgSizes().value_or(SmallVector<size_t>{32});
    int64_t maxSgSize = *llvm::max_element(sgSizes);
    SCFTileAndFuseOptions opts;
    opts.tilingOptions.setTileSizeComputationFunction(
        [&](OpBuilder &builder, Operation *op) -> SmallVector<OpFoldResult> {
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
                gcLogE("Dynamic tiles are not supported!");
                return {};
              }
            }
          }

          auto kernelName = kernelNameBase;
          if (++nameCounter != 1)
            kernelName += std::to_string(nameCounter);
          op->setDiscardableAttr(GC_ATTR_KERNEL_NAME,
                                 createAttr(rw.getContext(), kernelName));
          KernelAttrs kernelAttrs(fn, kernelName);
          int64_t sgSize = kernelAttrs.getSgSize().value_or(maxSgSize);
          // TODO: Analyse the graph of suppliers to be fused and adjust the
          // value.
          int64_t opsPerLane = 4;
          SmallVector<int64_t> tiles;

          if (auto staticTiles = kernelAttrs.getTiles()) {
            for (auto t : *staticTiles) {
              tiles.push_back(static_cast<int64_t>(t));
            }
          } else {
            int64_t totalSize = vectorWidth * sgSize * opsPerLane;
            if (totalSize > numIterations) {
              totalSize =
                  std::max(numIterations / vectorWidth * vectorWidth, 1L);
            }
            tiles = sizes;
            adjustTiles(totalSize, tiles);

            // If the tiles are equal to the sizes, split the largest tile.
            if (tiles == sizes) {
              auto tile = findFactor(maxSize, maxSize / 2);

              if (tile == maxSize) {
                // Find another size, that can be split
                auto another = maxSize;
                sort(sizes, std::greater<>());
                for (auto s : sizes) {
                  if (s != maxSize && (tile = findFactor(s, s / 2)) != s) {
                    another = s;
                    break;
                  }
                }
                if (another == maxSize) {
                  tile = 1;
                  // Find the smallest size that is not 1
                  for (auto s : reverse(sizes)) {
                    if (s != 1) {
                      maxSize = s;
                      break;
                    }
                  }
                } else {
                  maxSize = another;
                }
              }

              for (auto &t : tiles) {
                if (t == maxSize) {
                  t = tile;
                  break;
                }
              }
            }

            kernelAttrs.setTiles(
                SmallVector<size_t>(tiles.begin(), tiles.end()));
          }

          unsigned counter = 0;
          int64_t numThreads =
              numIterations * sgSize / vectorWidth / opsPerLane;
          SmallVector<OpFoldResult> result;
          result.reserve(itDomains.size());

          for (auto [t, r] : zip(itTypes, itDomains)) {
            if (t == utils::IteratorType::parallel) {
              int64_t s = tiles[counter++];
              numThreads /= s;
              result.emplace_back(rw.createConstant(s));
            } else {
              result.emplace_back(rw.createConstant(0L));
            }
          }

          if (!kernelAttrs.getThreads()) {
            numThreads = std::min(std::max(numThreads, 1L), wgSize);
            SmallVector<int64_t> threads = sizes;
            adjustTiles(numThreads, threads, false);
            kernelAttrs.setThreads(
                SmallVector<size_t>(threads.begin(), threads.end()));
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
          if (isOpDependsOnResult<0>(isMatmulOp, candidateSliceOp) &&
              isOperandDependsOnOp(isMatmulOp, candidateSliceOp)) {
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
            if (auto loop = dyn_cast<ForallOp>(repl.getDefiningOp())) {
              replaceEmptySlices(rw, loop);
              if (auto v = toReplace->getDiscardableAttr(GC_ATTR_KERNEL_NAME))
                loop->setDiscardableAttr(GC_ATTR_KERNEL_NAME, v);
            }
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
      if (!linalgOp.hasOnlyProjectedPermutations()) {
        return WalkResult::skip();
      }
      if (auto parentLoop = linalgOp->getParentOfType<ForallOp>();
          parentLoop && parentLoop->hasAttr(GC_ATTR_KERNEL_NAME)) {
        return WalkResult::skip();
      }

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

      return WalkResult::skip();
    });
    return last;
  }

  // If a slice inside the loop is created from an external empty tensor and the
  // tensor is not passed to the loop's shared_outs, but referenced directly,
  // replace the slice with an empty tensor of the same size.
  static void replaceEmptySlices(OpRewriter &rw, ForallOp loop) {
    loop.walk([&](tensor::ExtractSliceOp slice) {
      if (auto empty = slice.getSource().getDefiningOp<tensor::EmptyOp>();
          empty && empty->getParentOfType<ForallOp>() != loop) {
        auto type = slice.getType();
        rw.setInsertionPointAfter(slice);
        SmallVector<Value> dynDims;
        for (int64_t i = 0, r = type.getRank(); i < r; ++i) {
          if (type.isDynamicDim(i)) {
            dynDims.push_back(rw.create<tensor::DimOp>(slice, i));
          }
        }
        rw.replaceOp(slice, rw.create<tensor::EmptyOp>(type.getShape(),
                                                       type.getElementType(),
                                                       dynDims));
      }
    });
  }
};
} // namespace
