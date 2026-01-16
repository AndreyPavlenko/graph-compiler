//===--------- Vectorize.cpp - Vectorize structured ops ----------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Utils/Transform.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::gc;

namespace mlir::gc {
#define GEN_PASS_DECL_GPUKERNELOUTLINE
#define GEN_PASS_DEF_GPUKERNELOUTLINE
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

struct GpuKernelOutline final
    : gc::impl::GpuKernelOutlineBase<GpuKernelOutline> {

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    // pm.addPass(createForallToParallelLoopPass());
    // pm.addNestedPass<func::FuncOp>(createGpuMapParallelLoopsPass());
    // pm.addNestedPass<func::FuncOp>(createConvertParallelLoopToGpuPass());

    // Find all scf.forall ops marked with GC_ATTR_THREADS and outline them
    // into GPU kernels.
    IRRewriter rw(ctx);
    getOperation().walk([&](scf::ForallOp forallOp) {
      auto kernelName = dyn_cast_if_present<StringAttr>(
          forallOp->getAttr(GC_ATTR_KERNEL_NAME));
      if (!kernelName)
        return WalkResult::skip();

      rw.setInsertionPoint(forallOp);
      scf::ParallelOp parallelOp;

      if (failed(scf::forallToParallelLoop(rw, forallOp, &parallelOp))) {
        signalPassFailure();
        return WalkResult::interrupt();
      }

      parallelOp.getOperation()->setAttr(GC_ATTR_KERNEL_NAME, kernelName);
      return WalkResult::skip();
    });
  }
};

} // namespace
