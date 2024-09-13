//===- Pipeline.cpp - Graph Compiler all-in-one pipeline --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include <climits>

#ifdef GC_USE_IMEX
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include <imex/Conversion/Passes.h>
#include <imex/Transforms/Passes.h>
#endif

#include "gc/Dialect/CPURuntime/Transforms/CPURuntimePasses.h"
#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#ifdef GC_HAS_ONEDNN_DIALECT
#include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.h"
#endif
#include "gc/Transforms/Microkernel/MicrokernelPasses.h"
#include "gc/Transforms/Passes.h"

namespace mlir::gc {

void populateCleanUpPasses(mlir::OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

// linalg + linalgX + tensor
void populateFrontendPasses(mlir::OpPassManager &pm) {
#ifdef GC_HAS_ONEDNN_DIALECT
  pm.addPass(createConvertOneDNNGraphToLinalg());
#endif
}

// scf + arith + math + vector + tensor + linalg.brgemm + tensor.pack/unpack
void populateTensorPasses(mlir::OpPassManager &pm) {
  // todo: padding propagation pass
  // todo: layout propagation pass
  // todo: tensor constant propagation pass
  // linalg.matmul lowering to (scf.loop + linalg.brgemm) pass
  pm.addNestedPass<func::FuncOp>(createDeepTileContractionOp());

  // Fine-grain fusion pass
  pm.addNestedPass<func::FuncOp>(createIterativeTilingAndFusion());
  // todo: fine-grain fusion pass
  pm.addNestedPass<func::FuncOp>(
      mlir::microkernel::createConvertLinalgToMicrokernel());
  // todo: lower linalg to arith/math on virtual vector pass

  // REMOVE this pass after the above passes are added. Currently we add this
  // pass to make the pipeline work properly
  pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
  // copied from tpp project
  pm.addNestedPass<func::FuncOp>(createDecomposeAggregatedOps());
  // fold useless tensor operation pass
  pm.addPass(createFoldTensorOperation());
  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addPass(createControlFlowSinkPass());
  populateCleanUpPasses(pm);
}

// scf + arith + math + vector + tensor + linalg.brgemm
void populateVectorPasses(mlir::OpPassManager &pm) {
  // Do promotion for math / arith ops
  pm.addNestedPass<func::FuncOp>(math::createMathLegalizeToF32());
  // sourceTypeStrs can be extended
  arith::ArithEmulateUnsupportedFloatsOptions options;
  std::array<std::string, 1> typeStr = {"bf16"};
  options.sourceTypeStrs = typeStr;
  options.targetTypeStr = "f32";
  pm.addNestedPass<func::FuncOp>(
      arith::createArithEmulateUnsupportedFloats(options));
  // Bf16 cast elimilation pass
  pm.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());
  // oneDNN graph spec
  pm.addNestedPass<func::FuncOp>(arith::createArithExpandOpsPass());
  // todo: lower to physical vector pass, device dependent pass
  populateCleanUpPasses(pm);
}

// scf + arith + math + vector + memref + linalg.brgemm
void populateBufferizationPasses(mlir::OpPassManager &pm) {
  // The flow follows https://mlir.llvm.org/docs/Bufferization/#overview
  pm.addPass(bufferization::createEmptyTensorEliminationPass());
  bufferization::OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  options.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::IdentityLayoutMap);
  pm.addPass(bufferization::createOneShotBufferizePass(options));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(bufferization::createBufferHoistingPass());
  pm.addNestedPass<func::FuncOp>(bufferization::createBufferLoopHoistingPass());
  // todo: buffer schedule pass
  // todo: Need to improve this pass to support nested parallel.
  bufferization::BufferResultsToOutParamsOpts opt{};
  opt.hoistStaticAllocs = true;
  pm.addPass(bufferization::createBufferResultsToOutParamsPass(opt));
  pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
  pm.addNestedPass<func::FuncOp>(bufferization::createPromoteBuffersToStackPass(
      /*maxAllocSizeInBytes*/ UINT_MAX,
                              /*maxRankOfAllocatedMemRef*/ 8));
  mlir::bufferization::BufferDeallocationPipelineOptions deallocOption;
  bufferization::buildBufferDeallocationPipeline(pm, deallocOption);
  pm.addPass(createBufferizationToMemRefPass());
  populateCleanUpPasses(pm);
}

// scf + arith + math + vector + memref + func/microkernel
void populateMicroKernelPasses(mlir::OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(mlir::microkernel::createExpandMicrokernel());
  pm.addPass(mlir::microkernel::createEarlyDispatchMicrokernel());
  pm.addPass(mlir::microkernel::createConvertMicrokernelToDnnlFunc());
  pm.addPass(mlir::microkernel::createMergeBranchMicrokernelContext());
  pm.addPass(mlir::microkernel::createMicrokernelInvariantCodeMotion());
  populateCleanUpPasses(pm);
}

void populateCPURuntimePasses(mlir::OpPassManager &pm) {
  // todo: flatten nested parallel pass to support coarse-grain usion
  // remove this pass after we add FlattenNestedParallel

  pm.addPass(createSinkOpIntoInnerLoop());
  pm.addPass(createMergeNestedForall());
  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addPass(createControlFlowSinkPass());
  pm.addPass(createForallToParallelLoopPass());
  pm.addPass(createParallelLoopFusionPass());
  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addNestedPass<func::FuncOp>(createConvertMemRefToCPURuntime());
  pm.addPass(createConvertSCFToOpenMPPass());
  populateCleanUpPasses(pm);
}

void populateLoweringToLLVMPasses(mlir::OpPassManager &pm) {
  pm.addPass(createLowerAffinePass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(cpuruntime::createCPURuntimeToLLVM());
  pm.addPass(createConvertOpenMPToLLVMPass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createConvertMathToLibmPass());
  pm.addNestedPass<func::FuncOp>(createArithToLLVMConversionPass());
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(createSymbolDCEPass());
}

void populateLLVMPasses(mlir::OpPassManager &pm) {
  pm.addPass(memref::createExpandOpsPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  populateLoweringToLLVMPasses(pm);
}

void populateCPUPipeline(mlir::OpPassManager &pm) {
  // verify the target description attribute
  pm.addPass(createVerifyTargetDescription());
  // front-end, oneDNN graph dialect
  populateFrontendPasses(pm);
  // middle-end, LinalgX/Linalg/tensor dialects
  populateTensorPasses(pm);
  // middle-end, arith/math/vector dialects
  populateVectorPasses(pm);
  // back-end, arith/math/vector/memref dialects
  populateBufferizationPasses(pm);
  // REMOVE this pass after the TensorPasses are added. Currently we add this
  // pass to make the pipeline work properly
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  populateMicroKernelPasses(pm);
  populateCPURuntimePasses(pm);
  // back-end, llvm dialect
  populateLLVMPasses(pm);
}

void registerCPUPipeline() {
  PassPipelineRegistration<>("gc-cpu-pipeline",
                             "The CPU pipeline for Graph Compiler",
                             populateCPUPipeline);
}

#ifdef GC_USE_IMEX
void populateGPUPipeline(mlir::OpPassManager &pm) {
  IterativeTilingAndFusionOptions tilingOpts;
  std::string tileSize = "matmul:{16,16}";
  tilingOpts.defaultTileSize = tileSize;
  pm.addNestedPass<func::FuncOp>(createIterativeTilingAndFusion(tilingOpts));

  pm.addPass(bufferization::createEmptyTensorEliminationPass());
  pm.addPass(bufferization::createEmptyTensorToAllocTensorPass());

  bufferization::OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  options.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::IdentityLayoutMap);
  pm.addPass(bufferization::createOneShotBufferizePass(options));

  pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
  pm.addNestedPass<func::FuncOp>(
      bufferization::createFinalizingBufferizePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
  pm.addPass(memref::createExpandReallocPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(bufferization::createOwnershipBasedBufferDeallocationPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(bufferization::createBufferDeallocationSimplificationPass());
  pm.addPass(bufferization::createLowerDeallocationsPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createBufferizationToMemRefPass());

  pm.addNestedPass<func::FuncOp>(createForallToParallelLoopPass());
  pm.addNestedPass<func::FuncOp>(createLinalgToXeGPU({16, 1, {8, 16, 16}}));
  // pm.addNestedPass<func::FuncOp>(createConvertLinalgToParallelLoopsPass());
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addPass(xegpu::createXeGPUFoldAliasOps());
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addNestedPass<func::FuncOp>(createGpuMapParallelLoopsPass());
  pm.addNestedPass<func::FuncOp>(createParallelLoopToGpuPass());
  pm.addPass(createAddRemoveGpuAddressSpace());
  pm.addNestedPass<func::FuncOp>(
      imex::createInsertGPUAllocsPass("opencl", false, false));
  pm.addPass(createAddRemoveGpuAddressSpace({true}));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(memref::createNormalizeMemRefsPass());
  pm.addPass(createGpuKernelOutliningPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(imex::createSetSPIRVCapabilitiesPass());
  pm.addNestedPass<gpu::GPUModuleOp>(
      imex::createSetSPIRVAbiAttributePass("opencl"));
  pm.addPass(createLowerAffinePass());
  pm.addPass(imex::createVectorLinearizePass());
  pm.addNestedPass<gpu::GPUModuleOp>(imex::createConvertXeGPUToVCPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(imex::createBF16ToGPUPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertFuncToSPIRVPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertVectorToSPIRVPass());
  pm.addPass(imex::createConvertGPUXToSPIRVPass());
  pm.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVLowerABIAttributesPass());
  pm.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVUpdateVCEPass());
  pm.addNestedPass<func::FuncOp>(LLVM::createRequestCWrappersPass());
  pm.addPass(imex::createSerializeSPIRVPass());
  pm.addPass(createConvertVectorToSCFPass());
  // pm.addPass(imex::createConvertGPUToGPUXPass());
  // pm.addPass(createPrintIRPass());
  // // pm.addPass(createAddRemoveGpuAddressSpace({true}));
  // pm.addPass(createGpuToGpuOcl());
  // pm.addPass(createPrintIRPass());
  // pm.addPass(imex::createConvertGPUToGPUXPass());

  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createAddRemoveGpuAddressSpace({true}));
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertMathToLLVMPass());
  // pm.addPass(imex::createConvertGPUXToLLVMPass());

  // pm.addPass(createPrintIRPass());
  pm.addPass(createGpuToGpuOcl());
  // pm.addPass(createPrintIRPass());

  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

void registerGPUPipeline() {
  PassPipelineRegistration<>("gc-gpu-pipeline",
                             "The GPU pipeline for Graph Compiler",
                             populateGPUPipeline);
}
#endif // GC_USE_IMEX
} // namespace mlir::gc