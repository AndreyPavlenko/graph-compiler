//===-- GpuOclRuntime.cpp - -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/ExecutionEngine/GPURuntime/GpuOclRuntime.h"
#include "gc/Error.h"
#include "gc/ExecutionEngine/Driver/Driver.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <memory>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>

#include "gc/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include <CL/cl_ext.h>

using namespace mlir;

constexpr char mlirAdd[] = R"mlir(
module @test {
  func.func @entry(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>) {
    %0 = bufferization.to_tensor %arg0 restrict : memref<32x32xf32>
    %1 = bufferization.to_tensor %arg1 restrict : memref<32x32xf32>
    %2 = tensor.empty() : tensor<32x32xf32>
    %3 = linalg.add ins(%1, %0 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    bufferization.materialize_in_destination %3 in restrict writable %arg2 : (tensor<32x32xf32>, memref<32x32xf32>) -> ()
    return
  }
}
)mlir";

static OwningOpRef<ModuleOp> parse(MLIRContext &ctx, const char *code) {
  std::unique_ptr<llvm::MemoryBuffer> ir_buffer =
      llvm::MemoryBuffer::getMemBuffer(code);
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(ir_buffer), llvm::SMLoc());
  return parseSourceFile<ModuleOp>(sourceMgr, &ctx);
}

TEST(GpuOclRuntime, TestAdd) {
  MLIRContext mlirCtx{gc::initCompilerAndGetDialects()};
  OwningOpRef<ModuleOp> module = parse(mlirCtx, mlirAdd);
  auto rt = gcGetOrReport(gc::gpu::OclRuntime::get());

  auto size = 32 * 32;
  float cpuBuf1[size];
  float cpuBuf2[size];
  int64_t shape[] = {32, 32};
  int64_t strides[] = {32, 1};

  for (int i = 0; i < size; i++) {
    cpuBuf1[i] = 2.0;
  }

  auto buf0 = gcGetOrReport(rt.usmNewDev<float>(size));
  auto buf1 = gcGetOrReport(rt.usmNewDev<float>(size));
  auto buf2 = gcGetOrReport(rt.usmNewShared<float>(size));
  assert(buf0 && buf1 && buf2);

  auto queue = gcGetOrReport(rt.createQueue());
  gc::gpu::OclContext ctx(queue);
  assert(rt.usmCpy(ctx, cpuBuf1, buf0, size));
  assert(rt.usmCpy(ctx, cpuBuf1, buf1, size));

  gc::gpu::OclModuleBuilder modBuilder(module);
  auto mod = gcGetOrReport(modBuilder.build(rt));
  auto args = mod->args<27>();
  args.add(&buf0, 2, shape, strides);
  args.add(&buf1, 2, shape, strides);
  args.add(&buf2, 2, shape, strides);
  mod->exec(ctx, args);

  assert(rt.usmCpy(ctx, buf2, cpuBuf2, size));
  ctx.finish();
  gcGetOrReport(rt.releaseQueue(queue));
  assert(rt.usmFree(buf0));
  assert(rt.usmFree(buf1));

  for (int i = 0; i < size; i++) {
    // std::cout << buf2[i] << " ";
    assert(buf2[i] == 4.0f);
  }
  // std::cout << "\n";

  for (int i = 0; i < size; i++) {
    // std::cout << cpuBuf2[i] << " ";
    assert(cpuBuf2[i] == 4.0f);
  }
  // std::cout << "\n";
  assert(rt.usmFree(buf2));
}