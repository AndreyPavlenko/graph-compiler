//===-- OclGpuRuntimeWrappers.h - Ocl GPU wrappers --------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_OCLGPURUNTIMEWRAPPERS_H
#define GC_OCLGPURUNTIMEWRAPPERS_H

#define GC_OCL_GPU_MALLOC "gcOclGpuMaloc"
#define GC_OCL_GPU_DEALLOC "gcOclGpuDealloc"
#define GC_OCL_GPU_MEMCPY "gcOclGpuMemcpy"
#define GC_OCL_GPU_KERNEL_CREATE "gcOclGpuKernelCreate"
#define GC_OCL_GPU_KERNEL_DESTROY "gcOclGpuKernelDestroy"
#define GC_OCL_GPU_KERNEL_LAUNCH "gcOclGpuKernelLaunch"

#ifndef GC_OCL_GPU_DEF_ONLY

#include "gc/Utils.h"
#include <CL/cl.h>

struct GcOclGpuKernel;

struct GcOclGpuContext {
  cl_command_queue queue;

  // The following fields are used only if the queue has the
  // CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE property.
  cl_uint waitListLen;
  const cl_event *waitList;
  cl_event *lastEvent;
};

extern "C" {
GC_DLL_EXPORT void *gcOclGpuMaloc(GcOclGpuContext *ctx, size_t size);

GC_DLL_EXPORT void gcOclGpuDealloc(GcOclGpuContext *ctx, void *ptr);

GC_DLL_EXPORT void gcOclGpuMemcpy(GcOclGpuContext *ctx, void *src, void *dst,
                                  size_t size);

GC_DLL_EXPORT GcOclGpuKernel *
gcOclGpuKernelCreate(GcOclGpuContext *ctx, const char *spirv, const char *name,
                     const size_t *globalSize, const size_t *localSize,
                     size_t argNum, const size_t *argSize);

GC_DLL_EXPORT void gcOclGpuKernelDestroy(size_t count, GcOclGpuKernel *kernels);

GC_DLL_EXPORT void gcOclGpuKernelLaunch(GcOclGpuContext *ctx,
                                        GcOclGpuKernel *kernel, ...);
}
#else
#undef GC_OCL_GPU_DEF_ONLY
#endif
#endif
