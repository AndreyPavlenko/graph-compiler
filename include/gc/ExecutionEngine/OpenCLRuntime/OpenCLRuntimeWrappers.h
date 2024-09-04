//===-- OpenCLRuntimeWrappers.h ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined _WIN32 || defined __CYGWIN__
#define OCL_RUNTIME_EXPORT __declspec(dllexport)
#else
#define OCL_RUNTIME_EXPORT __attribute__((visibility("default")))
#endif

#include <CL/cl.h>

extern "C" {
    OCL_RUNTIME_EXPORT void gpuSetThreadLocalQueue(cl_command_queue queue);
}
