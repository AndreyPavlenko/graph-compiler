//===-- Utils.h - Utils -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_UTILS_H
#define GC_UTILS_H

#if defined _WIN32 || defined __CYGWIN__
#define GC_DLL_EXPORT __declspec(dllexport)
#else
#define GC_DLL_EXPORT __attribute__((visibility("default")))
#endif

#ifdef _NDEBUG
#define gcLogD(...)
#define gcLogE(...)
#else
#include <iostream>

static void _gcLogPrint(std::ostream &stream) { stream << std::endl; }

template <typename T, typename... Args>
static void _gcLogPrint(std::ostream &stream, T first, Args... args) {
  stream << first;
  _gcLogPrint(stream, args...);
}

template <typename... Args>
static void _gcLog(std::ostream &stream, const char *pref, const char *fileName,
                   int lineNum, Args... args) {
  stream << pref << " [" << fileName << ":" << lineNum << "] ";
  _gcLogPrint(stream, args...);
}

#define gcLog(stream, pref, ...)                                               \
  _gcLog(stream, pref, __FILE__, __LINE__, __VA_ARGS__)
#define gcLogD(...) gcLog(std::cout, "[DEBUG]", __VA_ARGS__)
#define gcLogE(...) gcLog(std::cerr, "[ERROR]", __VA_ARGS__)
#endif

#endif
