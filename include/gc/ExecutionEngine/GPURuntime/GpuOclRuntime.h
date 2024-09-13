//===-- GpuOclRuntime.h - GPU OpenCL runtime --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_GPUOCLRUNTIME_H
#define GC_GPUOCLRUNTIME_H

namespace mlir::gc::gpu {
constexpr char GPU_OCL_MALLOC[] = "gcGpuOclMalloc";
constexpr char GPU_OCL_DEALLOC[] = "gcGpuOclDealloc";
constexpr char GPU_OCL_MEMCPY[] = "gcGpuOclMemcpy";
constexpr char GPU_OCL_KERNEL_CREATE[] = "gcGpuOclKernelCreate";
constexpr char GPU_OCL_KERNEL_DESTROY[] = "gcGpuOclKernelDestroy";
constexpr char GPU_OCL_KERNEL_LAUNCH[] = "gcGpuOclKernelLaunch";
constexpr char GPU_OCL_MOD_DESTRUCTOR[] = "gcGpuOclModuleDestructor";
} // namespace mlir::gc::gpu

#ifndef GC_GPU_OCL_CONST_ONLY
#include <cstdarg>
#include <unordered_set>
#include <vector>

#include <CL/cl.h>

#include <llvm/ADT/SmallString.h>

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::gc::gpu {
struct OclContext;
struct OclModule;
struct OclModuleBuilder;

struct OclRuntime {
  cl_context context;
  cl_device_id device;

  // Returns the available Intel GPU device ids.
  [[nodiscard]] static llvm::Expected<SmallVector<cl_device_id, 2>>
  gcIntelDevices(size_t max = std::numeric_limits<size_t>::max());

  [[nodiscard]] static llvm::Expected<OclRuntime> get();

  [[nodiscard]] static llvm::Expected<OclRuntime> get(cl_device_id device);

  [[nodiscard]] static llvm::Expected<OclRuntime> get(cl_command_queue queue);

  [[nodiscard]] static llvm::Expected<OclRuntime> get(cl_context context,
                                                      cl_device_id device);

  static bool isOutOfOrder(cl_command_queue queue);

  [[nodiscard]] llvm::Expected<cl_command_queue>
  createQueue(bool outOfOrder = false) const;

  [[nodiscard]] llvm::Expected<bool> releaseQueue(cl_command_queue queue) const;

  [[nodiscard]] llvm::Expected<void *> usmAllocDev(size_t size) const;

  [[nodiscard]] llvm::Expected<void *> usmAllocShared(size_t size) const;

  [[nodiscard]] llvm::Expected<bool> usmFree(const void *ptr) const;

  [[nodiscard]] llvm::Expected<bool> usmCpy(OclContext *ctx, const void *src,
                                            void *dst, size_t size) const;

  [[nodiscard]] llvm::Expected<bool> usmCpy(OclContext &ctx, const void *src,
                                            void *dst, size_t size) const {
    return usmCpy(&ctx, src, dst, size);
  }

  template <typename T>
  [[nodiscard]] llvm::Expected<T *> usmNewDev(size_t size) const {
    auto expected = usmAllocDev(size * sizeof(T));
    if (expected) {
      return static_cast<T *>(*expected);
    }
    return expected.takeError();
  }

  template <typename T>
  [[nodiscard]] llvm::Expected<T *> usmNewShared(size_t size) const {
    auto expected = usmAllocShared(size * sizeof(T));
    if (expected) {
      return static_cast<T *>(*expected);
    }
    return expected.takeError();
  }

  template <typename T>
  [[nodiscard]] llvm::Expected<bool> usmCpy(OclContext &ctx, const T *src,
                                            T *dst, size_t size) const {
    return usmCpy(ctx, static_cast<const void *>(src), static_cast<void *>(dst),
                  size * sizeof(T));
  }

  // Use with caution! This is safe to check validity of USM, but may be false
  // positive for any other kinds.
  bool isUsm(const void *ptr) const;

  bool operator==(const OclRuntime &other) const {
    return context == other.context && device == other.device;
  }

private:
  struct Ext;
  struct Exports;
  friend OclContext;
  friend OclModuleBuilder;
  explicit OclRuntime(cl_context context, cl_device_id device, const Ext *ext)
      : context(context), device(device), ext(ext) {}
  const Ext *ext;
};
} // namespace mlir::gc::gpu
template <> struct std::hash<const mlir::gc::gpu::OclRuntime> {
  std::size_t
  operator()(const mlir::gc::gpu::OclRuntime &runtime) const noexcept {
    return std::hash<cl_context>()(runtime.context) ^
           std::hash<cl_device_id>()(runtime.device);
  }
}; // namespace std
namespace mlir::gc::gpu {

struct OclContext {
  cl_command_queue const queue;
  // Preserve the execution order. This is required in case of out-of-order
  // execution (CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE). When the execution
  // is completed, the 'lastEvent' field contains the event of the last enqueued
  // command. If this field is false, 'waitList' is ignored.
  const bool preserveOrder;
  cl_event lastEvent;

  explicit OclContext(cl_command_queue queue, cl_uint waitListLen = 0,
                      cl_event *waitList = nullptr)
      : OclContext(queue, OclRuntime::isOutOfOrder(queue), waitListLen,
                   waitList) {}

  explicit OclContext(cl_command_queue queue, bool preserveOrder,
                      cl_uint waitListLen, cl_event *waitList)
      : queue(queue), preserveOrder(preserveOrder), lastEvent(nullptr),
        waitListLen(preserveOrder ? waitListLen : 0),
        waitList(preserveOrder ? waitList : nullptr), runtime(nullptr),
        clPtrs(nullptr) {
    assert(!OclRuntime::isOutOfOrder(queue) || preserveOrder);
    assert(preserveOrder || (waitListLen == 0 && waitList == nullptr));
  }

  void finish();

private:
  friend OclModule;
  friend OclRuntime;
  friend OclRuntime::Exports;
  cl_uint waitListLen;
  cl_event *waitList;
  const OclRuntime *runtime;
  std::pmr::unordered_set<void *> *clPtrs;

  void setLastEvent(cl_event event) {
    lastEvent = event;
    if (event) {
      waitListLen = 1;
      waitList = &lastEvent;
    } else {
      waitListLen = 0;
      waitList = nullptr;
    }
  }
};

struct OclModule {
  static constexpr int64_t ZERO = 0;
  static constexpr auto ZERO_PTR = const_cast<int64_t *>(&ZERO);

  // The main function arguments in the following format -
  // https://mlir.llvm.org/docs/TargetLLVMIR/#c-compatible-wrapper-emission.
  // Note: the values are not copied, only the pointers are stored!
  template <unsigned N> struct Args {

    void add(void **alignedPtr, size_t rank, const int64_t *shape,
             const int64_t *strides, bool isUsm = true) {
      add(alignedPtr, alignedPtr, ZERO_PTR, rank, shape, strides, isUsm);
    }

    void add(void **allocatedPtr, void **alignedPtr, const int64_t *offset,
             size_t rank, const int64_t *shape, const int64_t *strides,
             bool isUsm = true) {
#ifndef NDEBUG
      assert(!isUsm || runtime->isUsm(*alignedPtr));
      // It's recommended to have at least 16-byte alignment
      assert(reinterpret_cast<std::uintptr_t>(*alignedPtr) % 16 == 0);
#endif

      args.emplace_back(allocatedPtr);
      args.emplace_back(alignedPtr);
      args.emplace_back(const_cast<int64_t *>(offset));
      for (size_t i = 0; i < rank; i++) {
        args.emplace_back(const_cast<int64_t *>(&shape[i]));
      }
      for (size_t i = 0; i < rank; i++) {
        args.emplace_back(const_cast<int64_t *>(&strides[i]));
      }
      if (!isUsm) {
        clPtrs.insert(alignedPtr);
      }
    }

    template <typename T>
    void add(T **alignedPtr, size_t rank, const int64_t *shape,
             const int64_t *strides, bool isUsm = true) {
      add(reinterpret_cast<void **>(alignedPtr), rank, shape, strides, isUsm);
    }

    template <typename T>
    void add(T **allocatedPtr, T **alignedPtr, const int64_t *offset,
             size_t rank, const int64_t *shape, const int64_t *strides,
             bool isUsm = true) {
      add(reinterpret_cast<void **>(allocatedPtr),
          reinterpret_cast<void **>(alignedPtr), offset, rank, shape, strides,
          isUsm);
    }

    void clear() {
      args.clear();
      clPtrs.clear();
    }

  private:
    friend OclModule;
    SmallVector<void *, N + 3> args;
    // Contains the pointers of all non-USM arguments. It's expected, that the
    // arguments are either USM or CL pointers and most probably are USM, thus,
    // in most cases, this set will be empty.
    std::pmr::unordered_set<void *> clPtrs;
#ifdef NDEBUG
    explicit Args(){};
#else
    const OclRuntime *runtime;
    explicit Args(const OclRuntime *runtime) : runtime(runtime) {}
#endif
  };

  using MainFunc = void (*)(void **);

  explicit OclModule(const OclRuntime &runtime,
                     std::unique_ptr<ExecutionEngine> engine, MainFunc main)
      : runtime(runtime), engine(std::move(engine)), main(main) {}

#ifdef NDEBUG
  template <unsigned N = 64> Args<N> args() const { return Args<N>(); }
#else
  template <unsigned N = 64> Args<N> args() const { return Args<N>(&runtime); }
#endif

  template <unsigned N> void exec(OclContext &ctx, Args<N> &args) const {
#ifndef NDEBUG
    auto rt = OclRuntime::get(ctx.queue);
    assert(rt);
    assert(*rt == this->runtime);
#endif
    auto size = args.args.size();
    auto ctxPtr = &ctx;
    ctx.runtime = &runtime;
    ctx.clPtrs = &args.clPtrs;
    args.args.emplace_back(&ctxPtr);
    args.args.emplace_back(&ctxPtr);
    args.args.emplace_back(ZERO_PTR);
    main(args.args.data());
    args.args.truncate(size);
  }

  ~OclModule();
  OclModule(const OclModule &) = delete;
  OclModule &operator=(const OclModule &) = delete;
  OclModule(const OclModule &&) = delete;
  OclModule &operator=(const OclModule &&) = delete;

private:
  OclRuntime runtime;
  std::unique_ptr<ExecutionEngine> engine;
  MainFunc main;
};

struct OclModuleBuilder {
  friend OclRuntime;
  explicit OclModuleBuilder(ModuleOp module);
  explicit OclModuleBuilder(OwningOpRef<ModuleOp> &module)
      : OclModuleBuilder(module.release()) {}

  llvm::Expected<std::shared_ptr<const OclModule>>
  build(const OclRuntime &runtime);

  llvm::Expected<std::shared_ptr<const OclModule>>
  build(cl_command_queue queue);

  llvm::Expected<std::shared_ptr<const OclModule>> build(cl_context context,
                                                         cl_device_id device);

private:
  std::shared_mutex mux;
  ModuleOp mlirModule;
  SmallString<32> funcName;
  std::unordered_map<const OclRuntime, std::shared_ptr<const OclModule>> cache;
};
}; // namespace mlir::gc::gpu
#else
#undef GC_GPU_OCL_CONST_ONLY
#endif
#endif
