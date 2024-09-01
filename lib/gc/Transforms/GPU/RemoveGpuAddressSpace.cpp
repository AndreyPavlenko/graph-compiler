//===- RemoveGpuAddressSpace.cpp - Remove #gpu.address_space ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"

using namespace mlir;

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_REMOVEGPUADDRESSSPACE
#include "gc/Transforms/Passes.h.inc"
} // namespace gc
} // namespace mlir

namespace {
struct RemoveGpuAddressSpace
    : gc::impl::RemoveGpuAddressSpaceBase<RemoveGpuAddressSpace> {
  void runOnOperation() override {
    AttrTypeReplacer replacer;
    replacer.addReplacement([](MemRefType type) {
      if (auto memSpace = type.getMemorySpace()) {
        if (dyn_cast<gpu::AddressSpaceAttr>(memSpace)) {
          return MemRefType::get(type.getShape(), type.getElementType(),
                                 type.getLayout());
        }
      }
      return type;
    });
    replacer.recursivelyReplaceElementsIn(getOperation(), true, false, true);
  }
};
} // namespace