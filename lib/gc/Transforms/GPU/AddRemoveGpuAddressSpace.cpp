//===- AddRemoveGpuAddressSpace.cpp -----------------------------*- C++ -*-===//
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
#define GEN_PASS_DECL_ADDREMOVEGPUADDRESSSPACE
#define GEN_PASS_DEF_ADDREMOVEGPUADDRESSSPACE
#include "gc/Transforms/Passes.h.inc"
}
}

namespace {
struct AddRemoveGpuAddressSpace
    : gc::impl::AddRemoveGpuAddressSpaceBase<AddRemoveGpuAddressSpace> {

  AddRemoveGpuAddressSpace() = default;

  AddRemoveGpuAddressSpace(const gc::AddRemoveGpuAddressSpaceOptions &options) :
    AddRemoveGpuAddressSpaceBase(options) {
  }

  void runOnOperation() override {
    AttrTypeReplacer replacer;
    if (remove) {
      replacer.addReplacement([](MemRefType type) {
        if (auto memSpace = type.getMemorySpace()) {
          if (dyn_cast<gpu::AddressSpaceAttr>(memSpace)) {
            return MemRefType::get(type.getShape(), type.getElementType(),
                                   type.getLayout());
          }
        }
        return type;
      });
    } else {
      auto addrSpace = gpu::AddressSpaceAttr::get(getOperation()->getContext(),
                                                  mlir::gpu::AddressSpace::Global);
      replacer.addReplacement([addrSpace](MemRefType type) {
        return MemRefType::get(type.getShape(), type.getElementType(),
                               type.getLayout(), addrSpace);
      });
    }

    replacer.recursivelyReplaceElementsIn(getOperation(), true, false, true);
  }
};
}