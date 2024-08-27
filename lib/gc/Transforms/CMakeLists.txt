add_subdirectory(Utils)

gc_set_mlir_link_components(MLIR_LINK_COMPONENTS
  MLIRIR
  MLIRSupport
  MLIRBufferizationToMemRef
  MLIRBufferizationPipelines)

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

gc_add_mlir_library(GcPasses
  MemRefToCPURuntime.cpp
  OneDNNGraphToLinalg.cpp
  Pipeline.cpp
  IterativeTilingAndFusion.cpp
  TilingUsingInterfaceX.cpp
  VerifyTargetDescription.cpp
  DecomposeAggregatedOps.cpp
  DeepTileContractionOp.cpp
  TilingUtil.cpp
  SinkOpIntoInnerLoop.cpp
  MergeNestedForall.cpp
  StaticMemoryPlanning.cpp
  MergeAlloc.cpp
  MergeAllocTickBased.cpp
  FoldTensorOperation.cpp
  LowerToTileVector.cpp

  DEPENDS
    GraphCompilerPassIncGen

  LINK_LIBS PUBLIC
    ${mlir_dialect_libs}
    ${mlir_conversion_libs}
    ${MLIR_LINK_COMPONENTS}
    ${GC_ONEDNN_DIALECT_LIB_NAME}
    GcInterface
    MLIRMicrokernelTransforms
  )

if(GC_ENABLE_IMEX)
  add_subdirectory(GPU)
  include(imex)
  get_property(IMEX_INCLUDES GLOBAL PROPERTY IMEX_INCLUDES)
  get_property(IMEX_DIALECT_LIBS GLOBAL PROPERTY IMEX_DIALECT_LIBS)
  get_property(IMEX_CONVERSION_LIBS GLOBAL PROPERTY IMEX_CONVERSION_LIBS)
  set(IMEX_LIBS
    ${IMEX_DIALECT_LIBS}
    ${IMEX_CONVERSION_LIBS}
    IMEXTransforms
    IMEXUtil
  )
  foreach(include_dir IN LISTS IMEX_INCLUDES)
    list(APPEND IMEX_BUILD_INCLUDES $<BUILD_INTERFACE:${include_dir}>)
  endforeach()
  target_include_directories(GcGpuPasses PUBLIC ${IMEX_BUILD_INCLUDES})
  target_link_libraries(GcGpuPasses PUBLIC ${IMEX_LIBS})
  set_property(GLOBAL APPEND PROPERTY IMEX_LIBS ${IMEX_LIBS})
endif()

target_link_libraries(GcPasses PUBLIC GcGpuPasses)
add_subdirectory(Microkernel)
