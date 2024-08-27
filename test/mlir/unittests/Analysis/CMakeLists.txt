add_mlir_unittest(GCAnalysisTests
  TargetDescriptionAnalysisTest.cpp
)
target_include_directories(GCAnalysisTests PUBLIC "/home/jovyan/graph-compiler/build/_deps/gtest-src/googletest/include")
target_link_libraries(GCAnalysisTests
  PRIVATE
  GcPasses
  GcAnalysis
  GcJitWrapper)
