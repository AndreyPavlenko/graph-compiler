add_mlir_unittest(GCExampleTests
  Example.cpp
)
target_include_directories(GCExampleTests PUBLIC "/home/jovyan/graph-compiler/build/_deps/gtest-src/googletest/include")
target_link_libraries(GCExampleTests
  PRIVATE
  MLIRLinalgx)
