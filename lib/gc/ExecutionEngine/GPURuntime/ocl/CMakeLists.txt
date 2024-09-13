find_package(OpenCL REQUIRED)

get_property(GC_JIT_WRAPPER_LIBS GLOBAL PROPERTY GC_JIT_WRAPPER_LIBS)
gc_add_mlir_library(GcGpuOclRuntime
        GpuOclRuntime.cpp
        LINK_LIBS PUBLIC
        ${GC_JIT_WRAPPER_LIBS}
)
target_include_directories(GcGpuOclRuntime PUBLIC ${OpenCL_INCLUDE_DIRS})
target_link_libraries(GcGpuOclRuntime PUBLIC GcInterface ${OpenCL_LIBRARIES})
