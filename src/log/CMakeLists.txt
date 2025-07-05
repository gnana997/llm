# llama-log library

add_library(llama-log STATIC
    log-ex.cpp
    log-ex.h
)

target_include_directories(llama-log PUBLIC . ${CMAKE_SOURCE_DIR}/include)
target_compile_features(llama-log PUBLIC cxx_std_17)

# Link with ggml to get its headers
target_link_libraries(llama-log PUBLIC ggml)

# Add CUDA support if needed
if(GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_include_directories(llama-log PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(llama-log PRIVATE GGML_USE_CUDA)
endif()

# Ensure position independent code for shared library builds
if (BUILD_SHARED_LIBS)
    set_target_properties(llama-log PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()