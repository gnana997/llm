llama_add_compile_flags()

#
# libraries
#

# llama

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache-unified.cpp
            llama-kv-cache-unified-iswa.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampling.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            orchestration/gpu_profiler.cpp
            orchestration/gpu_profiler.h
            orchestration/gpu_profiler_interface.h
            orchestration/gpu_profiler_factory.cpp
            orchestration/gpu_profiler_factory.h
            orchestration/ggml_gpu_profiler.cpp
            orchestration/ggml_gpu_profiler.h
            orchestration/cuda_gpu_profiler.cpp
            orchestration/cuda_gpu_profiler.h
            orchestration/generic_gpu_profiler.cpp
            orchestration/generic_gpu_profiler.h
            orchestration/unified_gpu_profiler.cpp
            orchestration/unified_gpu_profiler.h
            orchestration/rocm_gpu_profiler.cpp
            orchestration/rocm_gpu_profiler.h
            orchestration/layer_distributor.cpp
            orchestration/layer_distributor.h
            orchestration/metrics.cpp
            orchestration/metrics.h
            orchestration/gpu_architecture_analyzer.cpp
            orchestration/gpu_architecture_analyzer.h
            orchestration/orchestration_logger.cpp
            orchestration/orchestration_logger.h
            orchestration/advanced_metrics.cpp
            orchestration/advanced_metrics.h
            orchestration/runtime_performance_monitor.cpp
            orchestration/runtime_performance_monitor.h
            orchestration/adaptive_layer_redistributor.cpp
            orchestration/adaptive_layer_redistributor.h
            )

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml common)

# Add CUDA include directories if CUDA is enabled
if(GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_include_directories(llama PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(llama PRIVATE GGML_USE_CUDA)
    # Link CUDA libraries
    target_link_libraries(llama PUBLIC CUDA::cudart CUDA::cublas CUDA::curand)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
