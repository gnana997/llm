// Simple test program to verify GPU profiling functionality
#include <iostream>
#include <cstdlib>
#include "ggml/src/ggml-cuda/ggml-cuda.cu"

int main(int argc, char** argv) {
    std::cout << "Testing GPU Profiling..." << std::endl;
    
    // Enable profiling via environment variables
    setenv("GGML_CUDA_PROFILE_BANDWIDTH", "1", 1);
    setenv("GGML_CUDA_LOG_TOPOLOGY", "1", 1);
    
    // Initialize CUDA and run profiling
    const auto& gpu_info = ggml_cuda_info();
    
    std::cout << "\nFound " << gpu_info.device_count << " GPU device(s)" << std::endl;
    
    for (int i = 0; i < gpu_info.device_count; i++) {
        const auto& dev = gpu_info.devices[i];
        std::cout << "\nDevice " << i << ":" << std::endl;
        std::cout << "  Compute Capability: " << dev.cc << std::endl;
        std::cout << "  Total Memory: " << (dev.total_vram / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
        std::cout << "  Memory Bandwidth: " << dev.memory_bandwidth_gbps << " GB/s" << std::endl;
        std::cout << "  L2 Cache: " << (dev.l2_cache_size_kb / 1024) << " MB" << std::endl;
        std::cout << "  Features: ";
        if (dev.supports_fp16) std::cout << "FP16 ";
        if (dev.supports_int8_dp4a) std::cout << "INT8-DP4A ";
        if (dev.supports_tensor_cores) std::cout << "TensorCores ";
        if (dev.supports_bf16) std::cout << "BF16 ";
        std::cout << std::endl;
    }
    
    return 0;
}