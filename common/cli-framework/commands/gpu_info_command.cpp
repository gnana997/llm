#include "gpu_info_command.h"
#include "log.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdlib>

#ifdef GGML_USE_CUDA
#include "gpu_info_cuda.h"
#endif

namespace llama {
namespace cli {

int GpuInfoCommand::execute([[maybe_unused]] CommandContext& ctx) {
#ifndef GGML_USE_CUDA
    std::cerr << "Error: CUDA support is not enabled in this build.\n";
    std::cerr << "Please rebuild with GGML_CUDA=ON to use GPU features.\n";
    return 1;
#else
    // Parse options
    verbose_ = ctx.hasOption("verbose") || ctx.hasOption("v");
    json_ = ctx.hasOption("json");
    benchmark_ = ctx.hasOption("benchmark") || ctx.hasOption("bench");
    
    // Enable profiling if requested
    cuda_enable_profiling(benchmark_, verbose_);
    
    // Get GPU information
    cuda_system_info_t gpu_info;
    if (!cuda_get_system_info(&gpu_info)) {
        if (!json_) {
            std::cout << "Failed to get CUDA system information.\n";
        } else {
            std::cout << "{\"error\": \"Failed to get CUDA system information\"}\n";
        }
        return 1;
    }
    
    // Try to get GGML-specific info if available
    cuda_get_ggml_info(&gpu_info);
    
    if (gpu_info.device_count == 0) {
        if (!json_) {
            std::cout << "No CUDA devices found.\n";
        } else {
            std::cout << "{\"device_count\": 0, \"devices\": []}\n";
        }
        return 0;
    }
    
    // Display information based on format
    if (json_) {
        displayJsonInfo();
    } else if (verbose_) {
        displayVerboseInfo();
    } else {
        displayBasicInfo();
    }
    
    return 0;
#endif
}

#ifdef GGML_USE_CUDA
void GpuInfoCommand::displayBasicInfo() {
    cuda_system_info_t gpu_info;
    cuda_get_system_info(&gpu_info);
    cuda_get_ggml_info(&gpu_info);
    
    std::cout << "GPU Information:\n";
    std::cout << "================\n\n";
    
    double total_memory_gb = 0;
    
    for (int i = 0; i < gpu_info.device_count; i++) {
        const auto& dev = gpu_info.devices[i];
        
        std::cout << "Device " << i << ": " << dev.name << "\n";
        
        // Compute capability and architecture
        std::cout << "  Compute Capability: " << dev.cc_major << "." << dev.cc_minor;
        
        // Architecture name
        const char* arch_name = "";
        int cc = dev.compute_capability;
        if (cc >= 890) arch_name = " (Ada Lovelace)";
        else if (cc >= 860) arch_name = " (Hopper)";
        else if (cc >= 800) arch_name = " (Ampere)";
        else if (cc >= 750) arch_name = " (Turing)";
        else if (cc >= 700) arch_name = " (Volta)";
        else if (cc >= 600) arch_name = " (Pascal)";
        else if (cc >= 500) arch_name = " (Maxwell)";
        std::cout << arch_name << "\n";
        
        // Memory
        double memory_gb = dev.total_memory / (1024.0 * 1024.0 * 1024.0);
        total_memory_gb += memory_gb;
        std::cout << "  Memory: " << std::fixed << std::setprecision(2) << memory_gb << " GB\n";
        
        // SMs and cores
        int cores_per_sm = 0;
        if (cc >= 800) cores_per_sm = 128;  // Ampere and later
        else if (cc >= 700) cores_per_sm = 64;   // Volta, Turing
        else if (cc >= 600) cores_per_sm = 128;  // Pascal
        else if (cc >= 500) cores_per_sm = 128;  // Maxwell
        else cores_per_sm = 192;  // Kepler
        
        int total_cores = dev.sm_count * cores_per_sm;
        std::cout << "  SMs: " << dev.sm_count << ", Cores: " << total_cores << "\n";
        
        // Clock speed
        std::cout << "  Clock: " << dev.max_clock_mhz << " MHz\n";
        
        // L2 Cache
        std::cout << "  L2 Cache: " << (dev.l2_cache_size_kb / 1024) << " MB\n";
        
        // Features
        std::cout << "  Features:";
        if (dev.supports_fp16) std::cout << " FP16";
        if (dev.supports_int8_dp4a) std::cout << " INT8";
        if (dev.supports_tensor_cores) std::cout << " TensorCores";
        if (dev.supports_bf16) std::cout << " BF16";
        std::cout << "\n\n";
    }
    
    // Summary
    std::cout << "Total GPUs: " << gpu_info.device_count << "\n";
    std::cout << "Total Memory: " << std::fixed << std::setprecision(2) << total_memory_gb << " GB\n";
    
    if (gpu_info.device_count > 1) {
        std::cout << "Multi-GPU: Yes";
        
        // Check for NVLink
        bool has_nvlink = false;
        for (int i = 0; i < gpu_info.device_count && !has_nvlink; i++) {
            has_nvlink = gpu_info.devices[i].has_nvlink;
        }
        
        if (has_nvlink) {
            std::cout << " (NVLink)";
        } else {
            std::cout << " (PCIe P2P)";
        }
        std::cout << "\n";
    }
}

void GpuInfoCommand::displayVerboseInfo() {
    cuda_system_info_t gpu_info;
    cuda_get_system_info(&gpu_info);
    cuda_get_ggml_info(&gpu_info);
    
    std::cout << "GPU Information (Verbose):\n";
    std::cout << "==========================\n\n";
    
    for (int i = 0; i < gpu_info.device_count; i++) {
        const auto& dev = gpu_info.devices[i];
        
        std::cout << "Device " << i << ": " << dev.name << "\n";
        std::cout << "  Architecture:\n";
        std::cout << "    Compute Capability: " << dev.cc_major << "." << dev.cc_minor << "\n";
        std::cout << "    Warp Size: " << dev.warp_size << "\n";
        std::cout << "    Integrated: " << (dev.integrated ? "Yes" : "No") << "\n";
        
        std::cout << "  Memory:\n";
        std::cout << "    Total: " << std::fixed << std::setprecision(2) 
                  << (dev.total_memory / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
        if (dev.memory_bandwidth_gbps > 0) {
            std::cout << "    Bandwidth: " << std::fixed << std::setprecision(1) 
                      << dev.memory_bandwidth_gbps << " GB/s (measured)\n";
        }
        std::cout << "    Clock: " << dev.memory_clock_mhz << " MHz\n";
        
        std::cout << "  Compute:\n";
        std::cout << "    SMs: " << dev.sm_count << "\n";
        std::cout << "    Max Clock: " << dev.max_clock_mhz << " MHz\n";
        std::cout << "    Max Threads/Block: " << dev.max_threads_per_block << "\n";
        std::cout << "    Max Blocks/SM: " << dev.max_blocks_per_sm << "\n";
        std::cout << "    Max Registers/Block: " << dev.max_registers_per_block << "\n";
        
        std::cout << "  Cache:\n";
        std::cout << "    L1/Shared: " << dev.l1_cache_size_kb << " KB\n";
        std::cout << "    L2: " << (dev.l2_cache_size_kb / 1024) << " MB\n";
        std::cout << "    Constant: " << dev.constant_memory_kb << " KB\n";
        std::cout << "    Texture: " << dev.texture_memory_kb << " KB\n";
        
        std::cout << "  PCIe:\n";
        std::cout << "    Generation: " << dev.pcie_generation << "\n";
        std::cout << "    Link Width: x" << dev.pcie_link_width << "\n";
        
        std::cout << "  Features:\n";
        std::cout << "    FP16: " << (dev.supports_fp16 ? "Yes" : "No") << "\n";
        std::cout << "    INT8 DP4A: " << (dev.supports_int8_dp4a ? "Yes" : "No") << "\n";
        std::cout << "    Tensor Cores: " << (dev.supports_tensor_cores ? "Yes" : "No") << "\n";
        std::cout << "    BF16: " << (dev.supports_bf16 ? "Yes" : "No") << "\n";
        std::cout << "    Virtual Memory: " << (dev.vmm ? "Yes" : "No") << "\n";
        
        if (dev.vmm && dev.vmm_granularity > 0) {
            std::cout << "    VMM Granularity: " << (dev.vmm_granularity / 1024) << " KB\n";
        }
        
        std::cout << "\n";
    }
    
    // Display topology if multi-GPU
    if (gpu_info.device_count > 1) {
        displayTopology();
    }
}

void GpuInfoCommand::displayTopology() {
    cuda_system_info_t gpu_info;
    cuda_get_system_info(&gpu_info);
    cuda_get_ggml_info(&gpu_info);
    
    std::cout << "GPU Topology:\n";
    std::cout << "=============\n";
    
    // Create connectivity matrix
    for (int i = 0; i < gpu_info.device_count; i++) {
        for (int j = 0; j < gpu_info.device_count; j++) {
            if (i == j) continue;
            
            const auto& dev = gpu_info.devices[i];
            if (dev.nvlink_peers[j] >= 0) {
                std::cout << "  GPU " << i << " <-> GPU " << j << ": ";
                if (dev.nvlink_peers[j] > 0) {
                    std::cout << "NVLink";
                } else {
                    std::cout << "PCIe P2P";
                }
                std::cout << "\n";
            }
        }
    }
    std::cout << "\n";
}

void GpuInfoCommand::displayJsonInfo() {
    cuda_system_info_t gpu_info;
    cuda_get_system_info(&gpu_info);
    cuda_get_ggml_info(&gpu_info);
    
    std::cout << "{\n";
    std::cout << "  \"device_count\": " << gpu_info.device_count << ",\n";
    std::cout << "  \"devices\": [\n";
    
    for (int i = 0; i < gpu_info.device_count; i++) {
        const auto& dev = gpu_info.devices[i];
        
        std::cout << "    {\n";
        std::cout << "      \"id\": " << i << ",\n";
        std::cout << "      \"name\": \"" << dev.name << "\",\n";
        std::cout << "      \"compute_capability\": " << dev.compute_capability << ",\n";
        std::cout << "      \"compute_capability_major\": " << dev.cc_major << ",\n";
        std::cout << "      \"compute_capability_minor\": " << dev.cc_minor << ",\n";
        std::cout << "      \"memory_gb\": " << std::fixed << std::setprecision(2) 
                  << (dev.total_memory / (1024.0 * 1024.0 * 1024.0)) << ",\n";
        std::cout << "      \"memory_bytes\": " << dev.total_memory << ",\n";
        
        if (dev.memory_bandwidth_gbps > 0) {
            std::cout << "      \"memory_bandwidth_gbps\": " << std::fixed << std::setprecision(1) 
                      << dev.memory_bandwidth_gbps << ",\n";
        }
        
        std::cout << "      \"sms\": " << dev.sm_count << ",\n";
        std::cout << "      \"max_clock_mhz\": " << dev.max_clock_mhz << ",\n";
        std::cout << "      \"l2_cache_mb\": " << (dev.l2_cache_size_kb / 1024) << ",\n";
        std::cout << "      \"features\": [";
        
        bool first = true;
        if (dev.supports_fp16) {
            std::cout << "\"fp16\"";
            first = false;
        }
        if (dev.supports_int8_dp4a) {
            if (!first) std::cout << ", ";
            std::cout << "\"int8\"";
            first = false;
        }
        if (dev.supports_tensor_cores) {
            if (!first) std::cout << ", ";
            std::cout << "\"tensor_cores\"";
            first = false;
        }
        if (dev.supports_bf16) {
            if (!first) std::cout << ", ";
            std::cout << "\"bf16\"";
        }
        
        std::cout << "],\n";
        std::cout << "      \"integrated\": " << (dev.integrated ? "true" : "false") << ",\n";
        std::cout << "      \"has_nvlink\": " << (dev.has_nvlink ? "true" : "false");
        
        if (verbose_) {
            std::cout << ",\n";
            std::cout << "      \"warp_size\": " << dev.warp_size << ",\n";
            std::cout << "      \"max_threads_per_block\": " << dev.max_threads_per_block << ",\n";
            std::cout << "      \"max_blocks_per_sm\": " << dev.max_blocks_per_sm << ",\n";
            std::cout << "      \"pcie_generation\": " << dev.pcie_generation << ",\n";
            std::cout << "      \"pcie_link_width\": " << dev.pcie_link_width << ",\n";
            std::cout << "      \"l1_cache_kb\": " << dev.l1_cache_size_kb << ",\n";
            std::cout << "      \"constant_memory_kb\": " << dev.constant_memory_kb << ",\n";
            std::cout << "      \"vmm\": " << (dev.vmm ? "true" : "false");
        }
        
        std::cout << "\n    }";
        if (i < gpu_info.device_count - 1) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    
    std::cout << "  ],\n";
    
    // Calculate total memory
    double total_memory_gb = 0;
    for (int i = 0; i < gpu_info.device_count; i++) {
        total_memory_gb += gpu_info.devices[i].total_memory / (1024.0 * 1024.0 * 1024.0);
    }
    
    std::cout << "  \"total_memory_gb\": " << std::fixed << std::setprecision(2) << total_memory_gb << "\n";
    std::cout << "}\n";
}
#endif // GGML_USE_CUDA

} // namespace cli
} // namespace llama