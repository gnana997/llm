#include "gpu_info_cuda.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include "ggml-cuda/common.cuh"

// Check if ggml_cuda_info exists and is accessible
extern "C" {
    // Forward declaration - this should be available from ggml-cuda
    extern const ggml_cuda_device_info& ggml_cuda_info();
}
#endif

extern "C" {

bool cuda_get_system_info(cuda_system_info_t* info) {
    if (!info) return false;
    
    memset(info, 0, sizeof(cuda_system_info_t));
    
    // Check if CUDA is available
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        info->has_cuda = false;
        info->device_count = 0;
        return false;
    }
    
    info->has_cuda = true;
    info->device_count = device_count;
    
    // Get CUDA version
    cudaRuntimeGetVersion(&info->cuda_version);
    cudaDriverGetVersion(&info->driver_version);
    
    // Get info for each device
    for (int i = 0; i < device_count && i < MAX_CUDA_DEVICES; i++) {
        cuda_get_device_info(i, &info->devices[i]);
    }
    
    return true;
}

bool cuda_get_device_info(int device_id, cuda_device_info_t* info) {
    if (!info) return false;
    
    memset(info, 0, sizeof(cuda_device_info_t));
    info->device_id = device_id;
    
    // Set current device
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        return false;
    }
    
    // Get device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        return false;
    }
    
    // Basic info
    strncpy(info->name, prop.name, sizeof(info->name) - 1);
    info->compute_capability = prop.major * 10 + prop.minor;
    info->cc_major = prop.major;
    info->cc_minor = prop.minor;
    info->warp_size = prop.warpSize;
    info->integrated = prop.integrated;
    
    // Memory info
    info->total_memory = prop.totalGlobalMem;
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    info->free_memory = free_mem;
    info->used_memory = total_mem - free_mem;
    info->memory_clock_mhz = prop.memoryClockRate / 1000; // Convert KHz to MHz
    
    // Calculate theoretical bandwidth (GB/s)
    // bandwidth = (memory_clock_MHz * 2 * memory_bus_width_bits) / 8 / 1000
    info->memory_bandwidth_gbps = (prop.memoryClockRate / 1000.0f * 2.0f * prop.memoryBusWidth) / 8.0f / 1000.0f;
    
    // Compute info
    info->sm_count = prop.multiProcessorCount;
    info->max_clock_mhz = prop.clockRate / 1000; // Convert KHz to MHz
    info->max_threads_per_block = prop.maxThreadsPerBlock;
    info->max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;
    info->max_registers_per_block = prop.regsPerBlock;
    
    // Cache info
    info->l1_cache_size_kb = prop.sharedMemPerBlock / 1024;
    info->l2_cache_size_kb = prop.l2CacheSize / 1024;
    info->constant_memory_kb = prop.totalConstMem / 1024;
    
    // PCIe info
    info->pcie_generation = prop.pciDomainID; // This is not exactly generation, but domain ID
    info->pcie_link_width = prop.pciBusID;
    
    // Features based on compute capability
    info->supports_fp16 = (prop.major >= 5 && prop.minor >= 3) || prop.major >= 6;
    info->supports_int8_dp4a = prop.major >= 6;
    info->supports_tensor_cores = prop.major >= 7;
    info->supports_bf16 = prop.major >= 8;
    
    // Check for NVLink
    for (int peer = 0; peer < MAX_CUDA_DEVICES; peer++) {
        if (peer != device_id) {
            int can_access = 0;
            cudaDeviceCanAccessPeer(&can_access, device_id, peer);
            info->nvlink_peers[peer] = can_access;
        } else {
            info->nvlink_peers[peer] = -1; // Self
        }
    }
    
    // Virtual memory management
#if CUDART_VERSION >= 10000
    info->vmm = prop.managedMemory;
#else
    info->vmm = false;
#endif
    
    return true;
}

void cuda_enable_profiling(bool enable_bandwidth, bool enable_topology) {
    if (enable_bandwidth) {
        setenv("GGML_CUDA_PROFILE_BANDWIDTH", "1", 1);
    }
    if (enable_topology) {
        setenv("GGML_CUDA_LOG_TOPOLOGY", "1", 1);
    }
}

bool cuda_get_ggml_info(cuda_system_info_t* info) {
#ifdef GGML_USE_CUDA
    // Try to get GGML CUDA info if available
    try {
        const auto& ggml_info = ggml_cuda_info();
        
        // Update device info with GGML-specific data
        for (int i = 0; i < info->device_count && i < ggml_info.device_count; i++) {
            cuda_device_info_t* dev = &info->devices[i];
            const auto& ggml_dev = ggml_info.devices[i];
            
            // Update with more accurate GGML info if available
            dev->compute_capability = ggml_dev.cc;
            dev->sm_count = ggml_dev.nsm;
            dev->total_memory = ggml_dev.total_vram;
            
            // Update features
            dev->supports_fp16 = ggml_dev.supports_fp16;
            dev->supports_int8_dp4a = ggml_dev.supports_int8_dp4a;
            dev->supports_tensor_cores = ggml_dev.supports_tensor_cores;
            dev->supports_bf16 = ggml_dev.supports_bf16;
            dev->has_nvlink = ggml_dev.has_nvlink;
            dev->vmm = ggml_dev.vmm;
            dev->vmm_granularity = ggml_dev.vmm_granularity;
            
            // Cache sizes
            dev->l1_cache_size_kb = ggml_dev.l1_cache_size_kb;
            dev->l2_cache_size_kb = ggml_dev.l2_cache_size_kb;
            
            // Memory bandwidth if measured
            if (ggml_dev.memory_bandwidth_gbps > 0) {
                dev->memory_bandwidth_gbps = ggml_dev.memory_bandwidth_gbps;
            }
            
            // Update multi-GPU topology
            for (int j = 0; j < ggml_info.device_count; j++) {
                dev->nvlink_peers[j] = ggml_dev.nvlink_peers[j];
            }
        }
        
        return true;
    } catch (...) {
        // If ggml_cuda_info is not available or fails, continue with basic info
        return false;
    }
#else
    (void)info; // Unused parameter
    return false;
#endif
}

} // extern "C"