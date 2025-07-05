#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum number of GPUs supported
#define MAX_CUDA_DEVICES 16

// GPU device information structure
typedef struct {
    int device_id;
    char name[256];
    
    // Architecture info
    int compute_capability;
    int cc_major;
    int cc_minor;
    int warp_size;
    bool integrated;
    
    // Memory info
    size_t total_memory;
    size_t free_memory;
    size_t used_memory;
    int memory_clock_mhz;
    float memory_bandwidth_gbps;
    
    // Compute info
    int sm_count;
    int max_clock_mhz;
    int max_threads_per_block;
    int max_blocks_per_sm;
    int max_registers_per_block;
    
    // Cache info
    int l1_cache_size_kb;
    int l2_cache_size_kb;
    int constant_memory_kb;
    int texture_memory_kb;
    
    // PCIe info
    int pcie_generation;
    int pcie_link_width;
    
    // Features
    bool supports_fp16;
    bool supports_int8_dp4a;
    bool supports_tensor_cores;
    bool supports_bf16;
    bool has_nvlink;
    bool vmm;
    size_t vmm_granularity;
    
    // Multi-GPU topology
    int nvlink_peers[MAX_CUDA_DEVICES];
} cuda_device_info_t;

// Main GPU information structure
typedef struct {
    int device_count;
    cuda_device_info_t devices[MAX_CUDA_DEVICES];
    
    // System-wide info
    bool has_cuda;
    int cuda_version;
    int driver_version;
} cuda_system_info_t;

// Initialize CUDA and get system information
bool cuda_get_system_info(cuda_system_info_t* info);

// Get detailed info for a specific device
bool cuda_get_device_info(int device_id, cuda_device_info_t* info);

// Enable profiling/benchmarking features
void cuda_enable_profiling(bool enable_bandwidth, bool enable_topology);

// Get GGML CUDA internal info if available
bool cuda_get_ggml_info(cuda_system_info_t* info);

#ifdef __cplusplus
}
#endif