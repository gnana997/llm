#pragma once

#include "common.cuh"
#include <chrono>

namespace ggml_cuda_profiler {

// Memory bandwidth test parameters
constexpr size_t BANDWIDTH_TEST_SIZE = 256 * 1024 * 1024; // 256 MB
constexpr int BANDWIDTH_TEST_ITERATIONS = 10;

// Simple kernel for memory bandwidth testing
__global__ void bandwidth_test_kernel(float* dst, const float* src, size_t n);

// GPU profiling functions
struct gpu_profile_result {
    float measured_bandwidth_gbps;
    int pcie_generation;
    int pcie_link_width;
    bool has_nvlink;
    float profiling_duration_ms;
};

// Profile memory bandwidth of a specific device
gpu_profile_result profile_memory_bandwidth(int device_id);

// Detect peer-to-peer connectivity between GPUs
void detect_gpu_topology(ggml_cuda_device_info& info);

// Query extended device attributes
void query_device_capabilities(int device_id, ggml_cuda_device_info::cuda_device_info& dev_info);

// Log GPU topology in human-readable format
void log_gpu_topology(const ggml_cuda_device_info& info);

// Initialize profiling metrics
void init_profiling_metrics();

// Track profiling duration
void track_profiling_metrics(int device_id, float duration_ms, const gpu_profile_result& result);

} // namespace ggml_cuda_profiler