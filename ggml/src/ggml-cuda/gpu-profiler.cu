#include "gpu-profiler.cuh"
#include "ggml.h"
#include <cstdio>
#include <algorithm>

// Include metrics if available in the build
#if __has_include("../../common/metrics.h")
#include "../../common/metrics.h"
#include "../../common/metrics-gpu.h"
#define METRICS_AVAILABLE
#endif

// Include logging if available
#if __has_include("../../common/log-ex.h")
#include "../../common/log-ex.h"
#define LOGGING_AVAILABLE
#endif

namespace ggml_cuda_profiler {

// Simple memory copy kernel for bandwidth testing
__global__ void bandwidth_test_kernel(float* dst, const float* src, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

gpu_profile_result profile_memory_bandwidth(int device_id) {
    gpu_profile_result result = {};
    
    // Set device
    CUDA_CHECK(cudaSetDevice(device_id));
    
    // Allocate test buffers
    float *d_src = nullptr, *d_dst = nullptr;
    size_t test_size = BANDWIDTH_TEST_SIZE;
    
    cudaError_t err = cudaMalloc(&d_src, test_size);
    if (err != cudaSuccess) {
        // Try smaller size if allocation fails
        test_size = 64 * 1024 * 1024; // 64 MB
        err = cudaMalloc(&d_src, test_size);
        if (err != cudaSuccess) {
            return result; // Failed to allocate
        }
    }
    
    err = cudaMalloc(&d_dst, test_size);
    if (err != cudaSuccess) {
        cudaFree(d_src);
        return result;
    }
    
    // Initialize source buffer
    cudaMemset(d_src, 1, test_size);
    
    // Warm up
    int block_size = 256;
    int grid_size = (test_size / sizeof(float) + block_size - 1) / block_size;
    bandwidth_test_kernel<<<grid_size, block_size>>>(d_dst, d_src, test_size / sizeof(float));
    cudaDeviceSynchronize();
    
    // Measure bandwidth
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < BANDWIDTH_TEST_ITERATIONS; i++) {
        bandwidth_test_kernel<<<grid_size, block_size>>>(d_dst, d_src, test_size / sizeof(float));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Calculate bandwidth in GB/s
    double total_bytes = (double)test_size * BANDWIDTH_TEST_ITERATIONS * 2; // Read + Write
    double seconds = milliseconds / 1000.0;
    result.measured_bandwidth_gbps = (float)((total_bytes / (1024.0 * 1024.0 * 1024.0)) / seconds);
    result.profiling_duration_ms = milliseconds;
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_src);
    cudaFree(d_dst);
    
    return result;
}

void detect_gpu_topology(ggml_cuda_device_info& info) {
    // Initialize all peer connections to -1 (no connection)
    for (int i = 0; i < info.device_count; i++) {
        for (int j = 0; j < GGML_CUDA_MAX_DEVICES; j++) {
            info.devices[i].nvlink_peers[j] = -1;
        }
    }
    
    // Check peer-to-peer access between all GPU pairs
    for (int i = 0; i < info.device_count; i++) {
        for (int j = 0; j < info.device_count; j++) {
            if (i == j) continue;
            
            int can_access = 0;
            cudaDeviceCanAccessPeer(&can_access, i, j);
            
            if (can_access) {
                // Check if it's NVLink or PCIe
                // This is a heuristic - NVLink typically provides much higher bandwidth
                cudaSetDevice(i);
                cudaDeviceEnablePeerAccess(j, 0);
                
                // For now, mark as generic P2P connection (0)
                // More sophisticated detection would measure P2P bandwidth
                info.devices[i].nvlink_peers[j] = 0;
                
                // Check for NVLink by looking at bandwidth characteristics
                // (Would need actual P2P bandwidth test for accurate detection)
                cudaDeviceProp prop_i, prop_j;
                cudaGetDeviceProperties(&prop_i, i);
                cudaGetDeviceProperties(&prop_j, j);
                
                // Heuristic: Same GPU model with P2P likely means NVLink
                if (prop_i.major == prop_j.major && prop_i.minor == prop_j.minor) {
                    info.devices[i].has_nvlink = true;
                    info.devices[j].has_nvlink = true;
                    info.devices[i].nvlink_peers[j] = 1; // Mark as NVLink
                }
            }
        }
    }
}

void query_device_capabilities(int device_id, ggml_cuda_device_info::cuda_device_info& dev_info) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    
    // Basic properties (some already set in ggml_cuda_init)
    dev_info.max_threads_per_block = prop.maxThreadsPerBlock;
    dev_info.max_registers_per_block = prop.regsPerBlock;
    
    // Compute capability-based features
    int cc_major = prop.major;
    int cc_minor = prop.minor;
    
    // FP16 support (Pascal and later, or AMD)
    dev_info.supports_fp16 = cc_major >= 6 || GGML_CUDA_CC_IS_AMD(dev_info.cc);
    
    // INT8 DP4A support
    dev_info.supports_int8_dp4a = (cc_major > 6) || (cc_major == 6 && cc_minor >= 1);
    
    // Tensor cores (Volta and later)
    dev_info.supports_tensor_cores = cc_major >= 7;
    
    // BF16 support (Ampere and later)
    dev_info.supports_bf16 = cc_major >= 8;
    
    // Clock speeds
    dev_info.max_clock_mhz = prop.clockRate / 1000.0f;
    dev_info.memory_clock_mhz = prop.memoryClockRate / 1000.0f;
    
    // Cache sizes
    dev_info.l2_cache_size_kb = prop.l2CacheSize / 1024;
    dev_info.l1_cache_size_kb = prop.sharedMemPerMultiprocessor / 1024;
    dev_info.constant_memory_kb = prop.totalConstMem / 1024;
    
    // Texture memory (approximate - actual limit is complex)
    dev_info.texture_memory_kb = 65536; // 64MB typical limit
    
    // PCIe information (requires additional queries on some systems)
    dev_info.pcie_generation = 4; // Default assumption
    dev_info.pcie_link_width = 16; // Default assumption
    
    // Try to get actual PCIe info
    int pcie_link_gen = 0;
    int pcie_link_width = 0;
    cudaDeviceGetAttribute(&pcie_link_gen, cudaDevAttrPciDeviceId, device_id);
    cudaDeviceGetAttribute(&pcie_link_width, cudaDevAttrPciBusId, device_id);
    
    // These attributes might not directly give generation/width
    // More sophisticated detection would query sysfs on Linux
    
    // Calculate max blocks per SM based on compute capability
    if (cc_major >= 8) {
        dev_info.max_blocks_per_sm = 32;
    } else if (cc_major >= 7) {
        dev_info.max_blocks_per_sm = 32;
    } else if (cc_major >= 6) {
        dev_info.max_blocks_per_sm = 32;
    } else {
        dev_info.max_blocks_per_sm = 16;
    }
}

void log_gpu_topology(const ggml_cuda_device_info& info) {
#ifdef LOGGING_AVAILABLE
    LOG_INF("=== GPU Topology ===\n");
    
    for (int i = 0; i < info.device_count; i++) {
        const auto& dev = info.devices[i];
        
        // Get device name
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        LOG_INF("GPU %d: %s\n", i, prop.name);
        LOG_INF("  Compute Capability: %d.%d", dev.cc / 10, dev.cc % 10);
        
        // Add architecture name
        const char* arch_name = "Unknown";
        if (GGML_CUDA_CC_IS_NVIDIA(dev.cc)) {
            if (dev.cc >= 890) arch_name = " (Ada Lovelace)";
            else if (dev.cc >= 860) arch_name = " (Hopper)";
            else if (dev.cc >= 800) arch_name = " (Ampere)";
            else if (dev.cc >= 750) arch_name = " (Turing)";
            else if (dev.cc >= 700) arch_name = " (Volta)";
            else if (dev.cc >= 600) arch_name = " (Pascal)";
        } else if (GGML_CUDA_CC_IS_AMD(dev.cc)) {
            if (GGML_CUDA_CC_IS_RDNA3(dev.cc)) arch_name = " (RDNA3)";
            else if (GGML_CUDA_CC_IS_RDNA2(dev.cc)) arch_name = " (RDNA2)";
            else if (GGML_CUDA_CC_IS_RDNA1(dev.cc)) arch_name = " (RDNA1)";
            else if (GGML_CUDA_CC_IS_CDNA(dev.cc)) arch_name = " (CDNA)";
        }
        LOG_INF("%s\n", arch_name);
        
        LOG_INF("  Memory: %.2f GB (%.2f GB available)\n", 
                dev.total_vram / (1024.0 * 1024.0 * 1024.0),
                dev.total_vram / (1024.0 * 1024.0 * 1024.0));
        
        if (dev.memory_bandwidth_gbps > 0) {
            LOG_INF("  Memory Bandwidth: %.1f GB/s (theoretical), %.1f GB/s (measured)\n",
                    (dev.memory_clock_mhz * 2 * prop.memoryBusWidth / 8) / 1000.0,
                    dev.memory_bandwidth_gbps);
        }
        
        LOG_INF("  SM Count: %d, Max Threads/Block: %d\n", 
                dev.nsm, dev.max_threads_per_block);
        LOG_INF("  L2 Cache: %d MB\n", dev.l2_cache_size_kb / 1024);
        LOG_INF("  PCIe: Gen%d x%d\n", dev.pcie_generation, dev.pcie_link_width);
        
        // Log features
        LOG_INF("  Features:");
        if (dev.supports_fp16) LOG_INF(" FP16");
        if (dev.supports_int8_dp4a) LOG_INF(" INT8-DP4A");
        if (dev.supports_tensor_cores) LOG_INF(" TensorCores");
        if (dev.supports_bf16) LOG_INF(" BF16");
        LOG_INF("\n");
        
        // Log peer connections
        bool has_peers = false;
        for (int j = 0; j < info.device_count; j++) {
            if (i != j && dev.nvlink_peers[j] >= 0) {
                if (!has_peers) {
                    LOG_INF("  Peer Connections:\n");
                    has_peers = true;
                }
                const char* link_type = dev.nvlink_peers[j] > 0 ? "NVLink" : "PCIe P2P";
                LOG_INF("    GPU %d <-> GPU %d: %s\n", i, j, link_type);
            }
        }
        LOG_INF("\n");
    }
    
    // Log topology summary
    if (info.device_count > 1) {
        LOG_INF("GPU Topology Summary:\n");
        for (int i = 0; i < info.device_count; i++) {
            for (int j = i + 1; j < info.device_count; j++) {
                const auto& dev_i = info.devices[i];
                const auto& dev_j = info.devices[j];
                
                if (dev_i.nvlink_peers[j] >= 0 || dev_j.nvlink_peers[i] >= 0) {
                    bool is_nvlink = dev_i.nvlink_peers[j] > 0 || dev_j.nvlink_peers[i] > 0;
                    const char* link_type = is_nvlink ? "NVLink" : "PCIe P2P";
                    float bandwidth = is_nvlink ? 100.0f : 32.0f; // Approximate bandwidths
                    
                    LOG_INF("  GPU %d <-> GPU %d: %s (~%.0f GB/s bidirectional)\n", 
                            i, j, link_type, bandwidth);
                }
            }
        }
    }
#else
    GGML_UNUSED(info);
    fprintf(stderr, "GPU topology logging not available (logging not included in build)\n");
#endif
}

void init_profiling_metrics() {
#ifdef METRICS_AVAILABLE
    using namespace llama::metrics;
    
    auto& registry = metric_registry::instance();
    
    // Register GPU profiling metrics
    registry.register_metric<histogram>("gpu_initialization_duration_us",
        "GPU initialization duration in microseconds",
        std::vector<double>{1000, 10000, 100000, 1000000, 10000000}); // 1ms to 10s
    
    registry.register_metric<gauge>("gpu_count_total",
        "Total number of CUDA devices detected");
    
    // Per-device metrics will be registered dynamically
#endif
}

void track_profiling_metrics(int device_id, float duration_ms, const gpu_profile_result& result) {
#ifdef METRICS_AVAILABLE
    using namespace llama::metrics;
    
    // Track initialization duration
    METRIC_OBSERVE("gpu_initialization_duration_us", duration_ms * 1000);
    
    // Create device-specific metric names
    std::string suffix = "_gpu" + std::to_string(device_id);
    
    auto& registry = metric_registry::instance();
    
    // Register and update device-specific metrics
    auto bandwidth_metric = registry.register_metric<gauge>(
        "gpu_memory_bandwidth_gbps" + suffix,
        "Measured GPU memory bandwidth in GB/s for device " + std::to_string(device_id));
    if (bandwidth_metric) {
        bandwidth_metric->set(result.measured_bandwidth_gbps);
    }
    
    auto pcie_gen_metric = registry.register_metric<gauge>(
        "gpu_pcie_generation" + suffix,
        "PCIe generation for device " + std::to_string(device_id));
    if (pcie_gen_metric) {
        pcie_gen_metric->set(result.pcie_generation);
    }
    
    auto nvlink_metric = registry.register_metric<gauge>(
        "gpu_has_nvlink" + suffix,
        "NVLink availability (0/1) for device " + std::to_string(device_id));
    if (nvlink_metric) {
        nvlink_metric->set(result.has_nvlink ? 1.0 : 0.0);
    }
#else
    GGML_UNUSED(device_id);
    GGML_UNUSED(duration_ms);
    GGML_UNUSED(result);
#endif
}

} // namespace ggml_cuda_profiler