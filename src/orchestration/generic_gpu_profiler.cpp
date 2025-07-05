#include "generic_gpu_profiler.h"
#include "llama-impl.h"
#include <algorithm>
#include <cmath>

namespace llama {
namespace orchestration {

GenericGpuProfiler::GenericGpuProfiler() {
    LLAMA_LOG_INFO("[Generic GPU Profiler] Initialized\n");
}

bool GenericGpuProfiler::supports_feature(const std::string& feature_name) const {
    // Generic profiler has limited feature support
    if (feature_name == "basic_profiling") return true;
    if (feature_name == "memory_info") return true;
    if (feature_name == "multi_gpu") return true;
    return false;
}

std::unordered_map<std::string, std::string> GenericGpuProfiler::get_capabilities() const {
    std::unordered_map<std::string, std::string> caps;
    
    caps["backend"] = "Generic";
    caps["profiling_method"] = "GGML-based";
    caps["performance_estimation"] = "Conservative";
    caps["detailed_properties"] = "Limited";
    
    return caps;
}

bool GenericGpuProfiler::is_compatible_device(ggml_backend_dev_t device) const {
    if (!device) return false;
    
    // Get device properties
    struct ggml_backend_dev_props props;
    ggml_backend_dev_get_props(device, &props);
    
    // Generic profiler works with any GPU device
    return props.type == GGML_BACKEND_DEVICE_TYPE_GPU;
}

void GenericGpuProfiler::query_extended_properties(ggml_backend_dev_t device, GpuProfile& profile) {
    // Get backend type from device
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    if (reg) {
        const char* backend_name = ggml_backend_reg_name(reg);
        profile.backend_type = backend_name ? backend_name : "Unknown";
    } else {
        profile.backend_type = "Unknown";
    }
    
    // Estimate capabilities based on available information
    estimate_device_capabilities(profile);
    
    // Try to detect some features through GGML
    detect_backend_features(device, profile);
    
    // Conservative performance estimates
    if (profile.theoretical_tflops == 0.0f) {
        profile.theoretical_tflops = estimate_performance_from_memory(profile.total_memory_bytes);
    }
    
    LLAMA_LOG_INFO("[Generic GPU Profiler] Device %d (%s): %s\n",
                   profile.device_id, profile.backend_type.c_str(), profile.name.c_str());
    LLAMA_LOG_INFO("  Estimated performance: %.2f TFLOPS\n", profile.theoretical_tflops);
}

float GenericGpuProfiler::run_performance_benchmark(ggml_backend_dev_t device) {
    LLAMA_LOG_INFO("[Generic GPU Profiler] Running GGML-based benchmark\n");
    
    // Use the base class GGML benchmark
    float tflops = run_ggml_benchmark(device);
    
    if (tflops > 0.0f) {
        LLAMA_LOG_INFO("[Generic GPU Profiler] Benchmark result: %.2f TFLOPS\n", tflops);
        return tflops;
    }
    
    // If benchmark fails, estimate from device properties
    struct ggml_backend_dev_props props;
    ggml_backend_dev_get_props(device, &props);
    
    float estimated_tflops = estimate_performance_from_memory(props.memory_total);
    LLAMA_LOG_INFO("[Generic GPU Profiler] Estimated performance: %.2f TFLOPS\n", estimated_tflops);
    
    return estimated_tflops;
}

void GenericGpuProfiler::estimate_device_capabilities(GpuProfile& profile) {
    // Based on memory size, make educated guesses about capabilities
    float memory_gb = profile.total_memory_bytes / (1024.0f * 1024.0f * 1024.0f);
    
    // Memory bandwidth estimation
    if (profile.memory_bandwidth_gbps == 0.0f) {
        // Rough estimates based on typical GPU memory configurations
        if (memory_gb >= 80.0f) {
            // High-end datacenter GPU (A100, H100 class)
            profile.memory_bandwidth_gbps = 2000.0f;
            profile.has_tensor_cores = true;
            profile.has_fp16 = true;
            profile.has_int8 = true;
        } else if (memory_gb >= 40.0f) {
            // High-end consumer/prosumer GPU (RTX 4090, A6000 class)
            profile.memory_bandwidth_gbps = 1000.0f;
            profile.has_tensor_cores = true;
            profile.has_fp16 = true;
            profile.has_int8 = true;
        } else if (memory_gb >= 24.0f) {
            // Mid-high GPU (RTX 3090, A5000 class)
            profile.memory_bandwidth_gbps = 900.0f;
            profile.has_fp16 = true;
        } else if (memory_gb >= 16.0f) {
            // Mid-range GPU
            profile.memory_bandwidth_gbps = 600.0f;
            profile.has_fp16 = true;
        } else if (memory_gb >= 8.0f) {
            // Entry-level GPU
            profile.memory_bandwidth_gbps = 400.0f;
        } else {
            // Low-end GPU
            profile.memory_bandwidth_gbps = 200.0f;
        }
    }
    
    // SM/CU count estimation (very rough)
    if (profile.sm_count == 0) {
        // Estimate based on memory size (larger GPUs tend to have more compute units)
        if (memory_gb >= 80.0f) {
            profile.sm_count = 108;  // A100-class
        } else if (memory_gb >= 40.0f) {
            profile.sm_count = 84;   // RTX 4090-class
        } else if (memory_gb >= 24.0f) {
            profile.sm_count = 68;   // RTX 3090-class
        } else if (memory_gb >= 16.0f) {
            profile.sm_count = 48;   // RTX 3080-class
        } else if (memory_gb >= 8.0f) {
            profile.sm_count = 36;   // RTX 3070-class
        } else {
            profile.sm_count = 20;   // Entry-level
        }
    }
    
    // Clock speed estimation
    if (profile.max_clock_mhz == 0) {
        profile.max_clock_mhz = 1500;  // Conservative estimate
    }
}

float GenericGpuProfiler::estimate_performance_from_memory(size_t memory_bytes) {
    float memory_gb = memory_bytes / (1024.0f * 1024.0f * 1024.0f);
    
    // Conservative TFLOPS estimation based on typical GPU configurations
    // These are FP32 TFLOPS estimates
    if (memory_gb >= 80.0f) {
        return 60.0f;  // A100/H100 class
    } else if (memory_gb >= 40.0f) {
        return 40.0f;  // RTX 4090/A6000 class
    } else if (memory_gb >= 24.0f) {
        return 25.0f;  // RTX 3090/A5000 class
    } else if (memory_gb >= 16.0f) {
        return 20.0f;  // RTX 3080 class
    } else if (memory_gb >= 12.0f) {
        return 15.0f;  // RTX 3070 Ti class
    } else if (memory_gb >= 8.0f) {
        return 12.0f;  // RTX 3070 class
    } else if (memory_gb >= 6.0f) {
        return 8.0f;   // RTX 3060 class
    } else {
        // Very conservative for smaller GPUs
        return memory_gb * 1.5f;
    }
}

void GenericGpuProfiler::detect_backend_features(ggml_backend_dev_t device, GpuProfile& profile) {
    // Try to create a backend and test feature support
    ggml_backend_t backend = init_backend_from_device(device);
    if (!backend) {
        return;
    }
    
    // Check for FP16 support by trying to create FP16 tensors
    struct ggml_init_params params = {};
    params.mem_size   = 1024 * 1024;  // 1 MB
    params.mem_buffer = nullptr;
    params.no_alloc   = true;
    
    struct ggml_context* ctx = ggml_init(params);
    if (ctx) {
        // Try creating tensors of different types
        struct ggml_tensor* fp16_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 1024);
        if (fp16_tensor) {
            ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (buffer) {
                profile.has_fp16 = true;
                ggml_backend_buffer_free(buffer);
            }
        }
        
        // Similar checks could be added for other data types
        
        ggml_free(ctx);
    }
    
    ggml_backend_free(backend);
    
    // Backend-specific feature detection based on name
    if (profile.backend_type == "CUDA" || profile.backend_type == "HIP") {
        // These backends typically support advanced features
        profile.has_unified_memory = true;
        profile.supports_fp64 = true;
    } else if (profile.backend_type == "Metal") {
        // Metal has unified memory architecture
        profile.has_unified_memory = true;
    }
}

} // namespace orchestration
} // namespace llama