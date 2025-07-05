#include "ggml_gpu_profiler.h"
#include "llama-impl.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <chrono>

namespace llama {
namespace orchestration {

struct GgmlGpuProfiler::Impl {
    std::chrono::steady_clock::time_point last_cache_cleanup;
    static constexpr int CACHE_TTL_MINUTES = 5;
    
    Impl() : last_cache_cleanup(std::chrono::steady_clock::now()) {}
    
    void cleanup_cache_if_needed(std::unordered_map<int, GpuProfile>& cache) {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::minutes>(now - last_cache_cleanup).count() >= CACHE_TTL_MINUTES) {
            cache.clear();
            last_cache_cleanup = now;
        }
    }
};

GgmlGpuProfiler::GgmlGpuProfiler() : pImpl(std::make_unique<Impl>()) {}
GgmlGpuProfiler::~GgmlGpuProfiler() = default;

ProfilingResult GgmlGpuProfiler::profile_all_gpus() {
    ProfilingResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Load all backends if not already loaded
        ggml_backend_load_all();
        
        size_t device_count = ggml_backend_dev_count();
        LLAMA_LOG_INFO("[%s Profiler] Found %zu total devices\n", get_backend_type().c_str(), device_count);
        
        int compatible_device_index = 0;
        for (size_t i = 0; i < device_count; ++i) {
            ggml_backend_dev_t device = ggml_backend_dev_get(i);
            if (!device) continue;
            
            // Check if this device is compatible with our backend
            if (!is_compatible_device(device)) continue;
            
            try {
                GpuProfile profile = profile_gpu(compatible_device_index++);
                result.gpu_profiles.push_back(profile);
                log_gpu_profile(profile);
            } catch (const std::exception& e) {
                LLAMA_LOG_INFO("[%s Profiler] Failed to profile device %zu: %s\n", 
                              get_backend_type().c_str(), i, e.what());
            }
        }
        
        result.success = !result.gpu_profiles.empty();
        if (!result.success) {
            result.error_message = "No compatible " + get_backend_type() + " devices found";
        }
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Profiling failed: ") + e.what();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.profiling_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    log_profiling_summary(result);
    return result;
}

GpuProfile GgmlGpuProfiler::profile_gpu(int device_id) {
    GpuProfile profile;
    profile.device_id = device_id;
    profile.backend_type = get_backend_type();
    
    // Check cache first
    if (get_cached_profile(device_id, profile)) {
        return profile;
    }
    
    // Get GGML device
    ggml_backend_dev_t device = get_device_by_index(device_id);
    if (!device) {
        throw std::runtime_error("Invalid device ID: " + std::to_string(device_id));
    }
    
    // Query basic properties through GGML
    query_basic_properties(device, profile);
    
    // Query extended properties (backend-specific)
    query_extended_properties(device, profile);
    
    // Run performance benchmark
    try {
        profile.measured_performance_score = benchmark_gpu_performance(device_id);
    } catch (...) {
        // If benchmark fails, estimate from specs
        profile.measured_performance_score = profile.theoretical_tflops * 0.7f;
    }
    
    // Cache the result
    cache_profile(profile);
    
    return profile;
}

bool GgmlGpuProfiler::multi_gpu_supported() const {
    // Check if we have multiple compatible devices
    size_t compatible_count = 0;
    size_t total_devices = ggml_backend_dev_count();
    
    for (size_t i = 0; i < total_devices; ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (device && is_compatible_device(device)) {
            compatible_count++;
            if (compatible_count > 1) return true;
        }
    }
    
    return false;
}

void GgmlGpuProfiler::query_basic_properties(ggml_backend_dev_t device, GpuProfile& profile) {
    struct ggml_backend_dev_props props;
    ggml_backend_dev_get_props(device, &props);
    
    // Basic properties available from GGML
    profile.name = props.name ? props.name : "Unknown Device";
    profile.total_memory_bytes = props.memory_total;
    profile.available_memory_bytes = props.memory_free;
    
    // Set defaults for properties not available through GGML
    profile.memory_bandwidth_gbps = 0.0f;
    profile.compute_capability_major = 0;
    profile.compute_capability_minor = 0;
    profile.sm_count = 0;
    profile.max_clock_mhz = 0;
    profile.theoretical_tflops = 0.0f;
    profile.has_tensor_cores = false;
    profile.has_fp16 = false;
    profile.has_int8 = false;
    profile.has_unified_memory = false;
    profile.supports_fp64 = false;
    
    // Check device capabilities
    if (props.type == GGML_BACKEND_DEVICE_TYPE_GPU) {
        // Estimate some values based on memory size if not provided by backend
        if (profile.memory_bandwidth_gbps == 0.0f) {
            // Rough estimate: ~10-15 GB/s per GB of VRAM
            profile.memory_bandwidth_gbps = (profile.total_memory_bytes / (1024.0f * 1024.0f * 1024.0f)) * 12.0f;
        }
        
        if (profile.theoretical_tflops == 0.0f) {
            profile.theoretical_tflops = estimate_tflops_from_memory(profile.total_memory_bytes);
        }
    }
}

bool GgmlGpuProfiler::is_compatible_device(ggml_backend_dev_t device) const {
    // Get device properties
    struct ggml_backend_dev_props props;
    ggml_backend_dev_get_props(device, &props);
    
    // Only interested in GPU devices by default
    if (props.type != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return false;
    }
    
    // Check backend type
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    if (reg) {
        const char* backend_name = ggml_backend_reg_name(reg);
        // Derived classes will override to check for specific backend
        return backend_name != nullptr;
    }
    
    return false;
}

ggml_backend_dev_t GgmlGpuProfiler::get_device_by_index(int device_id) const {
    size_t device_count = ggml_backend_dev_count();
    int compatible_index = 0;
    
    for (size_t i = 0; i < device_count; ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (!device) continue;
        
        if (is_compatible_device(device)) {
            if (compatible_index == device_id) {
                return device;
            }
            compatible_index++;
        }
    }
    
    return nullptr;
}

ggml_backend_t GgmlGpuProfiler::init_backend_from_device(ggml_backend_dev_t device) const {
    return ggml_backend_dev_init(device, nullptr);
}

float GgmlGpuProfiler::benchmark_gpu_performance(int device_id) {
    ggml_backend_dev_t device = get_device_by_index(device_id);
    if (!device) {
        return 0.0f;
    }
    
    return run_performance_benchmark(device);
}

LayerProfile GgmlGpuProfiler::estimate_layer_profile(
    const std::string& layer_type,
    int layer_index,
    size_t hidden_size,
    size_t sequence_length,
    size_t vocab_size) {
    
    LayerProfile profile;
    profile.layer_type = layer_type;
    profile.layer_index = layer_index;
    
    if (layer_type == "attention" || layer_type == "self_attention") {
        // Attention layer: Q, K, V projections + attention computation
        size_t qkv_params = 3 * hidden_size * hidden_size;
        size_t attention_params = hidden_size * hidden_size;  // Output projection
        profile.memory_requirement_bytes = (qkv_params + attention_params) * sizeof(float);
        
        // FLOPs for attention
        size_t attention_flops = profiling_utils::estimate_attention_flops(
            sequence_length, hidden_size, hidden_size / 64);  // Assume 64 dims per head
        profile.compute_intensity = (float)attention_flops / profile.memory_requirement_bytes;
        
        profile.attention_weight = 1.0f;
        profile.feedforward_weight = 0.0f;
        
    } else if (layer_type == "feedforward" || layer_type == "mlp") {
        // Feedforward layer: typically 4x hidden size
        size_t ff_size = 4 * hidden_size;
        size_t ff_params = 2 * hidden_size * ff_size;  // Two linear layers
        profile.memory_requirement_bytes = ff_params * sizeof(float);
        
        // FLOPs for feedforward
        size_t ff_flops = profiling_utils::estimate_feedforward_flops(
            sequence_length, hidden_size, ff_size);
        profile.compute_intensity = (float)ff_flops / profile.memory_requirement_bytes;
        
        profile.attention_weight = 0.0f;
        profile.feedforward_weight = 1.0f;
        
    } else if (layer_type == "embedding" && vocab_size > 0) {
        profile.memory_requirement_bytes = vocab_size * hidden_size * sizeof(float);
        profile.compute_intensity = 1.0f;  // Simple lookup
        profile.activation_weight = 1.0f;
    }
    
    // Estimate bandwidth requirement
    profile.memory_bandwidth_requirement_gbps = profiling_utils::estimate_bandwidth_requirement(profile);
    
    return profile;
}

float GgmlGpuProfiler::calculate_gpu_score(const GpuProfile& profile) const {
    // Normalize to 0-100 scale
    float memory_score = std::min(100.0f, (profile.total_memory_bytes / (40.0f * 1024 * 1024 * 1024)) * 100);
    float compute_score = std::min(100.0f, profile.measured_performance_score * 5.0f);  // Assuming 20 TFLOPS = 100
    float bandwidth_score = std::min(100.0f, (profile.memory_bandwidth_gbps / 1000.0f) * 100);
    
    float feature_score = 0.0f;
    if (profile.has_tensor_cores) feature_score += 40.0f;
    if (profile.has_fp16) feature_score += 30.0f;
    if (profile.has_int8) feature_score += 30.0f;
    
    return memory_score * MEMORY_CAPACITY_WEIGHT + 
           compute_score * COMPUTE_CAPABILITY_WEIGHT + 
           bandwidth_score * MEMORY_BANDWIDTH_WEIGHT + 
           feature_score * SPECIAL_FEATURES_WEIGHT;
}

float GgmlGpuProfiler::calculate_relative_performance(
    const GpuProfile& gpu1, 
    const GpuProfile& gpu2) const {
    
    if (gpu2.measured_performance_score == 0) return 1.0f;
    return gpu1.measured_performance_score / gpu2.measured_performance_score;
}

void GgmlGpuProfiler::cache_profile(const GpuProfile& profile) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    pImpl->cleanup_cache_if_needed(profile_cache_);
    profile_cache_[profile.device_id] = profile;
}

bool GgmlGpuProfiler::get_cached_profile(int device_id, GpuProfile& profile) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = profile_cache_.find(device_id);
    if (it != profile_cache_.end()) {
        profile = it->second;
        return true;
    }
    return false;
}

void GgmlGpuProfiler::clear_cache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    profile_cache_.clear();
}

float GgmlGpuProfiler::estimate_tflops_from_memory(size_t memory_bytes) const {
    // Rough estimation based on typical GPU memory to compute ratios
    // 8GB VRAM ~ 10 TFLOPS, 16GB ~ 20 TFLOPS, 24GB ~ 30 TFLOPS, etc.
    float memory_gb = memory_bytes / (1024.0f * 1024.0f * 1024.0f);
    return memory_gb * 1.25f;  // ~1.25 TFLOPS per GB (conservative)
}

float GgmlGpuProfiler::run_ggml_benchmark(ggml_backend_dev_t device) {
    // Initialize backend
    ggml_backend_t backend = init_backend_from_device(device);
    if (!backend) {
        return 0.0f;
    }
    
    float tflops = 0.0f;
    
    try {
        // Create a simple GGML graph for benchmarking
        const int n = 4096;  // Matrix size
        const int n_iterations = 10;
        
        // Initialize GGML context
        struct ggml_init_params params = {};
        params.mem_size   = 256 * 1024 * 1024;  // 256 MB
        params.mem_buffer = nullptr;
        params.no_alloc   = true;
        
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            ggml_backend_free(backend);
            return 0.0f;
        }
        
        // Create tensors
        struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
        struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
        struct ggml_tensor* c = ggml_mul_mat(ctx, a, b);
        
        // Create compute graph
        struct ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, c);
        
        // Allocate tensors
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) {
            ggml_free(ctx);
            ggml_backend_free(backend);
            return 0.0f;
        }
        
        // Initialize with random data (simplified)
        // In real implementation, would use proper random initialization
        
        // Warm up
        ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < n_iterations; ++i) {
            ggml_backend_graph_compute(backend, graph);
        }
        ggml_backend_synchronize(backend);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Calculate TFLOPS
        size_t flops_per_mul = 2LL * n * n * n;
        double total_flops = (double)flops_per_mul * n_iterations;
        double seconds = duration.count() / 1e6;
        tflops = (float)(total_flops / 1e12 / seconds);
        
        // Cleanup
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    } catch (...) {
        tflops = 0.0f;
    }
    
    ggml_backend_free(backend);
    return tflops;
}

void GgmlGpuProfiler::log_gpu_profile(const GpuProfile& profile) const {
    LLAMA_LOG_INFO("\n[GPU Profile] Device %d (%s): %s\n", 
                   profile.device_id, profile.backend_type.c_str(), profile.name.c_str());
    LLAMA_LOG_INFO("  Memory: %s total, %s available\n",
                   profiling_utils::format_memory_size(profile.total_memory_bytes).c_str(),
                   profiling_utils::format_memory_size(profile.available_memory_bytes).c_str());
    
    if (profile.compute_capability_major > 0) {
        LLAMA_LOG_INFO("  Compute Capability: %s\n", 
                       profiling_utils::format_compute_capability(
                           profile.compute_capability_major, 
                           profile.compute_capability_minor).c_str());
    }
    
    if (profile.sm_count > 0) {
        LLAMA_LOG_INFO("  SMs/CUs: %d, Max Clock: %d MHz\n", 
                       profile.sm_count, profile.max_clock_mhz);
    }
    
    if (profile.memory_bandwidth_gbps > 0) {
        LLAMA_LOG_INFO("  Memory Bandwidth: %.1f GB/s\n", profile.memory_bandwidth_gbps);
    }
    
    LLAMA_LOG_INFO("  Features: ");
    if (profile.has_tensor_cores) LLAMA_LOG_INFO("TensorCores ");
    if (profile.has_fp16) LLAMA_LOG_INFO("FP16 ");
    if (profile.has_int8) LLAMA_LOG_INFO("INT8 ");
    if (profile.has_unified_memory) LLAMA_LOG_INFO("UnifiedMemory ");
    if (profile.supports_fp64) LLAMA_LOG_INFO("FP64 ");
    LLAMA_LOG_INFO("\n");
    
    if (profile.theoretical_tflops > 0) {
        LLAMA_LOG_INFO("  Theoretical Performance: %.2f TFLOPS\n", profile.theoretical_tflops);
    }
    LLAMA_LOG_INFO("  Measured Performance: %.2f TFLOPS\n", profile.measured_performance_score);
    LLAMA_LOG_INFO("  Capability Score: %.1f/100\n", calculate_gpu_score(profile));
}

void GgmlGpuProfiler::log_profiling_summary(const ProfilingResult& result) const {
    LLAMA_LOG_INFO("\n[%s Profiling Summary]\n", get_backend_type().c_str());
    LLAMA_LOG_INFO("  Total GPUs profiled: %zu\n", result.gpu_profiles.size());
    LLAMA_LOG_INFO("  Profiling duration: %ld us\n", result.profiling_duration.count());
    
    if (!result.gpu_profiles.empty()) {
        // Sort by capability score
        std::vector<GpuProfile> sorted_profiles = result.gpu_profiles;
        std::sort(sorted_profiles.begin(), sorted_profiles.end(),
                  [this](const GpuProfile& a, const GpuProfile& b) {
                      return calculate_gpu_score(a) > calculate_gpu_score(b);
                  });
        
        LLAMA_LOG_INFO("\n  GPU Ranking by Capability:\n");
        for (size_t i = 0; i < sorted_profiles.size(); ++i) {
            const auto& gpu = sorted_profiles[i];
            LLAMA_LOG_INFO("    %zu. GPU %d (%s): Score %.1f\n",
                          i + 1, gpu.device_id, gpu.name.c_str(), calculate_gpu_score(gpu));
        }
    }
}

} // namespace orchestration
} // namespace llama