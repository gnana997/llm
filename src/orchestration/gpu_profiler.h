#pragma once

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <chrono>

namespace llama {
namespace orchestration {

// GPU capability profile structure
struct GpuProfile {
    int device_id;
    std::string name;
    std::string backend_type;  // "CUDA", "ROCm", "Metal", etc.
    size_t total_memory_bytes;
    size_t available_memory_bytes;
    float memory_bandwidth_gbps;
    int compute_capability_major;
    int compute_capability_minor;
    int sm_count;  // Streaming Multiprocessor count
    int max_clock_mhz;
    float theoretical_tflops;
    bool has_tensor_cores;
    bool has_fp16;
    bool has_int8;
    bool has_unified_memory;  // For APUs and Metal
    bool supports_fp64;       // Double precision support
    float measured_performance_score;  // Runtime benchmark score
    
    // Derived metrics
    float get_compute_capability() const {
        return compute_capability_major + compute_capability_minor * 0.1f;
    }
    
    // Calculate theoretical performance in TFLOPS
    float calculate_theoretical_tflops() const;
    
    // Get a normalized score (0-100) for this GPU
    float get_capability_score() const;
};

// Layer computational requirements profile
struct LayerProfile {
    std::string layer_type;  // attention, feedforward, etc.
    int layer_index;
    size_t memory_requirement_bytes;
    float compute_intensity;  // FLOPs per byte
    float memory_bandwidth_requirement_gbps;
    float estimated_runtime_ms;
    
    // Weight factors for different operations
    float attention_weight = 0.0f;
    float feedforward_weight = 0.0f;
    float activation_weight = 0.0f;
};

// GPU profiling result with timing
struct ProfilingResult {
    std::vector<GpuProfile> gpu_profiles;
    std::chrono::microseconds profiling_duration;
    bool success;
    std::string error_message;
};

class GpuProfiler {
public:
    GpuProfiler();
    ~GpuProfiler();
    
    // Profile all available GPUs
    ProfilingResult profile_all_gpus();
    
    // Profile specific GPU
    GpuProfile profile_gpu(int device_id);
    
    // Run performance benchmark on GPU
    float benchmark_gpu_performance(int device_id);
    
    // Estimate layer profile based on model parameters
    LayerProfile estimate_layer_profile(
        const std::string& layer_type,
        int layer_index,
        size_t hidden_size,
        size_t sequence_length,
        size_t vocab_size = 0
    );
    
    // Calculate GPU score based on various factors
    float calculate_gpu_score(const GpuProfile& profile) const;
    
    // Compare GPUs for heterogeneous setups
    float calculate_relative_performance(
        const GpuProfile& gpu1, 
        const GpuProfile& gpu2
    ) const;
    
    // Cache profiling results
    void cache_profile(const GpuProfile& profile);
    bool get_cached_profile(int device_id, GpuProfile& profile) const;
    
    // Logging helpers
    void log_gpu_profile(const GpuProfile& profile) const;
    void log_profiling_summary(const ProfilingResult& result) const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
    
    // Internal methods
    void query_cuda_device_properties(int device_id, GpuProfile& profile);
    void measure_memory_bandwidth(int device_id, GpuProfile& profile);
    void run_compute_benchmark(int device_id, GpuProfile& profile);
    
    // Cache for profiling results
    mutable std::unordered_map<int, GpuProfile> profile_cache_;
    
    // Performance weights for scoring
    static constexpr float MEMORY_CAPACITY_WEIGHT = 0.35f;
    static constexpr float COMPUTE_CAPABILITY_WEIGHT = 0.30f;
    static constexpr float MEMORY_BANDWIDTH_WEIGHT = 0.20f;
    static constexpr float SPECIAL_FEATURES_WEIGHT = 0.15f;
};

// Helper functions for profiling
namespace profiling_utils {
    // Estimate FLOPs for transformer operations
    size_t estimate_attention_flops(size_t seq_len, size_t hidden_size, size_t num_heads);
    size_t estimate_feedforward_flops(size_t seq_len, size_t hidden_size, size_t ff_size);
    
    // Memory bandwidth requirements
    float estimate_bandwidth_requirement(const LayerProfile& layer);
    
    // Format memory size for logging
    std::string format_memory_size(size_t bytes);
    
    // Format compute capability
    std::string format_compute_capability(int major, int minor);
}

} // namespace orchestration
} // namespace llama