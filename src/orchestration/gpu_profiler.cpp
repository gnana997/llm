#include "gpu_profiler.h"
#include "gpu_profiler_factory.h"
#include "unified_gpu_profiler.h"
#include "llama-impl.h"

#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace llama {
namespace orchestration {

// Implementation details - now using unified profiler
struct GpuProfiler::Impl {
    std::unique_ptr<IGpuProfiler> profiler;
    
    Impl() {
        // Initialize the factory
        GpuProfilerFactory::initialize_backends();
        
        // Create unified profiler that supports all backends
        profiler = GpuProfilerFactory::create_unified();
        
        if (!profiler) {
            LLAMA_LOG_INFO("[GPU Profiler] Failed to create unified profiler, using fallback\n");
            profiler = GpuProfilerFactory::create_for_backend("Generic");
        }
    }
};

GpuProfiler::GpuProfiler() : pImpl(std::make_unique<Impl>()) {
    LLAMA_LOG_INFO("[GPU Profiler] Initialized with backend: %s\n", 
                   pImpl->profiler ? pImpl->profiler->get_backend_type().c_str() : "None");
}
GpuProfiler::~GpuProfiler() = default;

ProfilingResult GpuProfiler::profile_all_gpus() {
    if (!pImpl->profiler) {
        ProfilingResult result;
        result.success = false;
        result.error_message = "No profiler available";
        return result;
    }
    
    return pImpl->profiler->profile_all_gpus();
}

GpuProfile GpuProfiler::profile_gpu(int device_id) {
    // Check cache first
    GpuProfile profile;
    if (get_cached_profile(device_id, profile)) {
        return profile;
    }
    
    if (!pImpl->profiler) {
        profile.device_id = device_id;
        profile.name = "No profiler";
        profile.total_memory_bytes = 0;
        profile.available_memory_bytes = 0;
        profile.measured_performance_score = 0.0f;
        return profile;
    }
    
    profile = pImpl->profiler->profile_gpu(device_id);
    
    // Cache the result
    cache_profile(profile);
    
    return profile;
}

// CUDA-specific methods have been moved to CudaGpuProfiler
// These are kept for backward compatibility but now just log warnings

void GpuProfiler::query_cuda_device_properties([[maybe_unused]] int device_id, [[maybe_unused]] GpuProfile& profile) {
    LLAMA_LOG_INFO("[GPU Profiler] Warning: query_cuda_device_properties is deprecated\n");
}

void GpuProfiler::measure_memory_bandwidth([[maybe_unused]] int device_id, [[maybe_unused]] GpuProfile& profile) {
    LLAMA_LOG_INFO("[GPU Profiler] Warning: measure_memory_bandwidth is deprecated\n");
}

void GpuProfiler::run_compute_benchmark([[maybe_unused]] int device_id, [[maybe_unused]] GpuProfile& profile) {
    LLAMA_LOG_INFO("[GPU Profiler] Warning: run_compute_benchmark is deprecated\n");
}

float GpuProfiler::benchmark_gpu_performance(int device_id) {
    if (!pImpl->profiler) {
        return 0.0f;
    }
    
    return pImpl->profiler->benchmark_gpu_performance(device_id);
}

LayerProfile GpuProfiler::estimate_layer_profile(
    const std::string& layer_type,
    int layer_index,
    size_t hidden_size,
    size_t sequence_length,
    size_t vocab_size) {
    
    if (!pImpl->profiler) {
        LayerProfile profile;
        profile.layer_type = layer_type;
        profile.layer_index = layer_index;
        return profile;
    }
    
    return pImpl->profiler->estimate_layer_profile(
        layer_type, layer_index, hidden_size, sequence_length, vocab_size);
}

float GpuProfile::calculate_theoretical_tflops() const {
    // Base FP32 cores
    float fp32_tflops = (sm_count * 64 * 2 * max_clock_mhz) / 1e9f;
    
    // Tensor core boost (if available)
    if (has_tensor_cores) {
        if (compute_capability_major >= 8) {  // Ampere
            fp32_tflops *= 2.0f;  // Sparse tensor cores
        } else if (compute_capability_major >= 7) {  // Volta/Turing
            fp32_tflops *= 1.5f;
        }
    }
    
    return fp32_tflops;
}

float GpuProfile::get_capability_score() const {
    // Normalize to 0-100 scale
    float memory_score = std::min(100.0f, (total_memory_bytes / (40.0f * 1024 * 1024 * 1024)) * 100);
    float compute_score = std::min(100.0f, measured_performance_score * 5.0f);  // Assuming 20 TFLOPS = 100
    float bandwidth_score = std::min(100.0f, (memory_bandwidth_gbps / 1000.0f) * 100);
    
    float feature_score = 0.0f;
    if (has_tensor_cores) feature_score += 40.0f;
    if (has_fp16) feature_score += 30.0f;
    if (has_int8) feature_score += 30.0f;
    
    return memory_score * 0.35f + compute_score * 0.30f + 
           bandwidth_score * 0.20f + feature_score * 0.15f;
}

float GpuProfiler::calculate_gpu_score(const GpuProfile& profile) const {
    if (!pImpl->profiler) {
        return profile.get_capability_score();
    }
    
    return pImpl->profiler->calculate_gpu_score(profile);
}

float GpuProfiler::calculate_relative_performance(
    const GpuProfile& gpu1, 
    const GpuProfile& gpu2) const {
    
    if (!pImpl->profiler) {
        if (gpu2.measured_performance_score == 0) return 1.0f;
        return gpu1.measured_performance_score / gpu2.measured_performance_score;
    }
    
    return pImpl->profiler->calculate_relative_performance(gpu1, gpu2);
}

void GpuProfiler::cache_profile(const GpuProfile& profile) {
    profile_cache_[profile.device_id] = profile;
}

bool GpuProfiler::get_cached_profile(int device_id, GpuProfile& profile) const {
    auto it = profile_cache_.find(device_id);
    if (it != profile_cache_.end()) {
        profile = it->second;
        return true;
    }
    return false;
}

void GpuProfiler::log_gpu_profile(const GpuProfile& profile) const {
    LLAMA_LOG_INFO("\n[GPU Profile] Device %d: %s\n", profile.device_id, profile.name.c_str());
    LLAMA_LOG_INFO("  Compute Capability: %s\n", 
            profiling_utils::format_compute_capability(
                profile.compute_capability_major, 
                profile.compute_capability_minor).c_str());
    LLAMA_LOG_INFO("  Memory: %s total, %s available\n",
            profiling_utils::format_memory_size(profile.total_memory_bytes).c_str(),
            profiling_utils::format_memory_size(profile.available_memory_bytes).c_str());
    LLAMA_LOG_INFO("  SMs: %d, Max Clock: %d MHz\n", profile.sm_count, profile.max_clock_mhz);
    LLAMA_LOG_INFO("  Memory Bandwidth: %.1f GB/s\n", profile.memory_bandwidth_gbps);
    LLAMA_LOG_INFO("  Features: TensorCores=%s, FP16=%s, INT8=%s\n",
            profile.has_tensor_cores ? "Yes" : "No",
            profile.has_fp16 ? "Yes" : "No",
            profile.has_int8 ? "Yes" : "No");
    LLAMA_LOG_INFO("  Theoretical Performance: %.2f TFLOPS\n", profile.theoretical_tflops);
    LLAMA_LOG_INFO("  Measured Performance: %.2f TFLOPS\n", profile.measured_performance_score);
    LLAMA_LOG_INFO("  Capability Score: %.1f/100\n", profile.get_capability_score());
}

void GpuProfiler::log_profiling_summary(const ProfilingResult& result) const {
    LLAMA_LOG_INFO("\n[GPU Profiling Summary]\n");
    LLAMA_LOG_INFO("  Total GPUs profiled: %zu\n", result.gpu_profiles.size());
    LLAMA_LOG_INFO("  Profiling duration: %ld us\n", result.profiling_duration.count());
    
    if (!result.gpu_profiles.empty()) {
        // Sort by capability score
        std::vector<GpuProfile> sorted_profiles = result.gpu_profiles;
        std::sort(sorted_profiles.begin(), sorted_profiles.end(),
                  [](const GpuProfile& a, const GpuProfile& b) {
                      return a.get_capability_score() > b.get_capability_score();
                  });
        
        LLAMA_LOG_INFO("\n  GPU Ranking by Capability:\n");
        for (size_t i = 0; i < sorted_profiles.size(); ++i) {
            const auto& gpu = sorted_profiles[i];
            LLAMA_LOG_INFO("    %zu. GPU %d (%s): Score %.1f\n",
                    i + 1, gpu.device_id, gpu.name.c_str(), gpu.get_capability_score());
        }
    }
}

// Utility functions implementation
namespace profiling_utils {

size_t estimate_attention_flops(size_t seq_len, size_t hidden_size, [[maybe_unused]] size_t num_heads) {
    // QKV projections: 3 * seq_len * hidden_size * hidden_size
    size_t qkv_flops = 3 * seq_len * hidden_size * hidden_size * 2;
    
    // Attention scores: seq_len * seq_len * hidden_size
    size_t attention_flops = seq_len * seq_len * hidden_size * 2;
    
    // Output projection: seq_len * hidden_size * hidden_size
    size_t output_flops = seq_len * hidden_size * hidden_size * 2;
    
    return qkv_flops + attention_flops + output_flops;
}

size_t estimate_feedforward_flops(size_t seq_len, size_t hidden_size, size_t ff_size) {
    // First linear: seq_len * hidden_size * ff_size * 2
    // Second linear: seq_len * ff_size * hidden_size * 2
    // Activation: seq_len * ff_size
    return 2 * seq_len * hidden_size * ff_size * 2 + seq_len * ff_size;
}

float estimate_bandwidth_requirement(const LayerProfile& layer) {
    // Estimate based on memory access patterns
    // Assume we need to read weights once and activations multiple times
    float bandwidth_multiplier = 2.5f;  // Read weights + read/write activations
    
    if (layer.layer_type == "attention") {
        bandwidth_multiplier = 3.0f;  // More memory intensive
    }
    
    // GB/s = (memory_bytes * multiplier) / estimated_runtime_seconds
    float estimated_runtime_s = 0.001f;  // 1ms baseline
    return (layer.memory_requirement_bytes * bandwidth_multiplier) / 
           (1024.0f * 1024.0f * 1024.0f * estimated_runtime_s);
}

std::string format_memory_size(size_t bytes) {
    std::ostringstream oss;
    if (bytes >= 1024 * 1024 * 1024) {
        oss << std::fixed << std::setprecision(2) 
            << (bytes / (1024.0 * 1024.0 * 1024.0)) << " GB";
    } else if (bytes >= 1024 * 1024) {
        oss << std::fixed << std::setprecision(2) 
            << (bytes / (1024.0 * 1024.0)) << " MB";
    } else {
        oss << bytes << " bytes";
    }
    return oss.str();
}

std::string format_compute_capability(int major, int minor) {
    std::ostringstream oss;
    oss << major << "." << minor;
    
    // Add architecture name
    if (major == 9) oss << " (Hopper)";
    else if (major == 8) oss << " (Ampere)";
    else if (major == 7) {
        if (minor == 5) oss << " (Turing)";
        else oss << " (Volta)";
    }
    else if (major == 6) oss << " (Pascal)";
    else if (major == 5) oss << " (Maxwell)";
    
    return oss.str();
}

} // namespace profiling_utils

} // namespace orchestration
} // namespace llama