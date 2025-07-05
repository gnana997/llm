#pragma once

#include "gpu_profiler_interface.h"
#include "gpu_profiler.h"
#include "ggml-backend.h"
#include <unordered_map>
#include <mutex>

namespace llama {
namespace orchestration {

/**
 * Base implementation of GPU profiler using GGML backend abstraction
 * 
 * This class provides common functionality for all GPU backends by using
 * GGML's device enumeration and property query APIs. Backend-specific
 * implementations can override methods to provide more detailed information.
 */
class GgmlGpuProfiler : public IGpuProfiler {
public:
    GgmlGpuProfiler();
    virtual ~GgmlGpuProfiler();
    
    // IGpuProfiler interface implementation
    ProfilingResult profile_all_gpus() override;
    GpuProfile profile_gpu(int device_id) override;
    bool multi_gpu_supported() const override;
    
    // Performance measurement - base implementation
    float benchmark_gpu_performance(int device_id) override;
    
    // Layer profiling - common implementation
    LayerProfile estimate_layer_profile(
        const std::string& layer_type,
        int layer_index,
        size_t hidden_size,
        size_t sequence_length,
        size_t vocab_size = 0
    ) override;
    
    // Utility methods
    float calculate_gpu_score(const GpuProfile& profile) const override;
    float calculate_relative_performance(
        const GpuProfile& gpu1, 
        const GpuProfile& gpu2
    ) const override;

protected:
    /**
     * Query basic device properties using GGML APIs
     * This method fills in properties available through GGML backend
     */
    virtual void query_basic_properties(ggml_backend_dev_t device, GpuProfile& profile);
    
    /**
     * Query extended device properties specific to the backend
     * Derived classes should override this to add backend-specific information
     */
    virtual void query_extended_properties(ggml_backend_dev_t device, GpuProfile& profile) = 0;
    
    /**
     * Run backend-specific performance benchmark
     * Derived classes should implement their own benchmarking logic
     */
    virtual float run_performance_benchmark(ggml_backend_dev_t device) = 0;
    
    /**
     * Check if a device matches the backend type this profiler handles
     */
    virtual bool is_compatible_device(ggml_backend_dev_t device) const;
    
    /**
     * Get GGML device by index for this backend
     */
    ggml_backend_dev_t get_device_by_index(int device_id) const;
    
    /**
     * Initialize GGML backend from device
     */
    ggml_backend_t init_backend_from_device(ggml_backend_dev_t device) const;

    // Caching support
    void cache_profile(const GpuProfile& profile);
    bool get_cached_profile(int device_id, GpuProfile& profile) const;
    void clear_cache();

    // Logging helpers
    void log_gpu_profile(const GpuProfile& profile) const;
    void log_profiling_summary(const ProfilingResult& result) const;
    
    // Performance estimation helpers
    float estimate_tflops_from_memory(size_t memory_bytes) const;
    float run_ggml_benchmark(ggml_backend_dev_t device);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
    
    // Cache for profiling results
    mutable std::unordered_map<int, GpuProfile> profile_cache_;
    mutable std::mutex cache_mutex_;
    
    // Performance weights for scoring (same as original)
    static constexpr float MEMORY_CAPACITY_WEIGHT = 0.35f;
    static constexpr float COMPUTE_CAPABILITY_WEIGHT = 0.30f;
    static constexpr float MEMORY_BANDWIDTH_WEIGHT = 0.20f;
    static constexpr float SPECIAL_FEATURES_WEIGHT = 0.15f;
};

} // namespace orchestration
} // namespace llama