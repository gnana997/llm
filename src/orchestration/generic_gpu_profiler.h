#pragma once

#include "ggml_gpu_profiler.h"

namespace llama {
namespace orchestration {

/**
 * Generic GPU profiler implementation
 * 
 * This profiler provides basic functionality using only GGML backend APIs.
 * It serves as a fallback for backends without specific implementations
 * and provides conservative performance estimates.
 */
class GenericGpuProfiler : public GgmlGpuProfiler {
public:
    GenericGpuProfiler();
    ~GenericGpuProfiler() override = default;
    
    // Backend identification
    std::string get_backend_type() const override { return "Generic"; }
    
    // Override to indicate limited feature support
    bool supports_feature(const std::string& feature_name) const override;
    std::unordered_map<std::string, std::string> get_capabilities() const override;

protected:
    // Implement required virtual methods
    void query_extended_properties(ggml_backend_dev_t device, GpuProfile& profile) override;
    float run_performance_benchmark(ggml_backend_dev_t device) override;
    bool is_compatible_device(ggml_backend_dev_t device) const override;
    
private:
    // Helper methods
    void estimate_device_capabilities(GpuProfile& profile);
    float estimate_performance_from_memory(size_t memory_bytes);
    void detect_backend_features(ggml_backend_dev_t device, GpuProfile& profile);
};

} // namespace orchestration
} // namespace llama