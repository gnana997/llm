#pragma once

#include "ggml_gpu_profiler.h"

#ifdef GGML_USE_HIP
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#endif

namespace llama {
namespace orchestration {

/**
 * ROCm/HIP-specific GPU profiler implementation
 * 
 * This class provides detailed GPU profiling for AMD GPUs using HIP APIs.
 * HIP (Heterogeneous-compute Interface for Portability) is AMD's answer to CUDA,
 * providing similar functionality with nearly identical API.
 */
class RocmGpuProfiler : public GgmlGpuProfiler {
public:
    RocmGpuProfiler();
    ~RocmGpuProfiler() override;
    
    // Backend identification
    std::string get_backend_type() const override { return "ROCm"; }
    
    // ROCm-specific features
    bool supports_feature(const std::string& feature_name) const override;
    std::unordered_map<std::string, std::string> get_capabilities() const override;

protected:
    // Override base class methods for ROCm-specific implementation
    void query_extended_properties(ggml_backend_dev_t device, GpuProfile& profile) override;
    float run_performance_benchmark(ggml_backend_dev_t device) override;
    bool is_compatible_device(ggml_backend_dev_t device) const override;

private:
#ifdef GGML_USE_HIP
    // ROCm-specific helper methods
    void query_hip_device_properties(int device_id, GpuProfile& profile);
    void measure_memory_bandwidth(int device_id, GpuProfile& profile);
    void run_compute_benchmark(int device_id, GpuProfile& profile);
    
    // Detect AMD GPU architecture
    std::string detect_gpu_architecture(const hipDeviceProp_t& prop);
    
    // Internal implementation details
    struct RocmImpl;
    std::unique_ptr<RocmImpl> rocm_impl_;
#endif
};

} // namespace orchestration
} // namespace llama