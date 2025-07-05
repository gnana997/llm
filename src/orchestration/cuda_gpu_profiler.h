#pragma once

#include "ggml_gpu_profiler.h"

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace llama {
namespace orchestration {

/**
 * CUDA-specific GPU profiler implementation
 * 
 * This class provides detailed GPU profiling for NVIDIA GPUs using CUDA APIs.
 * It extends the base GGML profiler with CUDA-specific features like compute
 * capability detection, tensor core support, and cuBLAS benchmarking.
 */
class CudaGpuProfiler : public GgmlGpuProfiler {
public:
    CudaGpuProfiler();
    ~CudaGpuProfiler() override;
    
    // Backend identification
    std::string get_backend_type() const override { return "CUDA"; }
    
    // Optional: CUDA-specific features
    bool supports_feature(const std::string& feature_name) const override;
    std::unordered_map<std::string, std::string> get_capabilities() const override;

protected:
    // Override base class methods for CUDA-specific implementation
    void query_extended_properties(ggml_backend_dev_t device, GpuProfile& profile) override;
    float run_performance_benchmark(ggml_backend_dev_t device) override;
    bool is_compatible_device(ggml_backend_dev_t device) const override;

private:
#ifdef GGML_USE_CUDA
    // CUDA-specific helper methods (migrated from original implementation)
    void query_cuda_device_properties(int device_id, GpuProfile& profile);
    void measure_memory_bandwidth(int device_id, GpuProfile& profile);
    void run_compute_benchmark(int device_id, GpuProfile& profile);
    
    // Internal implementation details
    struct CudaImpl;
    std::unique_ptr<CudaImpl> cuda_impl_;
#endif
};

} // namespace orchestration
} // namespace llama