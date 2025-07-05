#include "rocm_gpu_profiler.h"
#include "llama-impl.h"

#ifdef GGML_USE_HIP
#include <hiprand/hiprand.h>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#endif

namespace llama {
namespace orchestration {

#ifdef GGML_USE_HIP
// ROCm implementation details
struct RocmGpuProfiler::RocmImpl {
    rocblas_handle rocblas_handle = nullptr;
    hipStream_t stream = nullptr;
    
    RocmImpl() {
        // Initialize rocBLAS for benchmarking
        rocblas_create_handle(&rocblas_handle);
        
        // Create HIP stream for async operations
        hipStreamCreate(&stream);
    }
    
    ~RocmImpl() {
        if (stream) {
            hipStreamDestroy(stream);
        }
        if (rocblas_handle) {
            rocblas_destroy_handle(rocblas_handle);
        }
    }
};
#endif

RocmGpuProfiler::RocmGpuProfiler() 
#ifdef GGML_USE_HIP
    : rocm_impl_(std::make_unique<RocmImpl>())
#endif
{
    LLAMA_LOG_INFO("[ROCm GPU Profiler] Initialized\n");
}

RocmGpuProfiler::~RocmGpuProfiler() = default;

bool RocmGpuProfiler::supports_feature([[maybe_unused]] const std::string& feature_name) const {
#ifdef GGML_USE_HIP
    if (feature_name == "matrix_cores") return true;  // MI100+ have matrix cores
    if (feature_name == "memory_pools") return true;
    if (feature_name == "multi_gpu") return true;
    if (feature_name == "unified_memory") return true;
#endif
    return false;
}

std::unordered_map<std::string, std::string> RocmGpuProfiler::get_capabilities() const {
    std::unordered_map<std::string, std::string> caps;
    
#ifdef GGML_USE_HIP
    int runtime_version = 0;
    hipRuntimeGetVersion(&runtime_version);
    caps["hip_runtime_version"] = std::to_string(runtime_version);
    
    int driver_version = 0;
    hipDriverGetVersion(&driver_version);
    caps["hip_driver_version"] = std::to_string(driver_version);
    
    caps["backend"] = "ROCm/HIP";
    caps["multi_gpu_support"] = "true";
    caps["infinity_fabric"] = "true";  // AMD's interconnect
#else
    caps["backend"] = "ROCm (not compiled)";
    caps["multi_gpu_support"] = "false";
#endif
    
    return caps;
}

bool RocmGpuProfiler::is_compatible_device(ggml_backend_dev_t device) const {
    struct ggml_backend_dev_props props;
    ggml_backend_dev_get_props(device, &props);
    
    // Check if it's a GPU device
    if (props.type != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return false;
    }
    
    // Check if it's a HIP/ROCm backend
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    if (reg) {
        const char* backend_name = ggml_backend_reg_name(reg);
        return backend_name && (std::string(backend_name) == "HIP" || 
                               std::string(backend_name) == "ROCm");
    }
    
    return false;
}

void RocmGpuProfiler::query_extended_properties([[maybe_unused]] ggml_backend_dev_t device, GpuProfile& profile) {
#ifdef GGML_USE_HIP
    // Get HIP device ID from profile (already set by base class)
    int hip_device_id = profile.device_id;
    
    // Query HIP-specific properties
    query_hip_device_properties(hip_device_id, profile);
    
    // Measure memory bandwidth
    measure_memory_bandwidth(hip_device_id, profile);
    
    // Calculate theoretical performance
    profile.theoretical_tflops = profile.calculate_theoretical_tflops();
#else
    // No HIP support compiled in
    profile.backend_type = "ROCm (not available)";
#endif
}

float RocmGpuProfiler::run_performance_benchmark(ggml_backend_dev_t device) {
#ifdef GGML_USE_HIP
    // Get device index
    struct ggml_backend_dev_props props;
    ggml_backend_dev_get_props(device, &props);
    
    // Find HIP device index
    int hip_device_id = -1;
    int device_count = 0;
    hipGetDeviceCount(&device_count);
    
    for (int i = 0; i < device_count; ++i) {
        hipDeviceProp_t hip_props;
        hipGetDeviceProperties(&hip_props, i);
        if (std::string(hip_props.name) == std::string(props.name)) {
            hip_device_id = i;
            break;
        }
    }
    
    if (hip_device_id < 0) {
        // Fallback to GGML benchmark
        return run_ggml_benchmark(device);
    }
    
    // Run HIP-specific benchmark
    GpuProfile temp_profile;
    temp_profile.device_id = hip_device_id;
    run_compute_benchmark(hip_device_id, temp_profile);
    return temp_profile.measured_performance_score;
#else
    // Fallback to GGML benchmark
    return run_ggml_benchmark(device);
#endif
}

#ifdef GGML_USE_HIP
void RocmGpuProfiler::query_hip_device_properties(int device_id, GpuProfile& profile) {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device_id);
    
    profile.name = std::string(prop.name);
    profile.total_memory_bytes = prop.totalGlobalMem;
    profile.compute_capability_major = prop.major;
    profile.compute_capability_minor = prop.minor;
    profile.sm_count = prop.multiProcessorCount;  // Compute Units in AMD terminology
    profile.max_clock_mhz = prop.clockRate / 1000;
    
    // Detect GPU architecture
    std::string arch = detect_gpu_architecture(prop);
    
    // Check for special features based on architecture
    profile.has_tensor_cores = false;  // Default
    profile.has_fp16 = true;  // All modern AMD GPUs support FP16
    profile.has_int8 = true;  // Most support INT8
    profile.has_unified_memory = prop.managedMemory;
    profile.supports_fp64 = (prop.major >= 3);  // GCN 3.0+
    
    // AMD-specific: Matrix cores on CDNA architecture (MI100+)
    if (arch.find("gfx908") != std::string::npos ||  // MI100
        arch.find("gfx90a") != std::string::npos ||  // MI200 series
        arch.find("gfx940") != std::string::npos ||  // MI300 series
        arch.find("gfx941") != std::string::npos ||
        arch.find("gfx942") != std::string::npos) {
        profile.has_tensor_cores = true;  // Matrix cores in AMD terminology
    }
    
    // Query available memory
    size_t free_mem, total_mem;
    hipMemGetInfo(&free_mem, &total_mem);
    profile.available_memory_bytes = free_mem;
    
    // Theoretical memory bandwidth (GB/s)
    // Note: AMD GPUs often have HBM with very high bandwidth
    profile.memory_bandwidth_gbps = (prop.memoryBusWidth / 8.0f) * 
                                   (prop.memoryClockRate / 1000.0f) / 1000.0f * 2.0f;
    
    // Add architecture info to backend type
    profile.backend_type = "ROCm (" + arch + ")";
}

void RocmGpuProfiler::measure_memory_bandwidth(int device_id, GpuProfile& profile) {
    const size_t test_size = 256 * 1024 * 1024;  // 256 MB
    void* d_src = nullptr;
    void* d_dst = nullptr;
    
    hipSetDevice(device_id);
    hipMalloc(&d_src, test_size);
    hipMalloc(&d_dst, test_size);
    
    // Warm up
    hipMemcpy(d_dst, d_src, test_size, hipMemcpyDeviceToDevice);
    hipDeviceSynchronize();
    
    // Measure bandwidth
    const int num_iterations = 10;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    
    hipEventRecord(start);
    for (int i = 0; i < num_iterations; ++i) {
        hipMemcpy(d_dst, d_src, test_size, hipMemcpyDeviceToDevice);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    
    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);
    
    // Calculate bandwidth in GB/s
    float bandwidth_gbps = (test_size * num_iterations / (1024.0f * 1024.0f * 1024.0f)) / 
                          (milliseconds / 1000.0f);
    
    // Use measured bandwidth if it's reasonable, otherwise use theoretical
    if (bandwidth_gbps > 0 && bandwidth_gbps < profile.memory_bandwidth_gbps * 1.5f) {
        profile.memory_bandwidth_gbps = bandwidth_gbps * 0.8f;  // 80% efficiency factor
    }
    
    hipFree(d_src);
    hipFree(d_dst);
    hipEventDestroy(start);
    hipEventDestroy(stop);
}

void RocmGpuProfiler::run_compute_benchmark(int device_id, GpuProfile& profile) {
    const int matrix_size = 4096;
    const int num_iterations = 100;
    
    hipSetDevice(device_id);
    rocblas_set_stream(rocm_impl_->rocblas_handle, rocm_impl_->stream);
    
    // Allocate matrices
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    
    size_t matrix_bytes = matrix_size * matrix_size * sizeof(float);
    hipMalloc(&d_A, matrix_bytes);
    hipMalloc(&d_B, matrix_bytes);
    hipMalloc(&d_C, matrix_bytes);
    
    // Initialize with random data
    hiprandGenerator_t gen;
    hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT);
    hiprandGenerateUniform(gen, d_A, matrix_size * matrix_size);
    hiprandGenerateUniform(gen, d_B, matrix_size * matrix_size);
    
    // Warm up
    float alpha = 1.0f, beta = 0.0f;
    rocblas_sgemm(rocm_impl_->rocblas_handle, rocblas_operation_none, rocblas_operation_none,
                  matrix_size, matrix_size, matrix_size,
                  &alpha, d_A, matrix_size, d_B, matrix_size,
                  &beta, d_C, matrix_size);
    hipDeviceSynchronize();
    
    // Benchmark
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    
    hipEventRecord(start);
    for (int i = 0; i < num_iterations; ++i) {
        rocblas_sgemm(rocm_impl_->rocblas_handle, rocblas_operation_none, rocblas_operation_none,
                      matrix_size, matrix_size, matrix_size,
                      &alpha, d_A, matrix_size, d_B, matrix_size,
                      &beta, d_C, matrix_size);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    
    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);
    
    // Calculate TFLOPS
    size_t flops_per_gemm = 2LL * matrix_size * matrix_size * matrix_size;
    double total_flops = (double)flops_per_gemm * num_iterations;
    double tflops = (total_flops / 1e12) / (milliseconds / 1000.0);
    
    profile.measured_performance_score = (float)tflops;
    
    // Clean up
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hiprandDestroyGenerator(gen);
}

std::string RocmGpuProfiler::detect_gpu_architecture(const hipDeviceProp_t& prop) {
    // Extract GCN/RDNA/CDNA architecture from the name
    std::string arch_name = prop.gcnArchName;
    
    // Common AMD GPU architectures
    if (arch_name.find("gfx90") != std::string::npos) {
        if (arch_name == "gfx906") return "Vega 20 (MI50/MI60)";
        if (arch_name == "gfx908") return "CDNA 1 (MI100)";
        if (arch_name == "gfx90a") return "CDNA 2 (MI200 series)";
        if (arch_name == "gfx940" || arch_name == "gfx941" || arch_name == "gfx942") {
            return "CDNA 3 (MI300 series)";
        }
    } else if (arch_name.find("gfx10") != std::string::npos) {
        if (arch_name == "gfx1030") return "RDNA 2 (RX 6000 series)";
        if (arch_name == "gfx1031") return "RDNA 2 (RX 6000M series)";
        if (arch_name == "gfx1032") return "RDNA 2 (RX 6400/6500)";
        if (arch_name == "gfx1100") return "RDNA 3 (RX 7900 series)";
        if (arch_name == "gfx1101") return "RDNA 3 (RX 7000 series)";
        if (arch_name == "gfx1102") return "RDNA 3 (RX 7000 series)";
    } else if (arch_name.find("gfx80") != std::string::npos) {
        return "GCN 3.0 (Fiji/Polaris)";
    } else if (arch_name.find("gfx90") != std::string::npos) {
        return "GCN 5.0 (Vega)";
    }
    
    return arch_name;  // Return raw architecture name if not recognized
}
#endif // GGML_USE_HIP

} // namespace orchestration
} // namespace llama