#include "cuda_gpu_profiler.h"
#include "llama-impl.h"

#ifdef GGML_USE_CUDA
#include <curand.h>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#endif

namespace llama {
namespace orchestration {

#ifdef GGML_USE_CUDA
// CUDA implementation details
struct CudaGpuProfiler::CudaImpl {
    cublasHandle_t cublas_handle = nullptr;
    cudaStream_t stream = nullptr;
    
    CudaImpl() {
        // Initialize cuBLAS for benchmarking
        cublasCreate(&cublas_handle);
        
        // Create CUDA stream for async operations
        cudaStreamCreate(&stream);
    }
    
    ~CudaImpl() {
        if (stream) {
            cudaStreamDestroy(stream);
        }
        if (cublas_handle) {
            cublasDestroy(cublas_handle);
        }
    }
};
#endif

CudaGpuProfiler::CudaGpuProfiler() 
#ifdef GGML_USE_CUDA
    : cuda_impl_(std::make_unique<CudaImpl>())
#endif
{
}

CudaGpuProfiler::~CudaGpuProfiler() = default;

bool CudaGpuProfiler::supports_feature([[maybe_unused]] const std::string& feature_name) const {
#ifdef GGML_USE_CUDA
    if (feature_name == "tensor_cores") return true;
    if (feature_name == "cuda_graphs") return true;
    if (feature_name == "memory_pools") return true;
    if (feature_name == "multi_gpu") return true;
#endif
    return false;
}

std::unordered_map<std::string, std::string> CudaGpuProfiler::get_capabilities() const {
    std::unordered_map<std::string, std::string> caps;
    
#ifdef GGML_USE_CUDA
    int runtime_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    caps["cuda_runtime_version"] = std::to_string(runtime_version);
    
    int driver_version = 0;
    cudaDriverGetVersion(&driver_version);
    caps["cuda_driver_version"] = std::to_string(driver_version);
    
    caps["backend"] = "CUDA";
    caps["multi_gpu_support"] = "true";
#else
    caps["backend"] = "CUDA (not compiled)";
    caps["multi_gpu_support"] = "false";
#endif
    
    return caps;
}

bool CudaGpuProfiler::is_compatible_device(ggml_backend_dev_t device) const {
    struct ggml_backend_dev_props props;
    ggml_backend_dev_get_props(device, &props);
    
    // Check if it's a GPU device
    if (props.type != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return false;
    }
    
    // Check if it's a CUDA backend
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    if (reg) {
        const char* backend_name = ggml_backend_reg_name(reg);
        return backend_name && std::string(backend_name) == "CUDA";
    }
    
    return false;
}

void CudaGpuProfiler::query_extended_properties([[maybe_unused]] ggml_backend_dev_t device, GpuProfile& profile) {
#ifdef GGML_USE_CUDA
    // Get CUDA device ID from profile (already set by base class)
    int cuda_device_id = profile.device_id;
    
    // Query CUDA-specific properties
    query_cuda_device_properties(cuda_device_id, profile);
    
    // Measure memory bandwidth
    measure_memory_bandwidth(cuda_device_id, profile);
    
    // Calculate theoretical performance
    profile.theoretical_tflops = profile.calculate_theoretical_tflops();
#else
    // No CUDA support compiled in
    profile.backend_type = "CUDA (not available)";
#endif
}

float CudaGpuProfiler::run_performance_benchmark(ggml_backend_dev_t device) {
#ifdef GGML_USE_CUDA
    // Get device index
    struct ggml_backend_dev_props props;
    ggml_backend_dev_get_props(device, &props);
    
    // Find CUDA device index
    int cuda_device_id = -1;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp cuda_props;
        cudaGetDeviceProperties(&cuda_props, i);
        if (std::string(cuda_props.name) == std::string(props.name)) {
            cuda_device_id = i;
            break;
        }
    }
    
    if (cuda_device_id < 0) {
        // Fallback to GGML benchmark
        return run_ggml_benchmark(device);
    }
    
    // Run CUDA-specific benchmark
    GpuProfile temp_profile;
    temp_profile.device_id = cuda_device_id;
    run_compute_benchmark(cuda_device_id, temp_profile);
    return temp_profile.measured_performance_score;
#else
    // Fallback to GGML benchmark
    return run_ggml_benchmark(device);
#endif
}

#ifdef GGML_USE_CUDA
void CudaGpuProfiler::query_cuda_device_properties(int device_id, GpuProfile& profile) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    
    profile.name = std::string(prop.name);
    profile.total_memory_bytes = prop.totalGlobalMem;
    profile.compute_capability_major = prop.major;
    profile.compute_capability_minor = prop.minor;
    profile.sm_count = prop.multiProcessorCount;
    profile.max_clock_mhz = prop.clockRate / 1000;
    
    // Check for special features
    profile.has_tensor_cores = (prop.major >= 7);  // Volta and newer
    profile.has_fp16 = (prop.major >= 5 && prop.minor >= 3);  // Maxwell 2.0+
    profile.has_int8 = (prop.major >= 6 && prop.minor >= 1);  // Pascal+
    profile.has_unified_memory = prop.unifiedAddressing;
    profile.supports_fp64 = (prop.major >= 1 && prop.minor >= 3);  // Compute 1.3+
    
    // Query available memory
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    profile.available_memory_bytes = free_mem;
    
    // Theoretical memory bandwidth (GB/s)
    profile.memory_bandwidth_gbps = (prop.memoryBusWidth / 8.0f) * 
                                   (prop.memoryClockRate / 1000.0f) / 1000.0f * 2.0f;
}

void CudaGpuProfiler::measure_memory_bandwidth(int device_id, GpuProfile& profile) {
    const size_t test_size = 256 * 1024 * 1024;  // 256 MB
    void* d_src = nullptr;
    void* d_dst = nullptr;
    
    cudaSetDevice(device_id);
    cudaMalloc(&d_src, test_size);
    cudaMalloc(&d_dst, test_size);
    
    // Warm up
    cudaMemcpy(d_dst, d_src, test_size, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    
    // Measure bandwidth
    const int num_iterations = 10;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; ++i) {
        cudaMemcpy(d_dst, d_src, test_size, cudaMemcpyDeviceToDevice);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Calculate bandwidth in GB/s
    float bandwidth_gbps = (test_size * num_iterations / (1024.0f * 1024.0f * 1024.0f)) / 
                          (milliseconds / 1000.0f);
    
    // Use measured bandwidth if it's reasonable, otherwise use theoretical
    if (bandwidth_gbps > 0 && bandwidth_gbps < profile.memory_bandwidth_gbps * 1.5f) {
        profile.memory_bandwidth_gbps = bandwidth_gbps * 0.8f;  // 80% efficiency factor
    }
    
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void CudaGpuProfiler::run_compute_benchmark(int device_id, GpuProfile& profile) {
    const int matrix_size = 4096;
    const int num_iterations = 100;
    
    cudaSetDevice(device_id);
    cublasSetStream(cuda_impl_->cublas_handle, 0);
    
    // Allocate matrices
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    
    size_t matrix_bytes = matrix_size * matrix_size * sizeof(float);
    cudaMalloc(&d_A, matrix_bytes);
    cudaMalloc(&d_B, matrix_bytes);
    cudaMalloc(&d_C, matrix_bytes);
    
    // Initialize with random data
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateUniform(gen, d_A, matrix_size * matrix_size);
    curandGenerateUniform(gen, d_B, matrix_size * matrix_size);
    
    // Warm up
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cuda_impl_->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                matrix_size, matrix_size, matrix_size,
                &alpha, d_A, matrix_size, d_B, matrix_size,
                &beta, d_C, matrix_size);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; ++i) {
        cublasSgemm(cuda_impl_->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    matrix_size, matrix_size, matrix_size,
                    &alpha, d_A, matrix_size, d_B, matrix_size,
                    &beta, d_C, matrix_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Calculate TFLOPS
    size_t flops_per_gemm = 2LL * matrix_size * matrix_size * matrix_size;
    double total_flops = (double)flops_per_gemm * num_iterations;
    double tflops = (total_flops / 1e12) / (milliseconds / 1000.0);
    
    profile.measured_performance_score = (float)tflops;
    
    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    curandDestroyGenerator(gen);
}
#endif // GGML_USE_CUDA

} // namespace orchestration
} // namespace llama