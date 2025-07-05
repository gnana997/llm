#pragma once

#include "metrics.h"
#include "ggml.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <string>
#include <memory>
#include <vector>

namespace llama {
namespace metrics {
namespace gpu {

// GPU device information
struct gpu_device_info {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    size_t used_memory;
    double utilization_percent;
    double temperature_celsius;
    double power_watts;
    double memory_bandwidth_gbps;
    int compute_capability_major;
    int compute_capability_minor;
};

// GPU metrics collector interface
class gpu_metrics_collector {
public:
    virtual ~gpu_metrics_collector() = default;
    
    // Initialize metrics for a specific device
    virtual void initialize(int device_id) = 0;
    
    // Collect current metrics
    virtual gpu_device_info collect() = 0;
    
    // Update metric registry with current values
    virtual void update_metrics() = 0;
    
protected:
    int device_id_;
    std::string device_name_;
};

#ifdef GGML_USE_CUDA
// CUDA-specific metrics collector
class cuda_metrics_collector : public gpu_metrics_collector {
public:
    cuda_metrics_collector() : initialized_(false) {}
    
    void initialize(int device_id) override;
    gpu_device_info collect() override;
    void update_metrics() override;
    
private:
    bool initialized_;
    void* nvml_device_;  // nvmlDevice_t handle
    
    // Metric references for this device
    std::shared_ptr<gauge> memory_used_metric_;
    std::shared_ptr<gauge> memory_free_metric_;
    std::shared_ptr<gauge> memory_total_metric_;
    std::shared_ptr<gauge> utilization_metric_;
    std::shared_ptr<gauge> temperature_metric_;
    std::shared_ptr<gauge> power_metric_;
    std::shared_ptr<gauge> memory_bandwidth_metric_;
    
    // Operation counters
    std::shared_ptr<counter> allocations_metric_;
    std::shared_ptr<counter> deallocations_metric_;
    std::shared_ptr<counter> h2d_transfers_metric_;
    std::shared_ptr<counter> d2h_transfers_metric_;
    std::shared_ptr<counter> d2d_transfers_metric_;
    std::shared_ptr<counter> kernel_launches_metric_;
    
    // Operation histograms
    std::shared_ptr<histogram> allocation_size_metric_;
    std::shared_ptr<histogram> h2d_transfer_size_metric_;
    std::shared_ptr<histogram> d2h_transfer_size_metric_;
    std::shared_ptr<histogram> kernel_duration_metric_;
};
#endif

// GPU metrics manager - handles multiple devices
class gpu_metrics_manager {
public:
    static gpu_metrics_manager& instance() {
        static gpu_metrics_manager instance;
        return instance;
    }
    
    // Initialize metrics for all available devices
    void initialize();
    
    // Update metrics for all devices
    void update_all();
    
    // Update metrics for specific device
    void update_device(int device_id);
    
    // Get device info
    gpu_device_info get_device_info(int device_id);
    
    // Track GPU operations
    void track_allocation(int device_id, size_t bytes);
    void track_deallocation(int device_id, size_t bytes);
    void track_h2d_transfer(int device_id, size_t bytes, int64_t duration_us);
    void track_d2h_transfer(int device_id, size_t bytes, int64_t duration_us);
    void track_d2d_transfer(int src_device, int dst_device, size_t bytes, int64_t duration_us);
    void track_kernel_launch(int device_id, const std::string& kernel_name, int64_t duration_us);
    
    // Log GPU metrics summary
    void log_summary();
    void log_detailed();
    
    // Get total GPU memory usage across all devices
    size_t get_total_memory_used();
    size_t get_total_memory_available();
    
private:
    gpu_metrics_manager();
    gpu_metrics_manager(const gpu_metrics_manager&) = delete;
    gpu_metrics_manager& operator=(const gpu_metrics_manager&) = delete;
    
    std::vector<std::unique_ptr<gpu_metrics_collector>> collectors_;
    std::mutex mutex_;
    bool initialized_;
    
    // Global GPU metrics
    std::shared_ptr<gauge> total_gpu_memory_used_;
    std::shared_ptr<gauge> total_gpu_memory_available_;
    std::shared_ptr<gauge> active_gpu_count_;
    std::shared_ptr<histogram> gpu_utilization_distribution_;
};

// Helper macros for GPU metrics
#define METRIC_GPU_ALLOC(device_id, bytes) \
    llama::metrics::gpu::gpu_metrics_manager::instance().track_allocation(device_id, bytes)

#define METRIC_GPU_FREE(device_id, bytes) \
    llama::metrics::gpu::gpu_metrics_manager::instance().track_deallocation(device_id, bytes)

#define METRIC_GPU_H2D(device_id, bytes, duration_us) \
    llama::metrics::gpu::gpu_metrics_manager::instance().track_h2d_transfer(device_id, bytes, duration_us)

#define METRIC_GPU_D2H(device_id, bytes, duration_us) \
    llama::metrics::gpu::gpu_metrics_manager::instance().track_d2h_transfer(device_id, bytes, duration_us)

#define METRIC_GPU_D2D(src_device, dst_device, bytes, duration_us) \
    llama::metrics::gpu::gpu_metrics_manager::instance().track_d2d_transfer(src_device, dst_device, bytes, duration_us)

#define METRIC_GPU_KERNEL(device_id, kernel_name, duration_us) \
    llama::metrics::gpu::gpu_metrics_manager::instance().track_kernel_launch(device_id, kernel_name, duration_us)

#define METRIC_GPU_UPDATE() \
    llama::metrics::gpu::gpu_metrics_manager::instance().update_all()

#define METRIC_GPU_UPDATE_DEVICE(device_id) \
    llama::metrics::gpu::gpu_metrics_manager::instance().update_device(device_id)

// RAII GPU operation timer
class gpu_operation_timer {
public:
    enum operation_type {
        H2D_TRANSFER,
        D2H_TRANSFER,
        D2D_TRANSFER,
        KERNEL_LAUNCH
    };
    
    gpu_operation_timer(int device_id, operation_type type, size_t bytes = 0, 
                       const std::string& name = "", int dst_device = -1)
        : device_id_(device_id), type_(type), bytes_(bytes), 
          name_(name), dst_device_(dst_device),
          start_(std::chrono::high_resolution_clock::now()) {}
    
    ~gpu_operation_timer() {
        auto duration = std::chrono::high_resolution_clock::now() - start_;
        auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        
        switch (type_) {
            case H2D_TRANSFER:
                METRIC_GPU_H2D(device_id_, bytes_, microseconds);
                break;
            case D2H_TRANSFER:
                METRIC_GPU_D2H(device_id_, bytes_, microseconds);
                break;
            case D2D_TRANSFER:
                METRIC_GPU_D2D(device_id_, dst_device_, bytes_, microseconds);
                break;
            case KERNEL_LAUNCH:
                METRIC_GPU_KERNEL(device_id_, name_, microseconds);
                break;
        }
    }
    
private:
    int device_id_;
    operation_type type_;
    size_t bytes_;
    std::string name_;
    int dst_device_;
    std::chrono::high_resolution_clock::time_point start_;
};

#define GPU_METRIC_H2D_TIMER(device_id, bytes) \
    llama::metrics::gpu::gpu_operation_timer _gpu_timer_##__LINE__(device_id, \
        llama::metrics::gpu::gpu_operation_timer::H2D_TRANSFER, bytes)

#define GPU_METRIC_D2H_TIMER(device_id, bytes) \
    llama::metrics::gpu::gpu_operation_timer _gpu_timer_##__LINE__(device_id, \
        llama::metrics::gpu::gpu_operation_timer::D2H_TRANSFER, bytes)

#define GPU_METRIC_D2D_TIMER(src_device, dst_device, bytes) \
    llama::metrics::gpu::gpu_operation_timer _gpu_timer_##__LINE__(src_device, \
        llama::metrics::gpu::gpu_operation_timer::D2D_TRANSFER, bytes, "", dst_device)

#define GPU_METRIC_KERNEL_TIMER(device_id, kernel_name) \
    llama::metrics::gpu::gpu_operation_timer _gpu_timer_##__LINE__(device_id, \
        llama::metrics::gpu::gpu_operation_timer::KERNEL_LAUNCH, 0, kernel_name)

} // namespace gpu
} // namespace metrics
} // namespace llama