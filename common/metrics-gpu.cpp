#include "metrics-gpu.h"
#include "../src/log/log-ex.h"
#include <sstream>
#include <iomanip>

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
// Note: NVML support would require linking with nvidia-ml library
// For now, we'll use CUDA runtime API for basic metrics
#endif

namespace llama {
namespace metrics {
namespace gpu {

#ifdef GGML_USE_CUDA
void cuda_metrics_collector::initialize(int device_id) {
    device_id_ = device_id;
    
    // Get device properties
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        LOG_CAT(COMMON_LOG_CAT_GPU, GGML_LOG_LEVEL_WARN, 
                "Failed to get CUDA device properties for device %d: %s\n", 
                device_id, cudaGetErrorString(err));
        return;
    }
    
    device_name_ = prop.name;
    
    // Create metric names with device suffix
    std::string suffix = "_gpu" + std::to_string(device_id);
    
    // Register device-specific metrics
    auto& registry = metric_registry::instance();
    
    // Memory metrics
    memory_used_metric_ = registry.register_metric<gauge>(
        "gpu_memory_used_bytes" + suffix, 
        "GPU memory used in bytes for device " + std::to_string(device_id));
    memory_free_metric_ = registry.register_metric<gauge>(
        "gpu_memory_free_bytes" + suffix,
        "GPU memory free in bytes for device " + std::to_string(device_id));
    memory_total_metric_ = registry.register_metric<gauge>(
        "gpu_memory_total_bytes" + suffix,
        "GPU memory total in bytes for device " + std::to_string(device_id));
    
    // Utilization metrics
    utilization_metric_ = registry.register_metric<gauge>(
        "gpu_utilization_percent" + suffix,
        "GPU utilization percentage for device " + std::to_string(device_id));
    temperature_metric_ = registry.register_metric<gauge>(
        "gpu_temperature_celsius" + suffix,
        "GPU temperature in Celsius for device " + std::to_string(device_id));
    power_metric_ = registry.register_metric<gauge>(
        "gpu_power_watts" + suffix,
        "GPU power consumption in watts for device " + std::to_string(device_id));
    memory_bandwidth_metric_ = registry.register_metric<gauge>(
        "gpu_memory_bandwidth_gbps" + suffix,
        "GPU memory bandwidth in GB/s for device " + std::to_string(device_id));
    
    // Operation counters
    allocations_metric_ = registry.register_metric<counter>(
        "gpu_allocations_total" + suffix,
        "Total GPU memory allocations for device " + std::to_string(device_id));
    deallocations_metric_ = registry.register_metric<counter>(
        "gpu_deallocations_total" + suffix,
        "Total GPU memory deallocations for device " + std::to_string(device_id));
    h2d_transfers_metric_ = registry.register_metric<counter>(
        "gpu_h2d_transfers_total" + suffix,
        "Total host to device transfers for device " + std::to_string(device_id));
    d2h_transfers_metric_ = registry.register_metric<counter>(
        "gpu_d2h_transfers_total" + suffix,
        "Total device to host transfers for device " + std::to_string(device_id));
    d2d_transfers_metric_ = registry.register_metric<counter>(
        "gpu_d2d_transfers_total" + suffix,
        "Total device to device transfers for device " + std::to_string(device_id));
    kernel_launches_metric_ = registry.register_metric<counter>(
        "gpu_kernel_launches_total" + suffix,
        "Total kernel launches for device " + std::to_string(device_id));
    
    // Operation histograms
    allocation_size_metric_ = registry.register_metric<histogram>(
        "gpu_allocation_size_bytes" + suffix,
        "GPU memory allocation size distribution for device " + std::to_string(device_id),
        histogram::default_memory_buckets());
    h2d_transfer_size_metric_ = registry.register_metric<histogram>(
        "gpu_h2d_transfer_size_bytes" + suffix,
        "Host to device transfer size distribution for device " + std::to_string(device_id),
        histogram::default_memory_buckets());
    d2h_transfer_size_metric_ = registry.register_metric<histogram>(
        "gpu_d2h_transfer_size_bytes" + suffix,
        "Device to host transfer size distribution for device " + std::to_string(device_id),
        histogram::default_memory_buckets());
    kernel_duration_metric_ = registry.register_metric<histogram>(
        "gpu_kernel_duration_us" + suffix,
        "GPU kernel execution duration for device " + std::to_string(device_id),
        std::vector<double>{10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000});
    
    // Set initial total memory
    memory_total_metric_->set(static_cast<double>(prop.totalGlobalMem));
    
    initialized_ = true;
    
    LOG_CAT(COMMON_LOG_CAT_GPU, GGML_LOG_LEVEL_INFO,
            "Initialized GPU metrics for device %d: %s (%.1f GB)\n",
            device_id, device_name_.c_str(), prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
}

gpu_device_info cuda_metrics_collector::collect() {
    gpu_device_info info{};
    info.device_id = device_id_;
    info.name = device_name_;
    
    if (!initialized_) {
        return info;
    }
    
    // Save current device
    int current_device;
    cudaGetDevice(&current_device);
    cudaSetDevice(device_id_);
    
    // Get memory info
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err == cudaSuccess) {
        info.total_memory = total_mem;
        info.free_memory = free_mem;
        info.used_memory = total_mem - free_mem;
    } else {
        LOG_CAT(COMMON_LOG_CAT_GPU, GGML_LOG_LEVEL_WARN,
                "Failed to get memory info for device %d: %s\n",
                device_id_, cudaGetErrorString(err));
    }
    
    // Get device properties for compute capability
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device_id_);
    if (err == cudaSuccess) {
        info.compute_capability_major = prop.major;
        info.compute_capability_minor = prop.minor;
        
        // Calculate theoretical memory bandwidth
        // Memory clock rate is in kHz, bus width in bits
        info.memory_bandwidth_gbps = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    }
    
    // Note: For utilization, temperature, and power, we would need NVML
    // These are set to 0 for now
    info.utilization_percent = 0;
    info.temperature_celsius = 0;
    info.power_watts = 0;
    
    // Restore original device
    cudaSetDevice(current_device);
    
    return info;
}

void cuda_metrics_collector::update_metrics() {
    if (!initialized_) return;
    
    auto info = collect();
    
    // Update memory metrics
    memory_used_metric_->set(static_cast<double>(info.used_memory));
    memory_free_metric_->set(static_cast<double>(info.free_memory));
    memory_total_metric_->set(static_cast<double>(info.total_memory));
    
    // Update other metrics (when available)
    if (info.utilization_percent > 0) {
        utilization_metric_->set(info.utilization_percent);
    }
    if (info.temperature_celsius > 0) {
        temperature_metric_->set(info.temperature_celsius);
    }
    if (info.power_watts > 0) {
        power_metric_->set(info.power_watts);
    }
    if (info.memory_bandwidth_gbps > 0) {
        memory_bandwidth_metric_->set(info.memory_bandwidth_gbps);
    }
}
#endif // GGML_USE_CUDA

// GPU metrics manager implementation
gpu_metrics_manager::gpu_metrics_manager() : initialized_(false) {
    // Register global GPU metrics
    auto& registry = metric_registry::instance();
    
    total_gpu_memory_used_ = registry.register_metric<gauge>(
        "gpu_memory_used_bytes_total",
        "Total GPU memory used across all devices");
    total_gpu_memory_available_ = registry.register_metric<gauge>(
        "gpu_memory_available_bytes_total",
        "Total GPU memory available across all devices");
    active_gpu_count_ = registry.register_metric<gauge>(
        "gpu_devices_active",
        "Number of active GPU devices");
    gpu_utilization_distribution_ = registry.register_metric<histogram>(
        "gpu_utilization_percent_all",
        "Distribution of GPU utilization across all devices",
        std::vector<double>{0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100});
}

void gpu_metrics_manager::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) return;
    
#ifdef GGML_USE_CUDA
    int device_count = ggml_backend_cuda_get_device_count();
    
    LOG_CAT(COMMON_LOG_CAT_GPU, GGML_LOG_LEVEL_INFO,
            "Initializing GPU metrics for %d devices\n", device_count);
    
    collectors_.clear();
    collectors_.reserve(device_count);
    
    for (int i = 0; i < device_count; ++i) {
        auto collector = std::make_unique<cuda_metrics_collector>();
        collector->initialize(i);
        collectors_.push_back(std::move(collector));
    }
    
    active_gpu_count_->set(static_cast<double>(device_count));
#else
    LOG_CAT(COMMON_LOG_CAT_GPU, GGML_LOG_LEVEL_INFO,
            "GPU metrics not available - CUDA not enabled\n");
    active_gpu_count_->set(0);
#endif
    
    initialized_ = true;
}

void gpu_metrics_manager::update_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        initialize();
    }
    
    size_t total_used = 0;
    size_t total_available = 0;
    
    for (auto& collector : collectors_) {
        collector->update_metrics();
        auto info = collector->collect();
        
        total_used += info.used_memory;
        total_available += info.total_memory;
        
        if (info.utilization_percent > 0) {
            gpu_utilization_distribution_->observe(info.utilization_percent);
        }
    }
    
    total_gpu_memory_used_->set(static_cast<double>(total_used));
    total_gpu_memory_available_->set(static_cast<double>(total_available));
}

void gpu_metrics_manager::update_device(int device_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        initialize();
    }
    
    if (device_id >= 0 && device_id < static_cast<int>(collectors_.size())) {
        collectors_[device_id]->update_metrics();
    }
}

gpu_device_info gpu_metrics_manager::get_device_info(int device_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        initialize();
    }
    
    if (device_id >= 0 && device_id < static_cast<int>(collectors_.size())) {
        return collectors_[device_id]->collect();
    }
    
    return gpu_device_info{};
}

void gpu_metrics_manager::track_allocation(int device_id, size_t bytes) {
    if (device_id < 0) return;
    
    std::string suffix = "_gpu" + std::to_string(device_id);
    auto counter = get_counter("gpu_allocations_total" + suffix);
    auto histogram = get_histogram("gpu_allocation_size_bytes" + suffix);
    
    if (counter) counter->increment();
    if (histogram) histogram->observe(static_cast<double>(bytes));
    
    LOG_TRACEV(4, "GPU%d: Allocated %zu bytes\n", device_id, bytes);
}

void gpu_metrics_manager::track_deallocation(int device_id, size_t bytes) {
    if (device_id < 0) return;
    
    std::string suffix = "_gpu" + std::to_string(device_id);
    auto counter = get_counter("gpu_deallocations_total" + suffix);
    
    if (counter) counter->increment();
    
    LOG_TRACEV(4, "GPU%d: Deallocated %zu bytes\n", device_id, bytes);
}

void gpu_metrics_manager::track_h2d_transfer(int device_id, size_t bytes, int64_t duration_us) {
    if (device_id < 0) return;
    
    std::string suffix = "_gpu" + std::to_string(device_id);
    auto counter = get_counter("gpu_h2d_transfers_total" + suffix);
    auto histogram = get_histogram("gpu_h2d_transfer_size_bytes" + suffix);
    
    if (counter) counter->increment();
    if (histogram) histogram->observe(static_cast<double>(bytes));
    
    LOG_TRACEV(3, "GPU%d: H2D transfer %zu bytes in %lldμs (%.2f GB/s)\n",
               device_id, bytes, duration_us,
               (bytes / 1e9) / (duration_us / 1e6));
}

void gpu_metrics_manager::track_d2h_transfer(int device_id, size_t bytes, int64_t duration_us) {
    if (device_id < 0) return;
    
    std::string suffix = "_gpu" + std::to_string(device_id);
    auto counter = get_counter("gpu_d2h_transfers_total" + suffix);
    auto histogram = get_histogram("gpu_d2h_transfer_size_bytes" + suffix);
    
    if (counter) counter->increment();
    if (histogram) histogram->observe(static_cast<double>(bytes));
    
    LOG_TRACEV(3, "GPU%d: D2H transfer %zu bytes in %lldμs (%.2f GB/s)\n",
               device_id, bytes, duration_us,
               (bytes / 1e9) / (duration_us / 1e6));
}

void gpu_metrics_manager::track_d2d_transfer(int src_device, int dst_device, size_t bytes, int64_t duration_us) {
    if (src_device < 0 || dst_device < 0) return;
    
    std::string suffix = "_gpu" + std::to_string(src_device);
    auto counter = get_counter("gpu_d2d_transfers_total" + suffix);
    
    if (counter) counter->increment();
    
    LOG_TRACEV(3, "GPU%d->GPU%d: D2D transfer %zu bytes in %lldμs (%.2f GB/s)\n",
               src_device, dst_device, bytes, duration_us,
               (bytes / 1e9) / (duration_us / 1e6));
}

void gpu_metrics_manager::track_kernel_launch(int device_id, const std::string& kernel_name, int64_t duration_us) {
    if (device_id < 0) return;
    
    std::string suffix = "_gpu" + std::to_string(device_id);
    auto counter = get_counter("gpu_kernel_launches_total" + suffix);
    auto histogram = get_histogram("gpu_kernel_duration_us" + suffix);
    
    if (counter) counter->increment();
    if (histogram) histogram->observe(static_cast<double>(duration_us));
    
    LOG_TRACEV(4, "GPU%d: Kernel '%s' executed in %lldμs\n",
               device_id, kernel_name.c_str(), duration_us);
}

void gpu_metrics_manager::log_summary() {
    update_all();
    
    LOG_PERF("=== GPU Metrics Summary ===\n");
    
    for (size_t i = 0; i < collectors_.size(); ++i) {
        auto info = collectors_[i]->collect();
        
        LOG_PERF("GPU%zu (%s):\n", i, info.name.c_str());
        LOG_PERF("  Memory: %.2f GB used / %.2f GB total (%.1f%%)\n",
                 info.used_memory / (1024.0 * 1024.0 * 1024.0),
                 info.total_memory / (1024.0 * 1024.0 * 1024.0),
                 100.0 * info.used_memory / info.total_memory);
        
        if (info.utilization_percent > 0) {
            LOG_PERF("  Utilization: %.1f%%\n", info.utilization_percent);
        }
        if (info.temperature_celsius > 0) {
            LOG_PERF("  Temperature: %.1f°C\n", info.temperature_celsius);
        }
        if (info.power_watts > 0) {
            LOG_PERF("  Power: %.1f W\n", info.power_watts);
        }
        if (info.memory_bandwidth_gbps > 0) {
            LOG_PERF("  Memory Bandwidth: %.1f GB/s\n", info.memory_bandwidth_gbps);
        }
        
        LOG_PERF("  Compute Capability: %d.%d\n",
                 info.compute_capability_major, info.compute_capability_minor);
    }
    
    LOG_PERF("Total GPU Memory: %.2f GB used / %.2f GB available\n",
             total_gpu_memory_used_->get() / (1024.0 * 1024.0 * 1024.0),
             total_gpu_memory_available_->get() / (1024.0 * 1024.0 * 1024.0));
}

void gpu_metrics_manager::log_detailed() {
    log_summary();
    
    LOG_PERF("\n=== GPU Operations Metrics ===\n");
    
    for (size_t i = 0; i < collectors_.size(); ++i) {
        std::string suffix = "_gpu" + std::to_string(i);
        
        LOG_PERF("GPU%zu Operations:\n", i);
        
        // Counters
        auto allocs = get_counter("gpu_allocations_total" + suffix);
        auto deallocs = get_counter("gpu_deallocations_total" + suffix);
        auto h2d = get_counter("gpu_h2d_transfers_total" + suffix);
        auto d2h = get_counter("gpu_d2h_transfers_total" + suffix);
        auto d2d = get_counter("gpu_d2d_transfers_total" + suffix);
        auto kernels = get_counter("gpu_kernel_launches_total" + suffix);
        
        if (allocs) LOG_PERF("  Allocations: %.0f\n", allocs->get());
        if (deallocs) LOG_PERF("  Deallocations: %.0f\n", deallocs->get());
        if (h2d) LOG_PERF("  H2D Transfers: %.0f\n", h2d->get());
        if (d2h) LOG_PERF("  D2H Transfers: %.0f\n", d2h->get());
        if (d2d) LOG_PERF("  D2D Transfers: %.0f\n", d2d->get());
        if (kernels) LOG_PERF("  Kernel Launches: %.0f\n", kernels->get());
        
        // Histograms
        auto alloc_hist = get_histogram("gpu_allocation_size_bytes" + suffix);
        if (alloc_hist) {
            auto values = alloc_hist->get_values();
            double count = values["gpu_allocation_size_bytes" + suffix + "_count"];
            double avg = values["gpu_allocation_size_bytes" + suffix + "_avg"];
            if (count > 0) {
                LOG_PERF("  Avg Allocation Size: %.2f MB\n", avg / (1024.0 * 1024.0));
            }
        }
        
        auto kernel_hist = get_histogram("gpu_kernel_duration_us" + suffix);
        if (kernel_hist) {
            LOG_PERF("  Kernel Durations: P50=%.2fμs, P95=%.2fμs, P99=%.2fμs\n",
                     kernel_hist->percentile(50),
                     kernel_hist->percentile(95),
                     kernel_hist->percentile(99));
        }
    }
}

size_t gpu_metrics_manager::get_total_memory_used() {
    return static_cast<size_t>(total_gpu_memory_used_->get());
}

size_t gpu_metrics_manager::get_total_memory_available() {
    return static_cast<size_t>(total_gpu_memory_available_->get());
}

} // namespace gpu
} // namespace metrics
} // namespace llama