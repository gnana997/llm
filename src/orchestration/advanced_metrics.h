#pragma once

#include "metrics.h"
#include <atomic>
#include <chrono>
#include <string>
#include <vector>
#include <mutex>

namespace llama {
namespace orchestration {

// Advanced metrics for intelligent layer distribution
struct AdvancedLayerMetrics {
    // Performance metrics
    Histogram* layer_compute_time_us[128];  // Per-layer execution times (up to 128 layers)
    Gauge* gpu_memory_bandwidth_utilization[8];  // Real-time bandwidth usage per GPU
    Counter* layer_migration_count;  // Number of runtime redistributions
    Counter* performance_anomaly_count;  // Detected performance issues
    Counter* thermal_throttle_events;  // GPU thermal events
    Counter* cross_gpu_transfer_bytes;  // Inter-GPU communication volume
    Gauge* layer_affinity_score[8];  // Architecture match quality per GPU
    
    // Additional performance tracking
    Histogram* layer_queue_wait_us;  // Time spent waiting in queue
    Histogram* gpu_kernel_launch_us;  // Kernel launch overhead
    Counter* memory_allocation_failures;  // OOM events
    Gauge* gpu_utilization_percent[8];  // GPU utilization percentage
    Gauge* gpu_temperature_celsius[8];  // GPU temperature
    Gauge* gpu_power_watts[8];  // Power consumption
    
    // Distribution quality metrics
    Gauge* load_imbalance_factor;  // Measure of load distribution quality
    Gauge* memory_imbalance_factor;  // Memory distribution quality
    Gauge* predicted_vs_actual_speedup;  // Model accuracy
    
    static AdvancedLayerMetrics& instance() {
        static AdvancedLayerMetrics metrics;
        return metrics;
    }
    
    void initialize() {
        auto& registry = MetricsRegistry::instance();
        
        // Initialize basic metrics
        layer_migration_count = registry.register_counter("layer_migration_count");
        performance_anomaly_count = registry.register_counter("performance_anomaly_count");
        thermal_throttle_events = registry.register_counter("thermal_throttle_events");
        cross_gpu_transfer_bytes = registry.register_counter("cross_gpu_transfer_bytes");
        
        // Initialize per-layer metrics (lazy initialization)
        for (int i = 0; i < 128; ++i) {
            layer_compute_time_us[i] = nullptr;  // Will be created on demand
        }
        
        // Initialize per-GPU metrics
        for (int i = 0; i < 8; ++i) {
            std::string prefix = "gpu_" + std::to_string(i);
            gpu_memory_bandwidth_utilization[i] = 
                registry.register_gauge(prefix + "_memory_bandwidth_utilization");
            layer_affinity_score[i] = 
                registry.register_gauge(prefix + "_layer_affinity_score");
            gpu_utilization_percent[i] = 
                registry.register_gauge(prefix + "_utilization_percent");
            gpu_temperature_celsius[i] = 
                registry.register_gauge(prefix + "_temperature_celsius");
            gpu_power_watts[i] = 
                registry.register_gauge(prefix + "_power_watts");
        }
        
        // Initialize histograms
        layer_queue_wait_us = registry.register_histogram("layer_queue_wait_us");
        gpu_kernel_launch_us = registry.register_histogram("gpu_kernel_launch_us");
        
        // Initialize quality metrics
        load_imbalance_factor = registry.register_gauge("load_imbalance_factor");
        memory_imbalance_factor = registry.register_gauge("memory_imbalance_factor");
        predicted_vs_actual_speedup = registry.register_gauge("predicted_vs_actual_speedup");
        
        // Initialize counter
        memory_allocation_failures = registry.register_counter("memory_allocation_failures");
    }
    
    // Get or create layer-specific histogram
    Histogram* get_layer_compute_histogram(int layer_id) {
        if (layer_id < 0 || layer_id >= 128) {
            return nullptr;
        }
        
        if (!layer_compute_time_us[layer_id]) {
            auto& registry = MetricsRegistry::instance();
            std::string name = "layer_" + std::to_string(layer_id) + "_compute_time_us";
            layer_compute_time_us[layer_id] = registry.register_histogram(name);
        }
        
        return layer_compute_time_us[layer_id];
    }
    
private:
    AdvancedLayerMetrics() {
        initialize();
    }
};

// Helper class for tracking layer execution
class LayerExecutionTracker {
public:
    LayerExecutionTracker(int layer_id, int gpu_id) 
        : layer_id_(layer_id), gpu_id_(gpu_id),
          start_time_(std::chrono::high_resolution_clock::now()) {}
    
    ~LayerExecutionTracker() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time_);
        
        // Record execution time
        auto& metrics = AdvancedLayerMetrics::instance();
        if (auto* hist = metrics.get_layer_compute_histogram(layer_id_)) {
            hist->observe(duration.count());
        }
        
        // Update GPU utilization estimate
        update_gpu_utilization(gpu_id_, duration);
    }
    
private:
    int layer_id_;
    int gpu_id_;
    std::chrono::high_resolution_clock::time_point start_time_;
    
    void update_gpu_utilization(int gpu_id, std::chrono::microseconds compute_time) {
        // Simple utilization tracking - would be enhanced with actual GPU queries
        static std::chrono::high_resolution_clock::time_point last_update[8];
        static std::chrono::microseconds accumulated_compute[8];
        static std::mutex utilization_mutex;
        
        std::lock_guard<std::mutex> lock(utilization_mutex);
        
        auto now = std::chrono::high_resolution_clock::now();
        accumulated_compute[gpu_id] += compute_time;
        
        // Update utilization every 100ms
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_update[gpu_id]);
        if (elapsed.count() >= 100) {
            auto& metrics = AdvancedLayerMetrics::instance();
            float utilization = (accumulated_compute[gpu_id].count() / 1000.0) / elapsed.count() * 100.0;
            utilization = std::min(100.0f, utilization);  // Cap at 100%
            
            metrics.gpu_utilization_percent[gpu_id]->set(utilization);
            
            // Reset counters
            accumulated_compute[gpu_id] = std::chrono::microseconds(0);
            last_update[gpu_id] = now;
        }
    }
};

// Helper class for tracking memory transfers
class MemoryTransferTracker {
public:
    static void track_transfer(int src_gpu, int dst_gpu, size_t bytes,
                              std::chrono::microseconds duration) {
        auto& metrics = AdvancedLayerMetrics::instance();
        
        // Track total cross-GPU transfer volume
        if (src_gpu != dst_gpu) {
            metrics.cross_gpu_transfer_bytes->increment(bytes);
        }
        
        // Calculate and update bandwidth utilization
        if (duration.count() > 0) {
            float bandwidth_gbps = (bytes / 1e9) / (duration.count() / 1e6);
            
            // Update source GPU bandwidth utilization
            // Assuming max bandwidth is known (would be from GPU profile)
            float max_bandwidth = 100.0f;  // GB/s - would come from actual GPU profile
            float utilization = (bandwidth_gbps / max_bandwidth) * 100.0f;
            
            metrics.gpu_memory_bandwidth_utilization[src_gpu]->set(utilization);
        }
    }
};

// Helper for tracking distribution quality
class DistributionQualityTracker {
public:
    static void update_quality_metrics(const std::vector<float>& gpu_loads,
                                     const std::vector<size_t>& gpu_memory,
                                     float predicted_speedup,
                                     float actual_speedup) {
        auto& metrics = AdvancedLayerMetrics::instance();
        
        // Calculate load imbalance
        float load_imbalance = calculate_imbalance(gpu_loads);
        metrics.load_imbalance_factor->set(load_imbalance);
        
        // Calculate memory imbalance
        std::vector<float> memory_floats(gpu_memory.begin(), gpu_memory.end());
        float memory_imbalance = calculate_imbalance(memory_floats);
        metrics.memory_imbalance_factor->set(memory_imbalance);
        
        // Track prediction accuracy
        float accuracy = actual_speedup / predicted_speedup;
        metrics.predicted_vs_actual_speedup->set(accuracy);
    }
    
private:
    static float calculate_imbalance(const std::vector<float>& values) {
        if (values.empty()) return 0.0f;
        
        float sum = 0.0f;
        float max_val = 0.0f;
        
        for (float val : values) {
            sum += val;
            max_val = std::max(max_val, val);
        }
        
        float avg = sum / values.size();
        if (avg == 0.0f) return 0.0f;
        
        // Imbalance factor: how much the max deviates from average
        return (max_val - avg) / avg;
    }
};

// Performance anomaly detection
class AnomalyDetector {
public:
    static void check_layer_execution(int layer_id, [[maybe_unused]] int gpu_id, 
                                    std::chrono::microseconds execution_time) {
        static std::unordered_map<int, std::chrono::microseconds> baseline_times;
        static std::mutex baseline_mutex;
        
        std::lock_guard<std::mutex> lock(baseline_mutex);
        
        // Initialize baseline on first execution
        if (baseline_times.find(layer_id) == baseline_times.end()) {
            baseline_times[layer_id] = execution_time;
            return;
        }
        
        // Check for anomaly (>2x baseline)
        if (execution_time > baseline_times[layer_id] * 2) {
            auto& metrics = AdvancedLayerMetrics::instance();
            metrics.performance_anomaly_count->increment();
            
            // Update baseline with exponential moving average
            baseline_times[layer_id] = std::chrono::microseconds(
                (baseline_times[layer_id].count() * 9 + execution_time.count()) / 10
            );
        }
    }
    
    static void check_thermal_throttle(int gpu_id, float temperature, float clock_mhz,
                                     float base_clock_mhz) {
        if (clock_mhz < base_clock_mhz * 0.9f) {  // 10% reduction indicates throttling
            auto& metrics = AdvancedLayerMetrics::instance();
            metrics.thermal_throttle_events->increment();
            metrics.gpu_temperature_celsius[gpu_id]->set(temperature);
        }
    }
};

// Convenience macros for metric tracking
#define TRACK_LAYER_EXECUTION(layer_id, gpu_id) \
    LayerExecutionTracker _tracker(layer_id, gpu_id)

#define TRACK_MEMORY_TRANSFER(src, dst, bytes, duration) \
    MemoryTransferTracker::track_transfer(src, dst, bytes, duration)

#define CHECK_PERFORMANCE_ANOMALY(layer_id, gpu_id, duration) \
    AnomalyDetector::check_layer_execution(layer_id, gpu_id, duration)

#define UPDATE_DISTRIBUTION_QUALITY(loads, memory, predicted, actual) \
    DistributionQualityTracker::update_quality_metrics(loads, memory, predicted, actual)

// Export functions for metrics
void export_advanced_metrics_json(std::ostream& out);
void export_advanced_metrics_prometheus(std::ostream& out);
void log_advanced_metrics_summary();

} // namespace orchestration
} // namespace llama