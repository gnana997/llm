#pragma once

#include "gpu_profiler_interface.h"
#include "layer_distributor.h"
#include "advanced_metrics.h"
#include "orchestration_logger.h"
#include <chrono>
#include <vector>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <mutex>
#include <queue>
#include <thread>

// Forward declarations
struct llama_model;
struct llama_context;
struct ggml_tensor;

namespace llama {
namespace orchestration {

// Performance sample for a single layer execution
struct LayerPerformanceSample {
    int layer_id;
    int gpu_id;
    std::chrono::microseconds execution_time;
    std::chrono::microseconds queue_wait_time;
    size_t memory_transferred_bytes;
    float gpu_utilization;
    float memory_bandwidth_utilization;
    std::chrono::steady_clock::time_point timestamp;
};

// Aggregated performance statistics
struct LayerPerformanceStats {
    int sample_count = 0;
    std::chrono::microseconds total_execution_time{0};
    std::chrono::microseconds min_execution_time{std::chrono::microseconds::max()};
    std::chrono::microseconds max_execution_time{0};
    std::chrono::microseconds avg_execution_time{0};
    float avg_gpu_utilization = 0.0f;
    float avg_memory_bandwidth_utilization = 0.0f;
    
    // Moving averages for trend detection
    std::chrono::microseconds moving_avg_5{0};   // Last 5 samples
    std::chrono::microseconds moving_avg_20{0};  // Last 20 samples
    
    void update(const LayerPerformanceSample& sample);
    bool is_degrading() const;  // Detect performance degradation
};

// GPU runtime status
struct GpuRuntimeStatus {
    int gpu_id;
    float current_utilization = 0.0f;
    float current_temperature = 0.0f;
    float current_power_watts = 0.0f;
    float current_clock_mhz = 0.0f;
    float base_clock_mhz = 0.0f;
    bool is_throttling = false;
    size_t available_memory_bytes = 0;
    size_t used_memory_bytes = 0;
    std::chrono::steady_clock::time_point last_update;
};

// Performance anomaly types
enum class PerformanceAnomalyType {
    SLOW_EXECUTION,      // Layer taking significantly longer than baseline
    THERMAL_THROTTLE,    // GPU thermal throttling detected
    MEMORY_PRESSURE,     // Low available memory
    LOAD_IMBALANCE,      // Significant load imbalance between GPUs
    HIGH_QUEUE_WAIT,     // Excessive queue wait times
    BANDWIDTH_SATURATION // Memory bandwidth saturation
};

// Performance anomaly event
struct PerformanceAnomaly {
    PerformanceAnomalyType type;
    int affected_gpu_id;
    int affected_layer_id;
    float severity;  // 0.0-1.0, higher is more severe
    std::string description;
    std::chrono::steady_clock::time_point timestamp;
    std::unordered_map<std::string, float> metrics;  // Additional metrics
};

// Callback for anomaly notifications
using AnomalyCallback = std::function<void(const PerformanceAnomaly&)>;

// Runtime performance monitor
class RuntimePerformanceMonitor {
public:
    RuntimePerformanceMonitor();
    ~RuntimePerformanceMonitor();
    
    // Initialize with GPU profiles and layer distribution
    void initialize(const std::vector<GpuProfile>& gpu_profiles,
                   const DistributionResult& distribution);
    
    // Start/stop monitoring
    void start_monitoring();
    void stop_monitoring();
    bool is_monitoring() const { return monitoring_active_.load(); }
    
    // Layer execution tracking
    void record_layer_start(int layer_id, int gpu_id);
    void record_layer_end(int layer_id, int gpu_id);
    
    // Memory transfer tracking
    void record_memory_transfer(int src_gpu, int dst_gpu, size_t bytes,
                               std::chrono::microseconds duration);
    
    // Queue wait tracking
    void record_queue_wait(int layer_id, std::chrono::microseconds wait_time);
    
    // Get current performance statistics
    LayerPerformanceStats get_layer_stats(int layer_id) const;
    GpuRuntimeStatus get_gpu_status(int gpu_id) const;
    std::vector<PerformanceAnomaly> get_recent_anomalies(size_t count = 10) const;
    
    // Performance analysis
    float calculate_current_speedup() const;
    std::vector<float> get_gpu_load_distribution() const;
    bool is_distribution_optimal() const;
    
    // Anomaly detection
    void set_anomaly_callback(AnomalyCallback callback);
    void check_for_anomalies();
    
    // Performance recommendations
    struct PerformanceRecommendation {
        enum Type {
            REDISTRIBUTE_LAYERS,
            REDUCE_BATCH_SIZE,
            ENABLE_MEMORY_OPTIMIZATION,
            WAIT_FOR_THERMAL_RECOVERY
        } type;
        std::string description;
        float expected_improvement;
        std::unordered_map<std::string, int> parameters;
    };
    
    std::vector<PerformanceRecommendation> get_recommendations() const;
    
    // Export performance data
    void export_performance_trace(const std::string& filename) const;
    void log_performance_summary() const;
    
    // Configuration
    struct MonitorConfig {
        std::chrono::milliseconds gpu_status_update_interval{100};
        std::chrono::milliseconds anomaly_check_interval{500};
        float slow_execution_threshold = 2.0f;  // 2x baseline
        float thermal_throttle_threshold = 85.0f;  // Celsius
        float memory_pressure_threshold = 0.9f;  // 90% used
        float load_imbalance_threshold = 0.3f;  // 30% deviation
        bool enable_trace_export = false;
        bool enable_auto_recommendations = true;
    };
    
    void set_config(const MonitorConfig& config) { config_ = config; }
    const MonitorConfig& get_config() const { return config_; }
    
private:
    // Internal state
    std::atomic<bool> monitoring_active_{false};
    std::vector<GpuProfile> gpu_profiles_;
    DistributionResult current_distribution_;
    MonitorConfig config_;
    
    // Performance tracking
    mutable std::mutex stats_mutex_;
    std::unordered_map<int, LayerPerformanceStats> layer_stats_;
    std::unordered_map<int, std::deque<LayerPerformanceSample>> layer_samples_;
    
    // GPU status tracking
    mutable std::mutex gpu_status_mutex_;
    std::vector<GpuRuntimeStatus> gpu_status_;
    std::thread gpu_monitor_thread_;
    
    // Active layer tracking
    struct ActiveLayerExecution {
        int layer_id;
        int gpu_id;
        std::chrono::steady_clock::time_point start_time;
    };
    std::mutex active_layers_mutex_;
    std::unordered_map<int, ActiveLayerExecution> active_layers_;
    
    // Anomaly tracking
    mutable std::mutex anomaly_mutex_;
    std::deque<PerformanceAnomaly> anomaly_history_;
    AnomalyCallback anomaly_callback_;
    std::thread anomaly_monitor_thread_;
    
    // Trace data for export
    struct TraceEvent {
        std::string name;
        std::string category;
        std::chrono::microseconds timestamp;
        std::chrono::microseconds duration;
        std::unordered_map<std::string, std::string> args;
    };
    mutable std::mutex trace_mutex_;
    std::vector<TraceEvent> trace_events_;
    
    // Internal methods
    void gpu_monitor_loop();
    void anomaly_monitor_loop();
    void update_gpu_status(int gpu_id);
    void detect_thermal_throttle(int gpu_id);
    void detect_memory_pressure(int gpu_id);
    void detect_load_imbalance();
    void detect_slow_execution(int layer_id, const LayerPerformanceSample& sample);
    void add_anomaly(const PerformanceAnomaly& anomaly);
    void update_metrics(const LayerPerformanceSample& sample);
    
    // Performance analysis helpers
    float calculate_gpu_efficiency(int gpu_id) const;
    float calculate_overall_efficiency() const;
    std::vector<int> identify_bottleneck_layers() const;
    
    // Recommendation engine
    PerformanceRecommendation recommend_redistribution() const;
    PerformanceRecommendation recommend_batch_adjustment() const;
    PerformanceRecommendation recommend_memory_optimization() const;
};

// RAII helper for automatic layer timing
class LayerExecutionScope {
public:
    LayerExecutionScope(RuntimePerformanceMonitor& monitor, int layer_id, int gpu_id)
        : monitor_(monitor), layer_id_(layer_id), gpu_id_(gpu_id) {
        monitor_.record_layer_start(layer_id_, gpu_id_);
    }
    
    ~LayerExecutionScope() {
        monitor_.record_layer_end(layer_id_, gpu_id_);
    }
    
private:
    RuntimePerformanceMonitor& monitor_;
    int layer_id_;
    int gpu_id_;
};

// Global runtime monitor instance (optional singleton pattern)
class RuntimeMonitor {
public:
    static RuntimePerformanceMonitor& instance() {
        static RuntimePerformanceMonitor monitor;
        return monitor;
    }
    
    static void enable(const std::vector<GpuProfile>& gpu_profiles,
                      const DistributionResult& distribution) {
        auto& monitor = instance();
        monitor.initialize(gpu_profiles, distribution);
        monitor.start_monitoring();
    }
    
    static void disable() {
        instance().stop_monitoring();
    }
};

// Integration macros
#define MONITOR_LAYER_EXECUTION(monitor, layer_id, gpu_id) \
    LayerExecutionScope _layer_scope(monitor, layer_id, gpu_id)

#define MONITOR_MEMORY_TRANSFER(monitor, src, dst, bytes, duration) \
    monitor.record_memory_transfer(src, dst, bytes, duration)

#define MONITOR_QUEUE_WAIT(monitor, layer_id, wait_time) \
    monitor.record_queue_wait(layer_id, wait_time)

} // namespace orchestration
} // namespace llama