#pragma once

#include "runtime_performance_monitor.h"
#include "layer_distributor.h"
#include "gpu_profiler.h"
#include <atomic>
#include <condition_variable>
#include <thread>
#include <queue>

// Forward declarations
struct llama_model;
struct llama_context;

namespace llama {
namespace orchestration {

// Redistribution request
struct RedistributionRequest {
    enum Type {
        PERFORMANCE_TRIGGERED,  // Based on performance metrics
        THERMAL_TRIGGERED,     // Due to thermal issues
        MEMORY_TRIGGERED,      // Due to memory pressure
        MANUAL_TRIGGERED       // User-requested
    } type;
    
    std::string reason;
    float urgency;  // 0.0-1.0, higher means more urgent
    std::vector<int> affected_gpus;
    std::chrono::steady_clock::time_point timestamp;
};

// Redistribution plan
struct RedistributionPlan {
    struct LayerMove {
        int layer_id;
        int from_gpu;
        int to_gpu;
        size_t memory_size;
        float expected_improvement;
    };
    
    std::vector<LayerMove> moves;
    float expected_overall_improvement;
    float risk_score;  // 0.0-1.0, higher means riskier
    std::string strategy_description;
    
    bool is_valid() const { return !moves.empty() && risk_score < 0.8f; }
};

// Redistribution result
struct RedistributionResult {
    bool success;
    int layers_moved;
    std::chrono::microseconds duration;
    float actual_improvement;
    std::string error_message;
    std::vector<int> affected_layers;
};

// Adaptive layer redistributor
class AdaptiveLayerRedistributor {
public:
    AdaptiveLayerRedistributor();
    ~AdaptiveLayerRedistributor();
    
    // Initialize with components
    void initialize(RuntimePerformanceMonitor* monitor,
                   LayerDistributor* distributor,
                   GpuProfiler* profiler);
    
    // Enable/disable adaptive redistribution
    void enable_adaptive_mode();
    void disable_adaptive_mode();
    bool is_adaptive_mode_enabled() const { return adaptive_mode_enabled_.load(); }
    
    // Manual redistribution trigger
    RedistributionResult trigger_redistribution(const RedistributionRequest& request);
    
    // Get current redistribution status
    bool is_redistributing() const { return redistributing_.load(); }
    float get_stability_score() const;  // 0.0-1.0, higher is more stable
    
    // Configuration
    struct AdaptiveConfig {
        // Triggering thresholds
        float performance_degradation_threshold = 0.2f;  // 20% degradation
        float thermal_severity_threshold = 0.5f;         // Severity 0.5+
        float memory_pressure_threshold = 0.85f;         // 85% memory usage
        float load_imbalance_threshold = 0.3f;          // 30% imbalance
        
        // Stability requirements
        int min_samples_before_redistribution = 100;    // Samples needed
        std::chrono::seconds min_time_between_redistributions{30};
        float min_expected_improvement = 0.1f;           // 10% improvement
        
        // Safety limits
        int max_redistributions_per_hour = 10;
        int max_layers_per_redistribution = 20;
        float max_risk_score = 0.7f;
        
        // Strategy preferences
        bool prefer_minimal_moves = true;
        bool allow_cross_node_moves = false;
        bool enable_predictive_redistribution = true;
    };
    
    void set_config(const AdaptiveConfig& config) { config_ = config; }
    const AdaptiveConfig& get_config() const { return config_; }
    
    // Statistics
    struct RedistributionStats {
        int total_redistributions = 0;
        int successful_redistributions = 0;
        int failed_redistributions = 0;
        float average_improvement = 0.0f;
        std::chrono::microseconds total_redistribution_time{0};
        std::chrono::steady_clock::time_point last_redistribution;
    };
    
    RedistributionStats get_stats() const;
    void reset_stats();
    
    // Callbacks
    using RedistributionCallback = std::function<void(const RedistributionResult&)>;
    void set_redistribution_callback(RedistributionCallback callback);
    
private:
    // Components
    RuntimePerformanceMonitor* performance_monitor_ = nullptr;
    LayerDistributor* layer_distributor_ = nullptr;
    GpuProfiler* gpu_profiler_ = nullptr;
    
    // State
    std::atomic<bool> adaptive_mode_enabled_{false};
    std::atomic<bool> redistributing_{false};
    std::atomic<bool> monitoring_active_{false};
    
    // Configuration
    AdaptiveConfig config_;
    
    // Monitoring thread
    std::thread monitor_thread_;
    std::mutex monitor_mutex_;
    std::condition_variable monitor_cv_;
    
    // Redistribution queue
    std::mutex request_mutex_;
    std::queue<RedistributionRequest> pending_requests_;
    
    // History tracking
    mutable std::mutex history_mutex_;
    std::deque<RedistributionResult> redistribution_history_;
    std::chrono::steady_clock::time_point last_redistribution_time_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    RedistributionStats stats_;
    
    // Callbacks
    RedistributionCallback redistribution_callback_;
    
    // Current model state
    std::mutex model_state_mutex_;
    DistributionResult current_distribution_;
    std::vector<GpuProfile> current_gpu_profiles_;
    
    // Internal methods
    void monitor_loop();
    void check_performance_triggers();
    void check_thermal_triggers();
    void check_memory_triggers();
    void check_predictive_triggers();
    
    // Redistribution planning
    RedistributionPlan create_redistribution_plan(const RedistributionRequest& request);
    RedistributionPlan plan_performance_redistribution(const std::vector<int>& affected_gpus);
    RedistributionPlan plan_thermal_redistribution(int throttled_gpu);
    RedistributionPlan plan_memory_redistribution(int pressured_gpu);
    
    // Plan optimization
    void optimize_plan(RedistributionPlan& plan);
    float evaluate_plan_risk(const RedistributionPlan& plan);
    float predict_plan_improvement(const RedistributionPlan& plan);
    
    // Redistribution execution
    RedistributionResult execute_redistribution(const RedistributionPlan& plan);
    bool prepare_redistribution(const RedistributionPlan& plan);
    bool migrate_layer(const RedistributionPlan::LayerMove& move);
    void rollback_redistribution(const RedistributionPlan& plan, int completed_moves);
    
    // Stability and safety checks
    bool is_safe_to_redistribute() const;
    bool check_redistribution_limits() const;
    void update_stability_metrics(const RedistributionResult& result);
    
    // Helper methods
    std::vector<int> identify_overloaded_gpus() const;
    std::vector<int> identify_underutilized_gpus() const;
    std::vector<int> find_movable_layers(int from_gpu) const;
    int find_best_target_gpu(int layer_id, const std::vector<int>& candidates) const;
    float calculate_actual_improvement() const;
    
    // Logging helpers
    void log_redistribution_plan(const RedistributionPlan& plan) const;
    void log_redistribution_result(const RedistributionResult& result) const;
};

// Global adaptive redistributor instance (optional)
class AdaptiveRedistributor {
public:
    static AdaptiveLayerRedistributor& instance() {
        static AdaptiveLayerRedistributor redistributor;
        return redistributor;
    }
    
    static void enable(RuntimePerformanceMonitor* monitor,
                      LayerDistributor* distributor,
                      GpuProfiler* profiler) {
        auto& redistributor = instance();
        redistributor.initialize(monitor, distributor, profiler);
        redistributor.enable_adaptive_mode();
    }
    
    static void disable() {
        instance().disable_adaptive_mode();
    }
};

// Integration helpers
namespace redistribution_utils {
    // Check if redistribution would be beneficial
    bool should_redistribute(const RuntimePerformanceMonitor& monitor,
                           const DistributionResult& current_distribution);
    
    // Create a redistribution request from anomalies
    RedistributionRequest create_request_from_anomalies(
        const std::vector<PerformanceAnomaly>& anomalies);
    
    // Validate a redistribution plan
    bool validate_plan(const RedistributionPlan& plan,
                      const std::vector<GpuProfile>& gpu_profiles);
    
    // Calculate redistribution impact
    struct RedistributionImpact {
        float performance_change;  // Positive = improvement
        float stability_change;    // Positive = more stable
        float risk_level;         // 0.0-1.0
        std::vector<std::string> warnings;
    };
    
    RedistributionImpact analyze_impact(const RedistributionPlan& plan,
                                       const DistributionResult& current,
                                       const RuntimePerformanceMonitor& monitor);
}

} // namespace orchestration
} // namespace llama