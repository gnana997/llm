#pragma once

#include "metrics.h"
#include <chrono>
#include <deque>
#include <functional>
#include <thread>

namespace llama {
namespace metrics {

// Time window for aggregation
struct aggregation_window {
    std::chrono::seconds duration;
    std::string name;
};

// Aggregated metric snapshot
struct metric_snapshot {
    std::chrono::steady_clock::time_point timestamp;
    std::unordered_map<std::string, double> values;
};

// Rate calculation for counters
class rate_calculator {
public:
    rate_calculator(const std::string& counter_name, std::chrono::seconds window)
        : counter_name_(counter_name), window_(window), last_value_(0), last_time_() {}
    
    // Calculate rate per second
    double calculate_rate();
    
private:
    std::string counter_name_;
    std::chrono::seconds window_;
    double last_value_;
    std::chrono::steady_clock::time_point last_time_;
    std::deque<std::pair<std::chrono::steady_clock::time_point, double>> history_;
};

// Metrics aggregator - calculates derived metrics and time-based aggregations
class metrics_aggregator {
public:
    static metrics_aggregator& instance() {
        static metrics_aggregator instance;
        return instance;
    }
    
    // Register a counter for rate calculation
    void register_rate(const std::string& counter_name, 
                      const std::string& rate_name,
                      std::chrono::seconds window = std::chrono::seconds(60));
    
    // Register a derived metric (calculated from other metrics)
    void register_derived(const std::string& name,
                         const std::string& description,
                         std::function<double()> calculator);
    
    // Register a ratio metric (numerator/denominator)
    void register_ratio(const std::string& name,
                       const std::string& numerator_metric,
                       const std::string& denominator_metric,
                       const std::string& description);
    
    // Update all aggregations
    void update();
    
    // Get snapshot of current metrics
    metric_snapshot get_snapshot();
    
    // Get historical snapshots for a time window
    std::vector<metric_snapshot> get_history(std::chrono::seconds window);
    
    // Calculate statistics over a time window
    struct window_stats {
        double min;
        double max;
        double avg;
        double stddev;
        size_t count;
    };
    
    window_stats calculate_stats(const std::string& metric_name, 
                                 std::chrono::seconds window);
    
    // Export metrics in various formats
    std::string export_prometheus();
    std::string export_json();
    std::string export_csv();
    
    // Alerting support
    struct alert_condition {
        std::string metric_name;
        std::function<bool(double)> condition;
        std::string message;
        std::chrono::seconds cooldown;
        std::chrono::steady_clock::time_point last_triggered;
    };
    
    void register_alert(const std::string& name, const alert_condition& condition);
    void check_alerts();
    
    // Automatic aggregation of common patterns
    void setup_common_aggregations();
    
private:
    metrics_aggregator();
    metrics_aggregator(const metrics_aggregator&) = delete;
    metrics_aggregator& operator=(const metrics_aggregator&) = delete;
    
    std::mutex mutex_;
    
    // Rate calculators
    std::unordered_map<std::string, std::unique_ptr<rate_calculator>> rate_calculators_;
    
    // Derived metrics
    struct derived_metric {
        std::string description;
        std::function<double()> calculator;
        std::shared_ptr<gauge> gauge_metric;
    };
    std::unordered_map<std::string, derived_metric> derived_metrics_;
    
    // Historical snapshots
    std::deque<metric_snapshot> history_;
    std::chrono::seconds history_retention_ = std::chrono::seconds(3600); // 1 hour
    
    // Alerts
    std::unordered_map<std::string, alert_condition> alerts_;
    
    // Update derived metrics
    void update_rates();
    void update_derived();
    void cleanup_history();
};

// Common derived metrics
namespace derived {

// Calculate tokens per second
inline double tokens_per_second() {
    auto& aggregator = metrics_aggregator::instance();
    auto stats = aggregator.calculate_stats("tokens_generated_total_rate", std::chrono::seconds(60));
    return stats.avg;
}

// Calculate cache hit rate
inline double cache_hit_rate() {
    auto hits = get_counter("cache_hits_total");
    auto misses = get_counter("cache_misses_total");
    if (!hits || !misses) return 0;
    
    double total = hits->get() + misses->get();
    return total > 0 ? (hits->get() / total) * 100.0 : 0;
}

// Calculate memory utilization percentage
inline double memory_utilization() {
    auto used = get_gauge("memory_used_bytes");
    auto allocated = get_gauge("memory_allocated_bytes");
    if (!used || !allocated || allocated->get() == 0) return 0;
    
    return (used->get() / allocated->get()) * 100.0;
}

// Calculate average request latency
inline double average_request_latency() {
    auto histogram = get_histogram("request_duration_us");
    if (!histogram) return 0;
    
    auto values = histogram->get_values();
    double count = values["request_duration_us_count"];
    double sum = values["request_duration_us_sum"];
    
    return count > 0 ? sum / count : 0;
}

// Calculate GPU memory utilization
inline double gpu_memory_utilization() {
    auto used = get_gauge("gpu_memory_used_bytes_total");
    auto available = get_gauge("gpu_memory_available_bytes_total");
    if (!used || !available || available->get() == 0) return 0;
    
    return (used->get() / available->get()) * 100.0;
}

} // namespace derived

// Helper macros for common aggregations
#define METRIC_REGISTER_RATE(counter_name, rate_name) \
    llama::metrics::metrics_aggregator::instance().register_rate(counter_name, rate_name)

#define METRIC_REGISTER_RATIO(name, numerator, denominator, description) \
    llama::metrics::metrics_aggregator::instance().register_ratio(name, numerator, denominator, description)

#define METRIC_UPDATE_AGGREGATIONS() \
    llama::metrics::metrics_aggregator::instance().update()

#define METRIC_CHECK_ALERTS() \
    llama::metrics::metrics_aggregator::instance().check_alerts()

// Periodic aggregation updater
class aggregation_updater {
public:
    aggregation_updater(std::chrono::milliseconds interval = std::chrono::milliseconds(1000))
        : interval_(interval), running_(false) {}
    
    ~aggregation_updater() { stop(); }
    
    void start();
    void stop();
    
private:
    std::chrono::milliseconds interval_;
    std::atomic<bool> running_;
    std::thread update_thread_;
};

} // namespace metrics
} // namespace llama