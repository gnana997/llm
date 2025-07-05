#include "metrics-aggregator.h"
#include "metrics-gpu.h"
#include "log-ex.h"
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <thread>
#include <set>

namespace llama {
namespace metrics {

// Rate calculator implementation
double rate_calculator::calculate_rate() {
    auto now = std::chrono::steady_clock::now();
    auto counter = get_counter(counter_name_);
    if (!counter) return 0;
    
    double current_value = counter->get();
    
    // First call - initialize
    if (last_time_ == std::chrono::steady_clock::time_point{}) {
        last_value_ = current_value;
        last_time_ = now;
        return 0;
    }
    
    // Add to history
    history_.emplace_back(now, current_value);
    
    // Remove old entries
    auto cutoff = now - window_;
    while (!history_.empty() && history_.front().first < cutoff) {
        history_.pop_front();
    }
    
    // Calculate rate
    if (history_.size() < 2) return 0;
    
    auto& oldest = history_.front();
    auto& newest = history_.back();
    
    auto time_diff = std::chrono::duration<double>(newest.first - oldest.first).count();
    if (time_diff <= 0) return 0;
    
    double value_diff = newest.second - oldest.second;
    return value_diff / time_diff; // rate per second
}

// Metrics aggregator implementation
metrics_aggregator::metrics_aggregator() {
    setup_common_aggregations();
}

void metrics_aggregator::register_rate(const std::string& counter_name,
                                      const std::string& rate_name,
                                      std::chrono::seconds window) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    rate_calculators_[rate_name] = std::make_unique<rate_calculator>(counter_name, window);
    
    // Create gauge for the rate
    auto& registry = metric_registry::instance();
    registry.register_metric<gauge>(rate_name, "Rate per second for " + counter_name);
    
    LOG_TRACE("Registered rate metric: %s (from %s)\n", rate_name.c_str(), counter_name.c_str());
}

void metrics_aggregator::register_derived(const std::string& name,
                                         const std::string& description,
                                         std::function<double()> calculator) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto& registry = metric_registry::instance();
    auto gauge_ptr = registry.register_metric<gauge>(name, description);
    
    derived_metrics_[name] = {description, calculator, gauge_ptr};
    
    LOG_TRACE("Registered derived metric: %s\n", name.c_str());
}

void metrics_aggregator::register_ratio(const std::string& name,
                                       const std::string& numerator_metric,
                                       const std::string& denominator_metric,
                                       const std::string& description) {
    register_derived(name, description, [numerator_metric, denominator_metric]() {
        auto num = metric_registry::instance().get(numerator_metric);
        auto den = metric_registry::instance().get(denominator_metric);
        
        if (!num || !den) return 0.0;
        
        auto num_values = num->get_values();
        auto den_values = den->get_values();
        
        if (num_values.empty() || den_values.empty()) return 0.0;
        
        double num_value = num_values.begin()->second;
        double den_value = den_values.begin()->second;
        
        return den_value > 0 ? num_value / den_value : 0.0;
    });
}

void metrics_aggregator::update() {
    update_rates();
    update_derived();
    
    // Take snapshot
    {
        std::lock_guard<std::mutex> lock(mutex_);
        
        metric_snapshot snapshot;
        snapshot.timestamp = std::chrono::steady_clock::now();
        snapshot.values = metric_registry::instance().get_all_values();
        
        history_.push_back(snapshot);
    }
    
    cleanup_history();
}

void metrics_aggregator::update_rates() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& [name, calculator] : rate_calculators_) {
        double rate = calculator->calculate_rate();
        auto gauge = get_gauge(name);
        if (gauge) {
            gauge->set(rate);
        }
    }
}

void metrics_aggregator::update_derived() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& [name, metric] : derived_metrics_) {
        try {
            double value = metric.calculator();
            if (metric.gauge_metric) {
                metric.gauge_metric->set(value);
            }
        } catch (const std::exception& e) {
            LOG_CAT(COMMON_LOG_CAT_GENERAL, GGML_LOG_LEVEL_WARN,
                    "Failed to calculate derived metric %s: %s\n", name.c_str(), e.what());
        }
    }
}

void metrics_aggregator::cleanup_history() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto cutoff = std::chrono::steady_clock::now() - history_retention_;
    while (!history_.empty() && history_.front().timestamp < cutoff) {
        history_.pop_front();
    }
}

metric_snapshot metrics_aggregator::get_snapshot() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    metric_snapshot snapshot;
    snapshot.timestamp = std::chrono::steady_clock::now();
    snapshot.values = metric_registry::instance().get_all_values();
    
    return snapshot;
}

std::vector<metric_snapshot> metrics_aggregator::get_history(std::chrono::seconds window) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<metric_snapshot> result;
    auto cutoff = std::chrono::steady_clock::now() - window;
    
    for (const auto& snapshot : history_) {
        if (snapshot.timestamp >= cutoff) {
            result.push_back(snapshot);
        }
    }
    
    return result;
}

metrics_aggregator::window_stats metrics_aggregator::calculate_stats(
    const std::string& metric_name, std::chrono::seconds window) {
    
    auto snapshots = get_history(window);
    window_stats stats{0, 0, 0, 0, 0};
    
    std::vector<double> values;
    for (const auto& snapshot : snapshots) {
        auto it = snapshot.values.find(metric_name);
        if (it != snapshot.values.end()) {
            values.push_back(it->second);
        }
    }
    
    if (values.empty()) return stats;
    
    stats.count = values.size();
    stats.min = *std::min_element(values.begin(), values.end());
    stats.max = *std::max_element(values.begin(), values.end());
    stats.avg = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    
    // Calculate standard deviation
    double variance = 0;
    for (double value : values) {
        variance += std::pow(value - stats.avg, 2);
    }
    stats.stddev = std::sqrt(variance / values.size());
    
    return stats;
}

std::string metrics_aggregator::export_prometheus() {
    auto snapshot = get_snapshot();
    std::ostringstream oss;
    
    // Group metrics by base name
    std::unordered_map<std::string, std::vector<std::pair<std::string, double>>> grouped;
    
    for (const auto& [name, value] : snapshot.values) {
        // Extract base name (before _bucket_, _count, etc.)
        std::string base_name = name;
        size_t pos = name.find("_bucket_");
        if (pos != std::string::npos) {
            base_name = name.substr(0, pos);
        } else if (name.size() > 6) {
            // Check for _count, _sum, _min, _max suffixes
            std::vector<std::string> suffixes = {"_count", "_sum", "_min", "_max", "_avg"};
            for (const auto& suffix : suffixes) {
                if (name.size() > suffix.size() && 
                    name.substr(name.size() - suffix.size()) == suffix) {
                    base_name = name.substr(0, name.size() - suffix.size());
                    break;
                }
            }
        }
        
        grouped[base_name].emplace_back(name, value);
    }
    
    // Export in Prometheus format
    for (const auto& [base_name, values] : grouped) {
        auto metric = metric_registry::instance().get(base_name);
        if (!metric) continue;
        
        // Write HELP and TYPE
        oss << "# HELP " << base_name << " " << metric->description() << "\n";
        
        std::string type_str;
        switch (metric->type()) {
            case metric_type::COUNTER: type_str = "counter"; break;
            case metric_type::GAUGE: type_str = "gauge"; break;
            case metric_type::HISTOGRAM: type_str = "histogram"; break;
            case metric_type::SUMMARY: type_str = "summary"; break;
        }
        oss << "# TYPE " << base_name << " " << type_str << "\n";
        
        // Write values
        for (const auto& [name, value] : values) {
            oss << name << " " << std::fixed << std::setprecision(6) << value << "\n";
        }
        
        oss << "\n";
    }
    
    return oss.str();
}

std::string metrics_aggregator::export_json() {
    auto snapshot = get_snapshot();
    std::ostringstream oss;
    
    oss << "{\n";
    oss << "  \"timestamp\": " << std::chrono::duration_cast<std::chrono::milliseconds>(
        snapshot.timestamp.time_since_epoch()).count() << ",\n";
    oss << "  \"metrics\": {\n";
    
    bool first = true;
    for (const auto& [name, value] : snapshot.values) {
        if (!first) oss << ",\n";
        first = false;
        
        oss << "    \"" << name << "\": " << std::fixed << std::setprecision(6) << value;
    }
    
    oss << "\n  }\n";
    oss << "}\n";
    
    return oss.str();
}

std::string metrics_aggregator::export_csv() {
    auto snapshots = get_history(std::chrono::seconds(3600)); // Last hour
    if (snapshots.empty()) return "";
    
    std::ostringstream oss;
    
    // Header
    oss << "timestamp";
    std::set<std::string> metric_names;
    for (const auto& snapshot : snapshots) {
        for (const auto& [name, _] : snapshot.values) {
            metric_names.insert(name);
        }
    }
    
    for (const auto& name : metric_names) {
        oss << "," << name;
    }
    oss << "\n";
    
    // Data rows
    for (const auto& snapshot : snapshots) {
        oss << std::chrono::duration_cast<std::chrono::milliseconds>(
            snapshot.timestamp.time_since_epoch()).count();
        
        for (const auto& name : metric_names) {
            oss << ",";
            auto it = snapshot.values.find(name);
            if (it != snapshot.values.end()) {
                oss << std::fixed << std::setprecision(6) << it->second;
            }
        }
        oss << "\n";
    }
    
    return oss.str();
}

void metrics_aggregator::register_alert(const std::string& name, const alert_condition& condition) {
    std::lock_guard<std::mutex> lock(mutex_);
    alerts_[name] = condition;
    LOG_TRACE("Registered alert: %s\n", name.c_str());
}

void metrics_aggregator::check_alerts() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto now = std::chrono::steady_clock::now();
    auto current_values = metric_registry::instance().get_all_values();
    
    for (auto& [name, alert] : alerts_) {
        // Check cooldown
        if (now - alert.last_triggered < alert.cooldown) {
            continue;
        }
        
        // Check condition
        auto it = current_values.find(alert.metric_name);
        if (it != current_values.end() && alert.condition(it->second)) {
            LOG_CAT(COMMON_LOG_CAT_GENERAL, GGML_LOG_LEVEL_WARN,
                    "ALERT [%s]: %s (metric %s = %.2f)\n",
                    name.c_str(), alert.message.c_str(),
                    alert.metric_name.c_str(), it->second);
            
            alert.last_triggered = now;
        }
    }
}

void metrics_aggregator::setup_common_aggregations() {
    // Token generation rate
    register_rate("tokens_generated_total", "tokens_per_second", std::chrono::seconds(60));
    
    // Request rate
    register_rate("requests_total", "requests_per_second", std::chrono::seconds(60));
    
    // Error rate
    register_rate("errors_total", "errors_per_second", std::chrono::seconds(300));
    
    // Memory allocation rate
    register_rate("memory_allocations_total", "memory_allocations_per_second", std::chrono::seconds(60));
    
    // Cache hit rate
    register_derived("cache_hit_rate_percent", "Cache hit rate percentage", derived::cache_hit_rate);
    
    // Memory utilization
    register_derived("memory_utilization_percent", "Memory utilization percentage", derived::memory_utilization);
    
    // GPU memory utilization
    register_derived("gpu_memory_utilization_percent", "GPU memory utilization percentage", 
                    derived::gpu_memory_utilization);
    
    // Average latencies
    register_derived("average_request_latency_ms", "Average request latency in milliseconds",
                    []() { return derived::average_request_latency() / 1000.0; });
    
    // Common alerts
    alert_condition high_memory_alert{
        "memory_utilization_percent",
        [](double value) { return value > 90.0; },
        "Memory utilization is above 90%",
        std::chrono::seconds(300)
    };
    register_alert("high_memory_usage", high_memory_alert);
    
    alert_condition high_error_rate{
        "errors_per_second",
        [](double value) { return value > 10.0; },
        "Error rate is above 10 errors/second",
        std::chrono::seconds(60)
    };
    register_alert("high_error_rate", high_error_rate);
    
    LOG_TRACE("Setup common metric aggregations\n");
}

// Aggregation updater implementation
void aggregation_updater::start() {
    if (running_.exchange(true)) return; // Already running
    
    update_thread_ = std::thread([this]() {
        while (running_) {
            metrics_aggregator::instance().update();
            metrics_aggregator::instance().check_alerts();
            
            // Update GPU metrics
            gpu::gpu_metrics_manager::instance().update_all();
            
            std::this_thread::sleep_for(interval_);
        }
    });
    
    LOG_TRACE("Started metrics aggregation updater\n");
}

void aggregation_updater::stop() {
    if (!running_.exchange(false)) return; // Not running
    
    if (update_thread_.joinable()) {
        update_thread_.join();
    }
    
    LOG_TRACE("Stopped metrics aggregation updater\n");
}

} // namespace metrics
} // namespace llama