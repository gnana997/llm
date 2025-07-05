#pragma once

#include <string>
#include <unordered_map>
#include <map>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace llama {
namespace metrics {

// Forward declarations
class metric_registry;

// Metric types
enum class metric_type {
    COUNTER,    // Monotonically increasing value
    GAUGE,      // Value that can go up or down
    HISTOGRAM,  // Distribution of values
    SUMMARY     // Statistical summary with time windows
};

// Base metric interface
class metric {
public:
    metric(const std::string& name, const std::string& description, metric_type type)
        : name_(name), description_(description), type_(type) {}
    
    virtual ~metric() = default;
    
    const std::string& name() const { return name_; }
    const std::string& description() const { return description_; }
    metric_type type() const { return type_; }
    
    // Get current value(s) as key-value pairs
    virtual std::unordered_map<std::string, double> get_values() const = 0;
    
    // Reset the metric (mainly for histograms/summaries)
    virtual void reset() {}
    
protected:
    std::string name_;
    std::string description_;
    metric_type type_;
};

// Counter: monotonically increasing metric
class counter : public metric {
public:
    counter(const std::string& name, const std::string& description)
        : metric(name, description, metric_type::COUNTER), value_(0) {}
    
    void increment(double value = 1.0) {
        if (value < 0) return; // Counters can only increase
        double old_value = value_.load();
        while (!value_.compare_exchange_weak(old_value, old_value + value)) {}
    }
    
    double get() const { return value_.load(); }
    
    std::unordered_map<std::string, double> get_values() const override {
        return {{name_, get()}};
    }
    
    void reset() override { value_.store(0); }
    
private:
    std::atomic<double> value_;
};

// Gauge: metric that can go up or down
class gauge : public metric {
public:
    gauge(const std::string& name, const std::string& description)
        : metric(name, description, metric_type::GAUGE), value_(0) {}
    
    void set(double value) { value_.store(value); }
    void increment(double value = 1.0) { 
        double old_value = value_.load();
        while (!value_.compare_exchange_weak(old_value, old_value + value)) {}
    }
    void decrement(double value = 1.0) { increment(-value); }
    
    double get() const { return value_.load(); }
    
    std::unordered_map<std::string, double> get_values() const override {
        return {{name_, get()}};
    }
    
private:
    std::atomic<double> value_;
};

// Histogram: distribution of values
class histogram : public metric {
public:
    // Default buckets for latency in microseconds
    static std::vector<double> default_latency_buckets() {
        return {10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000};
    }
    
    // Default buckets for memory in bytes
    static std::vector<double> default_memory_buckets() {
        return {1024, 10240, 102400, 1048576, 10485760, 104857600, 1073741824, 10737418240};
    }
    
    histogram(const std::string& name, const std::string& description,
              const std::vector<double>& buckets = default_latency_buckets())
        : metric(name, description, metric_type::HISTOGRAM),
          buckets_(buckets), bucket_counts_(buckets.size() + 1, 0),
          sum_(0), count_(0) {
        std::sort(buckets_.begin(), buckets_.end());
    }
    
    void observe(double value) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Find the bucket
        auto it = std::lower_bound(buckets_.begin(), buckets_.end(), value);
        size_t idx = std::distance(buckets_.begin(), it);
        bucket_counts_[idx]++;
        
        // Update statistics
        sum_ += value;
        count_++;
        
        // Update min/max
        if (count_ == 1) {
            min_ = max_ = value;
        } else {
            min_ = std::min(min_, value);
            max_ = std::max(max_, value);
        }
    }
    
    std::unordered_map<std::string, double> get_values() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        std::unordered_map<std::string, double> values;
        
        // Bucket counts
        for (size_t i = 0; i < buckets_.size(); ++i) {
            values[name_ + "_bucket_" + std::to_string(buckets_[i])] = 
                std::accumulate(bucket_counts_.begin(), bucket_counts_.begin() + i + 1, 0.0);
        }
        values[name_ + "_bucket_inf"] = count_;
        
        // Statistics
        values[name_ + "_count"] = count_;
        values[name_ + "_sum"] = sum_;
        if (count_ > 0) {
            values[name_ + "_min"] = min_;
            values[name_ + "_max"] = max_;
            values[name_ + "_avg"] = sum_ / count_;
        }
        
        return values;
    }
    
    void reset() override {
        std::lock_guard<std::mutex> lock(mutex_);
        std::fill(bucket_counts_.begin(), bucket_counts_.end(), 0);
        sum_ = count_ = 0;
        min_ = max_ = 0;
    }
    
    // Get percentile (requires sorting all observations - use sparingly)
    double percentile(double p) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (count_ == 0) return 0;
        
        // This is an approximation based on buckets
        size_t target = static_cast<size_t>(count_ * p / 100.0);
        size_t cumulative = 0;
        
        for (size_t i = 0; i < bucket_counts_.size(); ++i) {
            cumulative += bucket_counts_[i];
            if (cumulative >= target) {
                if (i == 0) return buckets_[0] / 2;  // Estimate for first bucket
                if (i > buckets_.size()) return buckets_.back() * 1.5;  // Estimate for +Inf bucket
                return (buckets_[i-1] + buckets_[i]) / 2;  // Midpoint of bucket
            }
        }
        
        return max_;
    }
    
private:
    mutable std::mutex mutex_;
    std::vector<double> buckets_;
    std::vector<size_t> bucket_counts_;
    double sum_;
    size_t count_;
    double min_, max_;
};

// Summary: like histogram but with time-windowed percentile calculations
class summary : public metric {
public:
    struct time_window {
        std::chrono::seconds duration;
        std::vector<std::pair<std::chrono::steady_clock::time_point, double>> observations;
    };
    
    summary(const std::string& name, const std::string& description,
            const std::vector<std::chrono::seconds>& windows = {std::chrono::seconds(60), std::chrono::seconds(300)})
        : metric(name, description, metric_type::SUMMARY) {
        for (const auto& window : windows) {
            windows_[window.count()] = time_window{window, {}};
        }
    }
    
    void observe(double value) {
        auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (auto& [duration_count, window] : windows_) {
            // Remove old observations
            auto cutoff = now - window.duration;
            window.observations.erase(
                std::remove_if(window.observations.begin(), window.observations.end(),
                    [cutoff](const auto& obs) { return obs.first < cutoff; }),
                window.observations.end()
            );
            
            // Add new observation
            window.observations.emplace_back(now, value);
        }
    }
    
    std::unordered_map<std::string, double> get_values() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        std::unordered_map<std::string, double> values;
        
        for (const auto& [duration_count, window] : windows_) {
            if (window.observations.empty()) continue;
            
            std::vector<double> sorted_values;
            sorted_values.reserve(window.observations.size());
            for (const auto& obs : window.observations) {
                sorted_values.push_back(obs.second);
            }
            std::sort(sorted_values.begin(), sorted_values.end());
            
            std::string prefix = name_ + "_" + std::to_string(duration_count) + "s";
            values[prefix + "_count"] = sorted_values.size();
            values[prefix + "_min"] = sorted_values.front();
            values[prefix + "_max"] = sorted_values.back();
            values[prefix + "_p50"] = sorted_values[sorted_values.size() / 2];
            values[prefix + "_p95"] = sorted_values[static_cast<size_t>(sorted_values.size() * 0.95)];
            values[prefix + "_p99"] = sorted_values[static_cast<size_t>(sorted_values.size() * 0.99)];
            
            double sum = std::accumulate(sorted_values.begin(), sorted_values.end(), 0.0);
            values[prefix + "_avg"] = sum / sorted_values.size();
        }
        
        return values;
    }
    
    void reset() override {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [_, window] : windows_) {
            window.observations.clear();
        }
    }
    
private:
    mutable std::mutex mutex_;
    std::map<std::chrono::seconds::rep, time_window> windows_;
};

// Metric registry - singleton pattern
class metric_registry {
public:
    static metric_registry& instance() {
        static metric_registry instance;
        return instance;
    }
    
    // Register a metric
    template<typename T, typename... Args>
    std::shared_ptr<T> register_metric(const std::string& name, Args&&... args) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = metrics_.find(name);
        if (it != metrics_.end()) {
            // Return existing metric if it matches the type
            auto existing = std::dynamic_pointer_cast<T>(it->second);
            if (existing) return existing;
            
            // Type mismatch - return nullptr
            return nullptr;
        }
        
        auto metric = std::make_shared<T>(name, std::forward<Args>(args)...);
        metrics_[name] = metric;
        return metric;
    }
    
    // Get a metric by name
    std::shared_ptr<metric> get(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = metrics_.find(name);
        return (it != metrics_.end()) ? it->second : nullptr;
    }
    
    // Get all metrics
    std::vector<std::shared_ptr<metric>> get_all() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::shared_ptr<metric>> result;
        result.reserve(metrics_.size());
        for (const auto& [_, m] : metrics_) {
            result.push_back(m);
        }
        return result;
    }
    
    // Get all metric values
    std::unordered_map<std::string, double> get_all_values() const {
        std::unordered_map<std::string, double> all_values;
        for (const auto& m : get_all()) {
            auto values = m->get_values();
            all_values.insert(values.begin(), values.end());
        }
        return all_values;
    }
    
    // Clear all metrics
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_.clear();
    }
    
private:
    metric_registry() = default;
    metric_registry(const metric_registry&) = delete;
    metric_registry& operator=(const metric_registry&) = delete;
    
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<metric>> metrics_;
};

// Helper functions
inline std::shared_ptr<counter> get_counter(const std::string& name) {
    return std::dynamic_pointer_cast<counter>(metric_registry::instance().get(name));
}

inline std::shared_ptr<gauge> get_gauge(const std::string& name) {
    return std::dynamic_pointer_cast<gauge>(metric_registry::instance().get(name));
}

inline std::shared_ptr<histogram> get_histogram(const std::string& name) {
    return std::dynamic_pointer_cast<histogram>(metric_registry::instance().get(name));
}

inline std::shared_ptr<summary> get_summary(const std::string& name) {
    return std::dynamic_pointer_cast<summary>(metric_registry::instance().get(name));
}

// Convenience macros
#define METRIC_INCREMENT(name, value) \
    do { \
        auto c = llama::metrics::get_counter(name); \
        if (c) c->increment(value); \
    } while(0)

#define METRIC_SET(name, value) \
    do { \
        auto g = llama::metrics::get_gauge(name); \
        if (g) g->set(value); \
    } while(0)

#define METRIC_OBSERVE(name, value) \
    do { \
        auto h = llama::metrics::get_histogram(name); \
        if (h) h->observe(value); \
    } while(0)

// RAII timer for histogram observations
class scoped_timer {
public:
    scoped_timer(const std::string& histogram_name)
        : histogram_name_(histogram_name),
          start_(std::chrono::high_resolution_clock::now()) {}
    
    ~scoped_timer() {
        auto duration = std::chrono::high_resolution_clock::now() - start_;
        auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        METRIC_OBSERVE(histogram_name_, static_cast<double>(microseconds));
    }
    
private:
    std::string histogram_name_;
    std::chrono::high_resolution_clock::time_point start_;
};

#define METRIC_TIMER(name) llama::metrics::scoped_timer _metric_timer_##__LINE__(name)

// Common metric tracking functions
void log_metrics_snapshot();
void log_metrics_detailed();

// Model metrics
void track_model_load_start(const std::string& model_name);
void track_model_load_end(const std::string& model_name, int64_t duration_us);
void track_model_unload(const std::string& model_name);

// Generation metrics
void track_tokens_generated(size_t count);
void track_generation_request_start();
void track_generation_request_end(int64_t duration_us);

// Batch processing metrics
void track_batch_processed(size_t batch_size, int64_t duration_us);

// Memory metrics
void track_memory_allocation(size_t bytes);
void track_memory_deallocation(size_t bytes);
void update_memory_usage(size_t used_bytes);

// Cache metrics
void track_cache_hit();
void track_cache_miss();
void track_cache_lookup(int64_t duration_us);
void update_cache_size(size_t bytes);

// Error tracking
void track_error(const std::string& error_type);
void track_warning(const std::string& warning_type);

} // namespace metrics
} // namespace llama