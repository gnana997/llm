#pragma once

#include <atomic>
#include <chrono>
#include <string>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <memory>

namespace llama {
namespace orchestration {

// Metrics types
enum class MetricType {
    COUNTER,
    GAUGE,
    HISTOGRAM
};

// Base metric class
class Metric {
public:
    Metric(const std::string& name, MetricType type) 
        : name_(name), type_(type) {}
    
    virtual ~Metric() = default;
    
    const std::string& name() const { return name_; }
    MetricType type() const { return type_; }
    
    virtual std::string to_string() const = 0;
    
protected:
    std::string name_;
    MetricType type_;
};

// Counter metric - monotonically increasing value
class Counter : public Metric {
public:
    Counter(const std::string& name) 
        : Metric(name, MetricType::COUNTER), value_(0) {}
    
    void increment(int64_t delta = 1) {
        value_.fetch_add(delta, std::memory_order_relaxed);
    }
    
    int64_t value() const {
        return value_.load(std::memory_order_relaxed);
    }
    
    std::string to_string() const override {
        return name_ + ": " + std::to_string(value());
    }
    
private:
    std::atomic<int64_t> value_;
};

// Gauge metric - can go up or down
class Gauge : public Metric {
public:
    Gauge(const std::string& name) 
        : Metric(name, MetricType::GAUGE), value_(0.0) {}
    
    void set(double value) {
        std::lock_guard<std::mutex> lock(mutex_);
        value_ = value;
    }
    
    double value() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return value_;
    }
    
    std::string to_string() const override {
        return name_ + ": " + std::to_string(value());
    }
    
private:
    mutable std::mutex mutex_;
    double value_;
};

// Histogram metric - distribution of values
class Histogram : public Metric {
public:
    Histogram(const std::string& name) 
        : Metric(name, MetricType::HISTOGRAM) {}
    
    void observe(double value) {
        std::lock_guard<std::mutex> lock(mutex_);
        values_.push_back(value);
        sum_ += value;
        count_++;
        
        if (value < min_) min_ = value;
        if (value > max_) max_ = value;
    }
    
    double mean() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_ > 0 ? sum_ / count_ : 0.0;
    }
    
    double min() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return min_;
    }
    
    double max() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return max_;
    }
    
    size_t count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_;
    }
    
    std::string to_string() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return name_ + ": {count=" + std::to_string(count_) + 
               ", mean=" + std::to_string(mean()) +
               ", min=" + std::to_string(min_) +
               ", max=" + std::to_string(max_) + "}";
    }
    
private:
    mutable std::mutex mutex_;
    std::vector<double> values_;
    double sum_ = 0.0;
    size_t count_ = 0;
    double min_ = std::numeric_limits<double>::max();
    double max_ = std::numeric_limits<double>::lowest();
};

// Metrics registry singleton
class MetricsRegistry {
public:
    static MetricsRegistry& instance() {
        static MetricsRegistry instance;
        return instance;
    }
    
    // Register metrics
    Counter* register_counter(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto counter = std::make_unique<Counter>(name);
        Counter* ptr = counter.get();
        metrics_[name] = std::move(counter);
        return ptr;
    }
    
    Gauge* register_gauge(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto gauge = std::make_unique<Gauge>(name);
        Gauge* ptr = gauge.get();
        metrics_[name] = std::move(gauge);
        return ptr;
    }
    
    Histogram* register_histogram(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto histogram = std::make_unique<Histogram>(name);
        Histogram* ptr = histogram.get();
        metrics_[name] = std::move(histogram);
        return ptr;
    }
    
    // Get metric by name
    Metric* get(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = metrics_.find(name);
        return it != metrics_.end() ? it->second.get() : nullptr;
    }
    
    // Log all metrics
    void log_all() const;
    
    // Clear all metrics
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_.clear();
    }
    
private:
    MetricsRegistry() = default;
    MetricsRegistry(const MetricsRegistry&) = delete;
    MetricsRegistry& operator=(const MetricsRegistry&) = delete;
    
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<Metric>> metrics_;
};

// Convenience class for timing operations
class TimerScope {
public:
    TimerScope(Histogram* histogram) 
        : histogram_(histogram), start_(std::chrono::high_resolution_clock::now()) {}
    
    ~TimerScope() {
        if (histogram_) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
            histogram_->observe(duration.count());
        }
    }
    
private:
    Histogram* histogram_;
    std::chrono::high_resolution_clock::time_point start_;
};

// Layer distribution specific metrics
struct LayerDistributionMetrics {
    Counter* cache_hits;
    Counter* cache_misses;
    Counter* total_distributions;
    Gauge* gpu_layers_assigned[8];  // Support up to 8 GPUs
    Histogram* distribution_duration_us;
    Histogram* layer_compute_time_us;
    
    static LayerDistributionMetrics& instance() {
        static LayerDistributionMetrics metrics;
        return metrics;
    }
    
    void initialize() {
        auto& registry = MetricsRegistry::instance();
        
        cache_hits = registry.register_counter("layer_distribution_cache_hits");
        cache_misses = registry.register_counter("layer_distribution_cache_misses");
        total_distributions = registry.register_counter("layer_distribution_total");
        
        distribution_duration_us = registry.register_histogram("layer_distribution_duration_us");
        layer_compute_time_us = registry.register_histogram("layer_compute_time_us");
        
        for (int i = 0; i < 8; ++i) {
            std::string name = "gpu_" + std::to_string(i) + "_layers_assigned";
            gpu_layers_assigned[i] = registry.register_gauge(name);
        }
    }
    
private:
    LayerDistributionMetrics() {
        initialize();
    }
};

} // namespace orchestration
} // namespace llama