#include "metrics.h"
#include "../src/log/log-ex.h"
#include <sstream>
#include <iomanip>
#include <numeric>

namespace llama {
namespace metrics {

// Static initialization of common metrics
static void initialize_common_metrics() {
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
        auto& registry = metric_registry::instance();
        
        // Model loading metrics
        registry.register_metric<counter>("model_loads_total", "Total number of models loaded");
        registry.register_metric<histogram>("model_load_duration_us", "Model load duration in microseconds",
            std::vector<double>{1000000, 2500000, 5000000, 10000000, 25000000, 50000000}); // 1s to 50s
        registry.register_metric<gauge>("models_loaded", "Number of models currently loaded");
        
        // Token generation metrics
        registry.register_metric<counter>("tokens_generated_total", "Total number of tokens generated");
        registry.register_metric<histogram>("token_generation_duration_us", "Token generation duration in microseconds");
        registry.register_metric<gauge>("active_generation_requests", "Number of active generation requests");
        
        // Batch processing metrics
        registry.register_metric<counter>("batches_processed_total", "Total number of batches processed");
        registry.register_metric<histogram>("batch_size", "Distribution of batch sizes",
            std::vector<double>{1, 2, 4, 8, 16, 32, 64, 128, 256, 512});
        registry.register_metric<histogram>("batch_processing_duration_us", "Batch processing duration in microseconds");
        
        // Memory metrics
        registry.register_metric<gauge>("memory_used_bytes", "Total memory used in bytes");
        registry.register_metric<gauge>("memory_allocated_bytes", "Total memory allocated in bytes");
        registry.register_metric<counter>("memory_allocations_total", "Total number of memory allocations");
        registry.register_metric<counter>("memory_deallocations_total", "Total number of memory deallocations");
        
        // Cache metrics
        registry.register_metric<counter>("cache_hits_total", "Total number of cache hits");
        registry.register_metric<counter>("cache_misses_total", "Total number of cache misses");
        registry.register_metric<gauge>("cache_size_bytes", "Current cache size in bytes");
        registry.register_metric<histogram>("cache_lookup_duration_us", "Cache lookup duration in microseconds",
            std::vector<double>{1, 5, 10, 25, 50, 100, 250, 500, 1000});
        
        // Error metrics
        registry.register_metric<counter>("errors_total", "Total number of errors");
        registry.register_metric<counter>("warnings_total", "Total number of warnings");
        
        // Request metrics
        registry.register_metric<counter>("requests_total", "Total number of requests");
        registry.register_metric<gauge>("requests_active", "Number of active requests");
        registry.register_metric<histogram>("request_duration_us", "Request duration in microseconds",
            std::vector<double>{1000, 10000, 100000, 1000000, 10000000, 60000000}); // 1ms to 60s
        
        LOG_TRACE("Initialized common metrics\n");
    });
}

// Ensure metrics are initialized on library load
struct metrics_initializer {
    metrics_initializer() {
        initialize_common_metrics();
    }
} static_metrics_init;

// Format a metric value for logging
static std::string format_metric_value(const std::string& name, double value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    
    // Special formatting for memory metrics
    if (name.find("bytes") != std::string::npos) {
        if (value >= 1073741824) {
            oss << name << "=" << (value / 1073741824) << "GB";
        } else if (value >= 1048576) {
            oss << name << "=" << (value / 1048576) << "MB";
        } else if (value >= 1024) {
            oss << name << "=" << (value / 1024) << "KB";
        } else {
            oss << name << "=" << value << "B";
        }
    }
    // Special formatting for time metrics
    else if (name.find("duration_us") != std::string::npos || name.find("_us") != std::string::npos) {
        if (value >= 1000000) {
            oss << name << "=" << (value / 1000000) << "s";
        } else if (value >= 1000) {
            oss << name << "=" << (value / 1000) << "ms";
        } else {
            oss << name << "=" << value << "μs";
        }
    }
    // Special formatting for percentages
    else if (name.find("percent") != std::string::npos || name.find("utilization") != std::string::npos) {
        oss << name << "=" << value << "%";
    }
    // Default formatting
    else {
        oss << name << "=" << value;
    }
    
    return oss.str();
}

// Export metrics to log
void log_metrics_snapshot() {
    auto& registry = metric_registry::instance();
    auto all_values = registry.get_all_values();
    
    if (all_values.empty()) {
        return;
    }
    
    std::ostringstream oss;
    oss << "Metrics snapshot: ";
    
    bool first = true;
    for (const auto& [name, value] : all_values) {
        // Skip histogram buckets and internal metrics in summary log
        if (name.find("_bucket_") != std::string::npos) continue;
        
        if (!first) oss << ", ";
        first = false;
        
        oss << format_metric_value(name, value);
    }
    
    LOG_PERF("%s\n", oss.str().c_str());
}

// Export detailed metrics to log (including histogram buckets)
void log_metrics_detailed() {
    auto& registry = metric_registry::instance();
    auto all_metrics = registry.get_all();
    
    LOG_PERF("=== Detailed Metrics Report ===\n");
    
    // Group metrics by type
    std::vector<std::shared_ptr<metric>> counters, gauges, histograms, summaries;
    
    for (const auto& m : all_metrics) {
        switch (m->type()) {
            case metric_type::COUNTER:
                counters.push_back(m);
                break;
            case metric_type::GAUGE:
                gauges.push_back(m);
                break;
            case metric_type::HISTOGRAM:
                histograms.push_back(m);
                break;
            case metric_type::SUMMARY:
                summaries.push_back(m);
                break;
        }
    }
    
    // Log counters
    if (!counters.empty()) {
        LOG_PERF("Counters:\n");
        for (const auto& m : counters) {
            auto values = m->get_values();
            for (const auto& [name, value] : values) {
                LOG_PERF("  %s\n", format_metric_value(name, value).c_str());
            }
        }
    }
    
    // Log gauges
    if (!gauges.empty()) {
        LOG_PERF("Gauges:\n");
        for (const auto& m : gauges) {
            auto values = m->get_values();
            for (const auto& [name, value] : values) {
                LOG_PERF("  %s\n", format_metric_value(name, value).c_str());
            }
        }
    }
    
    // Log histograms
    if (!histograms.empty()) {
        LOG_PERF("Histograms:\n");
        for (const auto& m : histograms) {
            auto values = m->get_values();
            LOG_PERF("  %s:\n", m->name().c_str());
            
            // Extract statistics
            double count = 0, sum = 0, min_val = 0, max_val = 0, avg = 0;
            for (const auto& [name, value] : values) {
                if (name.find("_count") != std::string::npos) count = value;
                else if (name.find("_sum") != std::string::npos) sum = value;
                else if (name.find("_min") != std::string::npos) min_val = value;
                else if (name.find("_max") != std::string::npos) max_val = value;
                else if (name.find("_avg") != std::string::npos) avg = value;
            }
            
            LOG_PERF("    Count: %.0f, Sum: %.2f, Avg: %.2f, Min: %.2f, Max: %.2f\n",
                     count, sum, avg, min_val, max_val);
            
            // Calculate percentiles if histogram
            if (auto hist = std::dynamic_pointer_cast<histogram>(m)) {
                LOG_PERF("    P50: %.2f, P95: %.2f, P99: %.2f\n",
                         hist->percentile(50), hist->percentile(95), hist->percentile(99));
            }
        }
    }
    
    // Log summaries
    if (!summaries.empty()) {
        LOG_PERF("Summaries:\n");
        for (const auto& m : summaries) {
            auto values = m->get_values();
            LOG_PERF("  %s:\n", m->name().c_str());
            
            // Group by time window
            std::unordered_map<std::string, std::unordered_map<std::string, double>> windows;
            for (const auto& [name, value] : values) {
                size_t underscore = name.find('_');
                size_t second_underscore = name.find('_', underscore + 1);
                if (second_underscore != std::string::npos) {
                    std::string window = name.substr(underscore + 1, second_underscore - underscore - 1);
                    std::string metric = name.substr(second_underscore + 1);
                    windows[window][metric] = value;
                }
            }
            
            for (const auto& [window, metrics] : windows) {
                LOG_PERF("    Window %s: ", window.c_str());
                LOG_PERF("Count=%.0f, Avg=%.2f, Min=%.2f, Max=%.2f, P50=%.2f, P95=%.2f, P99=%.2f\n",
                         metrics.at("count"), metrics.at("avg"), metrics.at("min"), metrics.at("max"),
                         metrics.at("p50"), metrics.at("p95"), metrics.at("p99"));
            }
        }
    }
    
    LOG_PERF("=== End Metrics Report ===\n");
}

// Helper to track model operations
void track_model_load_start(const std::string& model_name) {
    METRIC_INCREMENT("model_loads_total", 1);
    METRIC_INCREMENT("models_loaded", 1);
    LOG_PERF("Starting model load: %s\n", model_name.c_str());
}

void track_model_load_end(const std::string& model_name, int64_t duration_us) {
    METRIC_OBSERVE("model_load_duration_us", static_cast<double>(duration_us));
    LOG_PERF("Completed model load: %s (duration: %lldμs)\n", model_name.c_str(), duration_us);
}

void track_model_unload(const std::string& model_name) {
    METRIC_SET("models_loaded", std::max(0.0, get_gauge("models_loaded")->get() - 1));
    LOG_PERF("Unloaded model: %s\n", model_name.c_str());
}

// Helper to track token generation
void track_tokens_generated(size_t count) {
    METRIC_INCREMENT("tokens_generated_total", static_cast<double>(count));
}

void track_generation_request_start() {
    METRIC_INCREMENT("requests_total", 1);
    METRIC_INCREMENT("requests_active", 1);
    METRIC_INCREMENT("active_generation_requests", 1);
}

void track_generation_request_end(int64_t duration_us) {
    METRIC_INCREMENT("requests_active", -1);
    METRIC_INCREMENT("active_generation_requests", -1);
    METRIC_OBSERVE("request_duration_us", static_cast<double>(duration_us));
}

// Helper to track batch processing
void track_batch_processed(size_t batch_size, int64_t duration_us) {
    METRIC_INCREMENT("batches_processed_total", 1);
    METRIC_OBSERVE("batch_size", static_cast<double>(batch_size));
    METRIC_OBSERVE("batch_processing_duration_us", static_cast<double>(duration_us));
}

// Helper to track memory operations
void track_memory_allocation(size_t bytes) {
    METRIC_INCREMENT("memory_allocations_total", 1);
    METRIC_INCREMENT("memory_allocated_bytes", static_cast<double>(bytes));
}

void track_memory_deallocation(size_t bytes) {
    METRIC_INCREMENT("memory_deallocations_total", 1);
    METRIC_INCREMENT("memory_allocated_bytes", -static_cast<double>(bytes));
}

void update_memory_usage(size_t used_bytes) {
    METRIC_SET("memory_used_bytes", static_cast<double>(used_bytes));
}

// Helper to track cache operations
void track_cache_hit() {
    METRIC_INCREMENT("cache_hits_total", 1);
}

void track_cache_miss() {
    METRIC_INCREMENT("cache_misses_total", 1);
}

void track_cache_lookup(int64_t duration_us) {
    METRIC_OBSERVE("cache_lookup_duration_us", static_cast<double>(duration_us));
}

void update_cache_size(size_t bytes) {
    METRIC_SET("cache_size_bytes", static_cast<double>(bytes));
}

// Helper to track errors
void track_error(const std::string& error_type) {
    METRIC_INCREMENT("errors_total", 1);
    LOG_CAT(COMMON_LOG_CAT_GENERAL, GGML_LOG_LEVEL_ERROR, "Error tracked: %s\n", error_type.c_str());
}

void track_warning(const std::string& warning_type) {
    METRIC_INCREMENT("warnings_total", 1);
    LOG_CAT(COMMON_LOG_CAT_GENERAL, GGML_LOG_LEVEL_WARN, "Warning tracked: %s\n", warning_type.c_str());
}

} // namespace metrics
} // namespace llama