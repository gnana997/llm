#include "metrics.h"
#include "llama-impl.h"
#include <iomanip>
#include <sstream>

namespace llama {
namespace orchestration {

void MetricsRegistry::log_all() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (metrics_.empty()) {
        return;
    }
    
    LLAMA_LOG_INFO("\n========== Layer Distribution Metrics ==========\n");
    
    // Group metrics by type
    std::vector<const Metric*> counters;
    std::vector<const Metric*> gauges;
    std::vector<const Metric*> histograms;
    
    for (const auto& [name, metric] : metrics_) {
        switch (metric->type()) {
            case MetricType::COUNTER:
                counters.push_back(metric.get());
                break;
            case MetricType::GAUGE:
                gauges.push_back(metric.get());
                break;
            case MetricType::HISTOGRAM:
                histograms.push_back(metric.get());
                break;
        }
    }
    
    // Log counters
    if (!counters.empty()) {
        LLAMA_LOG_INFO("Counters:\n");
        for (const auto* metric : counters) {
            LLAMA_LOG_INFO("  %s\n", metric->to_string().c_str());
        }
        LLAMA_LOG_INFO("\n");
    }
    
    // Log gauges
    if (!gauges.empty()) {
        LLAMA_LOG_INFO("Gauges:\n");
        for (const auto* metric : gauges) {
            LLAMA_LOG_INFO("  %s\n", metric->to_string().c_str());
        }
        LLAMA_LOG_INFO("\n");
    }
    
    // Log histograms
    if (!histograms.empty()) {
        LLAMA_LOG_INFO("Histograms:\n");
        for (const auto* metric : histograms) {
            LLAMA_LOG_INFO("  %s\n", metric->to_string().c_str());
        }
        LLAMA_LOG_INFO("\n");
    }
    
    LLAMA_LOG_INFO("===============================================\n");
}

} // namespace orchestration
} // namespace llama