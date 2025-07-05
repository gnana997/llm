#include "advanced_metrics.h"
#include "orchestration_logger.h"
#include "llama-impl.h"
#include <algorithm>
#include <iostream>
#include <numeric>

namespace llama {
namespace orchestration {

// Export metrics in various formats
void export_advanced_metrics_json(std::ostream& out) {
    auto& metrics = AdvancedLayerMetrics::instance();
    auto& registry = MetricsRegistry::instance();
    
    out << "{\n";
    out << "  \"advanced_layer_metrics\": {\n";
    
    // Export counters
    out << "    \"counters\": {\n";
    out << "      \"layer_migration_count\": " << metrics.layer_migration_count->value() << ",\n";
    out << "      \"performance_anomaly_count\": " << metrics.performance_anomaly_count->value() << ",\n";
    out << "      \"thermal_throttle_events\": " << metrics.thermal_throttle_events->value() << ",\n";
    out << "      \"cross_gpu_transfer_bytes\": " << metrics.cross_gpu_transfer_bytes->value() << ",\n";
    out << "      \"memory_allocation_failures\": " << metrics.memory_allocation_failures->value() << "\n";
    out << "    },\n";
    
    // Export gauges
    out << "    \"gauges\": {\n";
    
    // GPU-specific metrics
    bool first_gpu = true;
    for (int i = 0; i < 8; ++i) {
        if (metrics.gpu_utilization_percent[i]->value() > 0) {
            if (!first_gpu) out << ",\n";
            first_gpu = false;
            
            out << "      \"gpu_" << i << "\": {\n";
            out << "        \"memory_bandwidth_utilization\": " 
                << metrics.gpu_memory_bandwidth_utilization[i]->value() << ",\n";
            out << "        \"layer_affinity_score\": " 
                << metrics.layer_affinity_score[i]->value() << ",\n";
            out << "        \"utilization_percent\": " 
                << metrics.gpu_utilization_percent[i]->value() << ",\n";
            out << "        \"temperature_celsius\": " 
                << metrics.gpu_temperature_celsius[i]->value() << ",\n";
            out << "        \"power_watts\": " 
                << metrics.gpu_power_watts[i]->value() << "\n";
            out << "      }";
        }
    }
    if (!first_gpu) out << ",\n";
    
    // Quality metrics
    out << "      \"distribution_quality\": {\n";
    out << "        \"load_imbalance_factor\": " << metrics.load_imbalance_factor->value() << ",\n";
    out << "        \"memory_imbalance_factor\": " << metrics.memory_imbalance_factor->value() << ",\n";
    out << "        \"predicted_vs_actual_speedup\": " << metrics.predicted_vs_actual_speedup->value() << "\n";
    out << "      }\n";
    out << "    },\n";
    
    // Export histograms
    out << "    \"histograms\": {\n";
    out << "      \"layer_queue_wait_us\": " << metrics.layer_queue_wait_us->to_string() << ",\n";
    out << "      \"gpu_kernel_launch_us\": " << metrics.gpu_kernel_launch_us->to_string();
    
    // Export per-layer compute times (only non-null ones)
    [[maybe_unused]] bool first_layer = true;
    for (int i = 0; i < 128; ++i) {
        if (metrics.layer_compute_time_us[i] && metrics.layer_compute_time_us[i]->count() > 0) {
            out << ",\n";
            out << "      \"layer_" << i << "_compute_time_us\": " 
                << metrics.layer_compute_time_us[i]->to_string();
        }
    }
    
    out << "\n    }\n";
    out << "  }\n";
    out << "}\n";
}

// Export metrics in Prometheus format
void export_advanced_metrics_prometheus(std::ostream& out) {
    auto& metrics = AdvancedLayerMetrics::instance();
    
    // Counters
    out << "# HELP layer_migration_count Number of runtime layer redistributions\n";
    out << "# TYPE layer_migration_count counter\n";
    out << "layer_migration_count " << metrics.layer_migration_count->value() << "\n\n";
    
    out << "# HELP performance_anomaly_count Number of detected performance anomalies\n";
    out << "# TYPE performance_anomaly_count counter\n";
    out << "performance_anomaly_count " << metrics.performance_anomaly_count->value() << "\n\n";
    
    out << "# HELP thermal_throttle_events Number of GPU thermal throttle events\n";
    out << "# TYPE thermal_throttle_events counter\n";
    out << "thermal_throttle_events " << metrics.thermal_throttle_events->value() << "\n\n";
    
    out << "# HELP cross_gpu_transfer_bytes Bytes transferred between GPUs\n";
    out << "# TYPE cross_gpu_transfer_bytes counter\n";
    out << "cross_gpu_transfer_bytes " << metrics.cross_gpu_transfer_bytes->value() << "\n\n";
    
    // GPU-specific gauges
    for (int i = 0; i < 8; ++i) {
        if (metrics.gpu_utilization_percent[i]->value() > 0) {
            out << "# HELP gpu_memory_bandwidth_utilization GPU memory bandwidth utilization percentage\n";
            out << "# TYPE gpu_memory_bandwidth_utilization gauge\n";
            out << "gpu_memory_bandwidth_utilization{gpu=\"" << i << "\"} " 
                << metrics.gpu_memory_bandwidth_utilization[i]->value() << "\n";
            
            out << "# HELP gpu_utilization_percent GPU compute utilization percentage\n";
            out << "# TYPE gpu_utilization_percent gauge\n";
            out << "gpu_utilization_percent{gpu=\"" << i << "\"} " 
                << metrics.gpu_utilization_percent[i]->value() << "\n";
            
            out << "# HELP gpu_temperature_celsius GPU temperature in Celsius\n";
            out << "# TYPE gpu_temperature_celsius gauge\n";
            out << "gpu_temperature_celsius{gpu=\"" << i << "\"} " 
                << metrics.gpu_temperature_celsius[i]->value() << "\n";
            
            out << "# HELP gpu_power_watts GPU power consumption in watts\n";
            out << "# TYPE gpu_power_watts gauge\n";
            out << "gpu_power_watts{gpu=\"" << i << "\"} " 
                << metrics.gpu_power_watts[i]->value() << "\n\n";
        }
    }
    
    // Distribution quality metrics
    out << "# HELP load_imbalance_factor Load distribution imbalance factor\n";
    out << "# TYPE load_imbalance_factor gauge\n";
    out << "load_imbalance_factor " << metrics.load_imbalance_factor->value() << "\n\n";
    
    out << "# HELP memory_imbalance_factor Memory distribution imbalance factor\n";
    out << "# TYPE memory_imbalance_factor gauge\n";
    out << "memory_imbalance_factor " << metrics.memory_imbalance_factor->value() << "\n\n";
}

// Log comprehensive metrics summary
void log_advanced_metrics_summary() {
    auto& metrics = AdvancedLayerMetrics::instance();
    [[maybe_unused]] auto& logger = OrchestrationLogger::instance();
    
    ORCH_LOG_INFO("========== Advanced Layer Distribution Metrics ==========\n");
    
    // Runtime statistics
    ORCH_LOG_INFO("Runtime Statistics:\n");
    ORCH_LOG_INFO("  Layer migrations: %ld\n", metrics.layer_migration_count->value());
    ORCH_LOG_INFO("  Performance anomalies: %ld\n", metrics.performance_anomaly_count->value());
    ORCH_LOG_INFO("  Thermal throttle events: %ld\n", metrics.thermal_throttle_events->value());
    ORCH_LOG_INFO("  Cross-GPU transfer: %.2f GB\n", 
                  metrics.cross_gpu_transfer_bytes->value() / (1024.0 * 1024.0 * 1024.0));
    ORCH_LOG_INFO("  Memory allocation failures: %ld\n", 
                  metrics.memory_allocation_failures->value());
    
    // GPU utilization summary
    ORCH_LOG_INFO("\nGPU Utilization:\n");
    for (int i = 0; i < 8; ++i) {
        if (metrics.gpu_utilization_percent[i]->value() > 0) {
            ORCH_LOG_INFO("  GPU %d: %.1f%% compute, %.1f%% memory bandwidth, "
                         "%.1f°C, %.1fW\n",
                         i,
                         metrics.gpu_utilization_percent[i]->value(),
                         metrics.gpu_memory_bandwidth_utilization[i]->value(),
                         metrics.gpu_temperature_celsius[i]->value(),
                         metrics.gpu_power_watts[i]->value());
        }
    }
    
    // Distribution quality
    ORCH_LOG_INFO("\nDistribution Quality:\n");
    ORCH_LOG_INFO("  Load imbalance factor: %.2f\n", 
                  metrics.load_imbalance_factor->value());
    ORCH_LOG_INFO("  Memory imbalance factor: %.2f\n", 
                  metrics.memory_imbalance_factor->value());
    ORCH_LOG_INFO("  Prediction accuracy: %.2f\n", 
                  metrics.predicted_vs_actual_speedup->value());
    
    // Performance histograms
    ORCH_LOG_INFO("\nPerformance Histograms:\n");
    ORCH_LOG_INFO("  Layer queue wait: %s\n", 
                  metrics.layer_queue_wait_us->to_string().c_str());
    ORCH_LOG_INFO("  Kernel launch overhead: %s\n", 
                  metrics.gpu_kernel_launch_us->to_string().c_str());
    
    // Per-layer statistics (top 5 slowest layers)
    std::vector<std::pair<int, double>> layer_times;
    for (int i = 0; i < 128; ++i) {
        if (metrics.layer_compute_time_us[i] && metrics.layer_compute_time_us[i]->count() > 0) {
            layer_times.push_back({i, metrics.layer_compute_time_us[i]->mean()});
        }
    }
    
    if (!layer_times.empty()) {
        std::sort(layer_times.begin(), layer_times.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        ORCH_LOG_INFO("\nSlowest Layers (top 5):\n");
        for (size_t i = 0; i < std::min(size_t(5), layer_times.size()); ++i) {
            ORCH_LOG_INFO("  Layer %d: %.1f ms average\n",
                         layer_times[i].first, layer_times[i].second / 1000.0);
        }
    }
    
    ORCH_LOG_INFO("========================================================\n");
}

} // namespace orchestration
} // namespace llama