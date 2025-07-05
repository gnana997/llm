#pragma once

#include "../common/log-ex.h"
#include "gpu_profiler_interface.h"
#include "layer_distributor.h"
#include <string>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace llama {
namespace orchestration {

// Orchestration-specific log categories
constexpr uint32_t LOG_CAT_ORCHESTRATION = COMMON_LOG_CAT_GPU;  // Reuse GPU category
constexpr uint32_t LOG_CAT_DISTRIBUTION = COMMON_LOG_CAT_INFERENCE;
constexpr uint32_t LOG_CAT_PROFILING = COMMON_LOG_CAT_PERF;

// Orchestration event types
enum class OrchestrationEventType {
    LAYER_DISTRIBUTION_START,
    LAYER_DISTRIBUTION_END,
    GPU_PROFILE_START,
    GPU_PROFILE_END,
    LAYER_ASSIGNMENT,
    REDISTRIBUTION_TRIGGERED,
    PERFORMANCE_ANOMALY,
    THERMAL_THROTTLE_DETECTED,
    MEMORY_PRESSURE_DETECTED
};

// Structured logging wrapper for orchestration components
class OrchestrationLogger {
public:
    static OrchestrationLogger& instance() {
        static OrchestrationLogger logger;
        return logger;
    }
    
    // Log GPU profiling events
    void log_gpu_profile(const GpuProfile& profile, bool is_start = false) {
        if (is_start) {
            LOG_PERF_START(PERF_EVENT_GPU_ALLOC, profile.name);
        }
        
        LOG_CAT(static_cast<common_log_category>(LOG_CAT_PROFILING), GGML_LOG_LEVEL_INFO,
                "[GPU Profile] %s: Memory %.2f GB, Compute %.1f TFLOPS, Bandwidth %.1f GB/s\n",
                profile.name.c_str(),
                profile.total_memory_bytes / (1024.0 * 1024.0 * 1024.0),
                profile.theoretical_tflops,
                profile.memory_bandwidth_gbps);
        
        // Log capability score
        LOG_CAT(static_cast<common_log_category>(LOG_CAT_PROFILING), GGML_LOG_LEVEL_INFO,
                "  Capability Score: %.1f/100\n", profile.get_capability_score());
    }
    
    // Log layer distribution events
    void log_distribution_start(const std::string& model_name, int total_layers) {
        LOG_PERF_START(PERF_EVENT_MODEL_LOAD_START, model_name);
        LOG_CAT(static_cast<common_log_category>(LOG_CAT_DISTRIBUTION), GGML_LOG_LEVEL_INFO,
                "[Distribution] Starting layer distribution for model '%s' with %d layers\n",
                model_name.c_str(), total_layers);
    }
    
    void log_distribution_end(const DistributionResult& result, 
                             std::chrono::microseconds duration) {
        LOG_PERF_END(PERF_EVENT_MODEL_LOAD_END, "distribution");
        
        LOG_CAT(static_cast<common_log_category>(LOG_CAT_DISTRIBUTION), GGML_LOG_LEVEL_INFO,
                "[Distribution] Completed in %ld us. Expected speedup: %.1fx, Cache hit: %s\n",
                duration.count(), result.expected_speedup, 
                result.cache_hit ? "true" : "false");
        
        // Log summary
        for (size_t i = 0; i < result.layers_per_gpu.size(); ++i) {
            LOG_CAT(static_cast<common_log_category>(LOG_CAT_DISTRIBUTION), GGML_LOG_LEVEL_INFO,
                    "  GPU %zu: %d layers, %.1f GB memory, %.1f%% compute load\n",
                    i, result.layers_per_gpu[i], result.memory_per_gpu[i],
                    result.compute_load_per_gpu[i]);
        }
    }
    
    // Log individual layer assignments
    void log_layer_assignment(const LayerAssignment& assignment) {
        LOG_TRACE("[Layer Assignment] Layer %d -> GPU %d (Pattern: %s, Affinity: %.2f, "
                  "Predicted time: %.1f ms, %s)\n",
                  assignment.layer_index, assignment.gpu_id,
                  get_pattern_name(assignment.compute_pattern).c_str(),
                  assignment.architecture_affinity_score,
                  assignment.predicted_execution_time_ms,
                  assignment.rationale.c_str());
        
        // Performance event for layer assignment
        common_perf_event event;
        event.type = PERF_EVENT_CUSTOM;
        event.timestamp = std::chrono::high_resolution_clock::now();
        event.name = "layer_assignment";
        event.metadata["layer_id"] = std::to_string(assignment.layer_index);
        event.metadata["gpu_id"] = std::to_string(assignment.gpu_id);
        event.metadata["pattern"] = get_pattern_name(assignment.compute_pattern);
        common_log_ex::instance().log_perf_event(event);
    }
    
    // Log runtime performance events
    void log_performance_anomaly(int gpu_id, const std::string& description, 
                                float severity) {
        LOG_CAT(static_cast<common_log_category>(LOG_CAT_ORCHESTRATION), GGML_LOG_LEVEL_WARN,
                "[Performance Anomaly] GPU %d: %s (severity: %.1f)\n",
                gpu_id, description.c_str(), severity);
    }
    
    void log_thermal_throttle(int gpu_id, float temperature, float reduction_factor) {
        LOG_CAT(static_cast<common_log_category>(LOG_CAT_ORCHESTRATION), GGML_LOG_LEVEL_WARN,
                "[Thermal Throttle] GPU %d at %.1f°C, performance reduced by %.1f%%\n",
                gpu_id, temperature, (1.0f - reduction_factor) * 100.0f);
    }
    
    void log_redistribution_triggered(const std::string& reason) {
        LOG_CAT(static_cast<common_log_category>(LOG_CAT_ORCHESTRATION), GGML_LOG_LEVEL_INFO,
                "[Redistribution] Triggered: %s\n", reason.c_str());
        LOG_PERF_START(PERF_EVENT_CUSTOM, "layer_redistribution");
    }
    
    void log_redistribution_completed(bool success, int layers_moved, 
                                     std::chrono::microseconds duration) {
        LOG_PERF_END(PERF_EVENT_CUSTOM, "layer_redistribution");
        
        if (success) {
            LOG_CAT(static_cast<common_log_category>(LOG_CAT_ORCHESTRATION), GGML_LOG_LEVEL_INFO,
                    "[Redistribution] Success: Moved %d layers in %ld us\n",
                    layers_moved, duration.count());
        } else {
            LOG_CAT(static_cast<common_log_category>(LOG_CAT_ORCHESTRATION), GGML_LOG_LEVEL_ERROR,
                    "[Redistribution] Failed after %ld us\n", duration.count());
        }
    }
    
    // Log memory operations
    void log_memory_transfer(int src_gpu, int dst_gpu, size_t bytes, 
                           std::chrono::microseconds duration) {
        LOG_TRACE("[Memory Transfer] GPU %d -> GPU %d: %.2f MB in %ld us (%.1f GB/s)\n",
                  src_gpu, dst_gpu, bytes / (1024.0 * 1024.0), duration.count(),
                  (bytes / 1e9) / (duration.count() / 1e6));
        
        // Performance event
        common_perf_event event;
        event.type = PERF_EVENT_GPU_COPY_TO;
        event.timestamp = std::chrono::high_resolution_clock::now();
        event.duration_us = duration.count();
        event.size = bytes;
        event.metadata["src_gpu"] = std::to_string(src_gpu);
        event.metadata["dst_gpu"] = std::to_string(dst_gpu);
        common_log_ex::instance().log_perf_event(event);
    }
    
    // Architecture analysis logging
    void log_architecture_analysis(const std::string& gpu_name, 
                                  const std::string& architecture,
                                  const ArchitectureCapabilities& caps) {
        LOG_CAT(static_cast<common_log_category>(LOG_CAT_PROFILING), GGML_LOG_LEVEL_INFO,
                "[Architecture] %s detected as %s: TensorCores=%s (gen %d), "
                "MatrixCores=%s, AsyncCopy=%s, TDP=%.0fW\n",
                gpu_name.c_str(), architecture.c_str(),
                caps.has_tensor_cores ? "yes" : "no", caps.tensor_core_generation,
                caps.has_matrix_cores ? "yes" : "no",
                caps.supports_async_copy ? "yes" : "no",
                caps.tdp_watts);
    }
    
    // Utility to enable orchestration logging
    static void enable_orchestration_logs(bool perf_tracking = true, 
                                        bool trace_enabled = false) {
        auto& logger = common_log_ex::instance();
        
        // Enable relevant categories
        uint32_t categories = LOG_CAT_ORCHESTRATION | LOG_CAT_DISTRIBUTION;
        if (perf_tracking) {
            categories |= LOG_CAT_PROFILING;
        }
        logger.set_category_filter(categories);
        
        // Enable performance summary if requested
        logger.enable_performance_summary(perf_tracking);
        
        // Set appropriate log level
        if (trace_enabled) {
            common_log_set_verbosity_thold(1);  // Enable trace logs
        }
    }
    
private:
    OrchestrationLogger() = default;
    
    std::string get_pattern_name(LayerComputePattern pattern) const {
        switch (pattern) {
            case LayerComputePattern::GEMM_HEAVY: return "GEMM_HEAVY";
            case LayerComputePattern::MEMORY_BOUND: return "MEMORY_BOUND";
            case LayerComputePattern::ATTENTION_PATTERN: return "ATTENTION_PATTERN";
            case LayerComputePattern::EMBEDDING_LOOKUP: return "EMBEDDING_LOOKUP";
            case LayerComputePattern::MIXED: return "MIXED";
            default: return "UNKNOWN";
        }
    }
};

// Convenience macros for orchestration logging
#define ORCH_LOG_INFO(fmt, ...) \
    LOG_CAT(static_cast<common_log_category>(LOG_CAT_ORCHESTRATION), GGML_LOG_LEVEL_INFO, fmt, ##__VA_ARGS__)

#define ORCH_LOG_PERF(fmt, ...) \
    LOG_CAT(static_cast<common_log_category>(LOG_CAT_PROFILING), GGML_LOG_LEVEL_INFO, fmt, ##__VA_ARGS__)

#define ORCH_LOG_TRACE(fmt, ...) \
    LOG_TRACE(fmt, ##__VA_ARGS__)

// RAII timer for orchestration operations
class OrchestrationTimer {
public:
    OrchestrationTimer(const std::string& operation_name)
        : name_(operation_name), start_(std::chrono::high_resolution_clock::now()) {
        LOG_PERF_START(PERF_EVENT_CUSTOM, name_);
    }
    
    ~OrchestrationTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        LOG_PERF_END(PERF_EVENT_CUSTOM, name_);
        ORCH_LOG_TRACE("[Timer] %s completed in %ld us\n", name_.c_str(), duration.count());
    }
    
private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
};

} // namespace orchestration
} // namespace llama