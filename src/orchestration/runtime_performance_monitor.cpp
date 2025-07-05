#include "runtime_performance_monitor.h"
#include "gpu_architecture_analyzer.h"
#include "llama-impl.h"
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace llama {
namespace orchestration {

// LayerPerformanceStats implementation
void LayerPerformanceStats::update(const LayerPerformanceSample& sample) {
    sample_count++;
    total_execution_time += sample.execution_time;
    min_execution_time = std::min(min_execution_time, sample.execution_time);
    max_execution_time = std::max(max_execution_time, sample.execution_time);
    avg_execution_time = total_execution_time / sample_count;
    
    // Update GPU utilization average
    avg_gpu_utilization = ((avg_gpu_utilization * (sample_count - 1)) + 
                          sample.gpu_utilization) / sample_count;
    
    // Update memory bandwidth average
    avg_memory_bandwidth_utilization = 
        ((avg_memory_bandwidth_utilization * (sample_count - 1)) + 
         sample.memory_bandwidth_utilization) / sample_count;
}

bool LayerPerformanceStats::is_degrading() const {
    // Performance is degrading if recent average is significantly worse than overall
    return moving_avg_5.count() > avg_execution_time.count() * 1.5;  // 1.5x threshold
}

// RuntimePerformanceMonitor implementation
RuntimePerformanceMonitor::RuntimePerformanceMonitor() = default;

RuntimePerformanceMonitor::~RuntimePerformanceMonitor() {
    stop_monitoring();
}

void RuntimePerformanceMonitor::initialize(const std::vector<GpuProfile>& gpu_profiles,
                                         const DistributionResult& distribution) {
    gpu_profiles_ = gpu_profiles;
    current_distribution_ = distribution;
    
    // Initialize GPU status tracking
    gpu_status_.resize(gpu_profiles.size());
    for (size_t i = 0; i < gpu_profiles.size(); ++i) {
        gpu_status_[i].gpu_id = i;
        gpu_status_[i].base_clock_mhz = gpu_profiles[i].max_clock_mhz;
        gpu_status_[i].available_memory_bytes = gpu_profiles[i].available_memory_bytes;
    }
    
    // Initialize layer statistics
    for (const auto& assignment : distribution.assignments) {
        layer_stats_[assignment.layer_index] = LayerPerformanceStats();
    }
    
    ORCH_LOG_INFO("[Runtime Monitor] Initialized with %zu GPUs and %zu layer assignments\n",
                  gpu_profiles.size(), distribution.assignments.size());
}

void RuntimePerformanceMonitor::start_monitoring() {
    if (monitoring_active_.exchange(true)) {
        return;  // Already monitoring
    }
    
    // Start GPU monitoring thread
    gpu_monitor_thread_ = std::thread(&RuntimePerformanceMonitor::gpu_monitor_loop, this);
    
    // Start anomaly detection thread
    anomaly_monitor_thread_ = std::thread(&RuntimePerformanceMonitor::anomaly_monitor_loop, this);
    
    ORCH_LOG_INFO("[Runtime Monitor] Started monitoring\n");
}

void RuntimePerformanceMonitor::stop_monitoring() {
    if (!monitoring_active_.exchange(false)) {
        return;  // Not monitoring
    }
    
    // Stop threads
    if (gpu_monitor_thread_.joinable()) {
        gpu_monitor_thread_.join();
    }
    if (anomaly_monitor_thread_.joinable()) {
        anomaly_monitor_thread_.join();
    }
    
    // Log final summary
    log_performance_summary();
    
    ORCH_LOG_INFO("[Runtime Monitor] Stopped monitoring\n");
}

void RuntimePerformanceMonitor::record_layer_start(int layer_id, int gpu_id) {
    std::lock_guard<std::mutex> lock(active_layers_mutex_);
    
    ActiveLayerExecution execution;
    execution.layer_id = layer_id;
    execution.gpu_id = gpu_id;
    execution.start_time = std::chrono::steady_clock::now();
    
    active_layers_[layer_id] = execution;
    
    ORCH_LOG_TRACE("[Runtime Monitor] Layer %d started on GPU %d\n", layer_id, gpu_id);
}

void RuntimePerformanceMonitor::record_layer_end(int layer_id, int gpu_id) {
    auto end_time = std::chrono::steady_clock::now();
    
    std::unique_lock<std::mutex> active_lock(active_layers_mutex_);
    auto it = active_layers_.find(layer_id);
    if (it == active_layers_.end()) {
        ORCH_LOG_TRACE("[Runtime Monitor] Warning: Layer %d end without start\n", layer_id);
        return;
    }
    
    auto execution = it->second;
    active_layers_.erase(it);
    active_lock.unlock();
    
    // Calculate execution time
    auto execution_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - execution.start_time);
    
    // Create performance sample
    LayerPerformanceSample sample;
    sample.layer_id = layer_id;
    sample.gpu_id = gpu_id;
    sample.execution_time = execution_time;
    sample.timestamp = end_time;
    
    // Get current GPU status
    {
        std::lock_guard<std::mutex> gpu_lock(gpu_status_mutex_);
        if (gpu_id < static_cast<int>(gpu_status_.size())) {
            sample.gpu_utilization = gpu_status_[gpu_id].current_utilization;
            sample.memory_bandwidth_utilization = 0.0f;  // Would be updated by memory tracking
        }
    }
    
    // Update statistics
    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        
        // Update layer stats
        layer_stats_[layer_id].update(sample);
        
        // Keep sample history (last 100 samples per layer)
        auto& samples = layer_samples_[layer_id];
        samples.push_back(sample);
        if (samples.size() > 100) {
            samples.pop_front();
        }
        
        // Update moving averages
        if (samples.size() >= 5) {
            std::chrono::microseconds sum5{0};
            auto it = samples.rbegin();
            for (int i = 0; i < 5 && it != samples.rend(); ++i, ++it) {
                sum5 += it->execution_time;
            }
            layer_stats_[layer_id].moving_avg_5 = sum5 / 5;
        }
        
        if (samples.size() >= 20) {
            std::chrono::microseconds sum20{0};
            auto it = samples.rbegin();
            for (int i = 0; i < 20 && it != samples.rend(); ++i, ++it) {
                sum20 += it->execution_time;
            }
            layer_stats_[layer_id].moving_avg_20 = sum20 / 20;
        }
    }
    
    // Update metrics
    update_metrics(sample);
    
    // Check for slow execution
    detect_slow_execution(layer_id, sample);
    
    // Add trace event if enabled
    if (config_.enable_trace_export) {
        std::lock_guard<std::mutex> trace_lock(trace_mutex_);
        TraceEvent event;
        event.name = "layer_" + std::to_string(layer_id);
        event.category = "compute";
        event.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            execution.start_time.time_since_epoch());
        event.duration = execution_time;
        event.args["gpu_id"] = std::to_string(gpu_id);
        event.args["utilization"] = std::to_string(sample.gpu_utilization);
        trace_events_.push_back(event);
    }
    
    ORCH_LOG_TRACE("[Runtime Monitor] Layer %d completed on GPU %d in %ld us\n",
                   layer_id, gpu_id, execution_time.count());
}

void RuntimePerformanceMonitor::record_memory_transfer(int src_gpu, int dst_gpu, 
                                                     size_t bytes,
                                                     std::chrono::microseconds duration) {
    // Update cross-GPU transfer metrics
    TRACK_MEMORY_TRANSFER(src_gpu, dst_gpu, bytes, duration);
    
    // Log the transfer
    OrchestrationLogger::instance().log_memory_transfer(src_gpu, dst_gpu, bytes, duration);
    
    // Add trace event if enabled
    if (config_.enable_trace_export) {
        std::lock_guard<std::mutex> trace_lock(trace_mutex_);
        TraceEvent event;
        event.name = "memory_transfer";
        event.category = "memory";
        event.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch());
        event.duration = duration;
        event.args["src_gpu"] = std::to_string(src_gpu);
        event.args["dst_gpu"] = std::to_string(dst_gpu);
        event.args["bytes"] = std::to_string(bytes);
        trace_events_.push_back(event);
    }
}

void RuntimePerformanceMonitor::record_queue_wait(int layer_id, 
                                                 std::chrono::microseconds wait_time) {
    // Update queue wait metrics
    auto& metrics = AdvancedLayerMetrics::instance();
    metrics.layer_queue_wait_us->observe(wait_time.count());
    
    // Check for high queue wait
    if (wait_time > std::chrono::milliseconds(10)) {  // 10ms threshold
        PerformanceAnomaly anomaly;
        anomaly.type = PerformanceAnomalyType::HIGH_QUEUE_WAIT;
        anomaly.affected_layer_id = layer_id;
        anomaly.severity = std::min(1.0f, wait_time.count() / 50000.0f);  // 50ms = severity 1.0
        anomaly.description = "High queue wait time for layer " + std::to_string(layer_id);
        anomaly.timestamp = std::chrono::steady_clock::now();
        anomaly.metrics["wait_time_us"] = static_cast<float>(wait_time.count());
        
        add_anomaly(anomaly);
    }
}

LayerPerformanceStats RuntimePerformanceMonitor::get_layer_stats(int layer_id) const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    auto it = layer_stats_.find(layer_id);
    return it != layer_stats_.end() ? it->second : LayerPerformanceStats();
}

GpuRuntimeStatus RuntimePerformanceMonitor::get_gpu_status(int gpu_id) const {
    std::lock_guard<std::mutex> lock(gpu_status_mutex_);
    if (gpu_id >= 0 && gpu_id < static_cast<int>(gpu_status_.size())) {
        return gpu_status_[gpu_id];
    }
    return GpuRuntimeStatus();
}

std::vector<PerformanceAnomaly> RuntimePerformanceMonitor::get_recent_anomalies(
    size_t count) const {
    std::lock_guard<std::mutex> lock(anomaly_mutex_);
    std::vector<PerformanceAnomaly> result;
    
    auto it = anomaly_history_.rbegin();
    for (size_t i = 0; i < count && it != anomaly_history_.rend(); ++i, ++it) {
        result.push_back(*it);
    }
    
    return result;
}

float RuntimePerformanceMonitor::calculate_current_speedup() const {
    // Calculate speedup based on current performance vs baseline
    float total_compute_time = 0.0f;
    float baseline_compute_time = 0.0f;
    
    std::lock_guard<std::mutex> lock(stats_mutex_);
    for (const auto& [layer_id, stats] : layer_stats_) {
        if (stats.sample_count > 0) {
            total_compute_time += stats.avg_execution_time.count();
            // Assume baseline is equal distribution
            baseline_compute_time += stats.avg_execution_time.count() * 1.2f;  // 20% overhead
        }
    }
    
    return baseline_compute_time > 0 ? baseline_compute_time / total_compute_time : 1.0f;
}

void RuntimePerformanceMonitor::gpu_monitor_loop() {
    while (monitoring_active_.load()) {
        for (size_t i = 0; i < gpu_profiles_.size(); ++i) {
            update_gpu_status(i);
        }
        
        std::this_thread::sleep_for(config_.gpu_status_update_interval);
    }
}

void RuntimePerformanceMonitor::anomaly_monitor_loop() {
    while (monitoring_active_.load()) {
        check_for_anomalies();
        std::this_thread::sleep_for(config_.anomaly_check_interval);
    }
}

void RuntimePerformanceMonitor::update_gpu_status(int gpu_id) {
    std::lock_guard<std::mutex> lock(gpu_status_mutex_);
    
    auto& status = gpu_status_[gpu_id];
    status.last_update = std::chrono::steady_clock::now();
    
    // In a real implementation, these would query actual GPU metrics
    // For now, we'll use placeholder values
    
    // Simulate GPU metrics (would be replaced with actual GPU queries)
    auto& metrics = AdvancedLayerMetrics::instance();
    status.current_utilization = metrics.gpu_utilization_percent[gpu_id]->value();
    status.current_temperature = 65.0f + (rand() % 20);  // 65-85°C
    status.current_power_watts = status.current_utilization * 3.0f;  // Rough estimate
    status.current_clock_mhz = status.base_clock_mhz * (status.is_throttling ? 0.8f : 1.0f);
    
    // Update metrics
    metrics.gpu_temperature_celsius[gpu_id]->set(status.current_temperature);
    metrics.gpu_power_watts[gpu_id]->set(status.current_power_watts);
    
    // Check for thermal throttling
    detect_thermal_throttle(gpu_id);
    
    // Check for memory pressure
    detect_memory_pressure(gpu_id);
}

void RuntimePerformanceMonitor::detect_thermal_throttle(int gpu_id) {
    const auto& status = gpu_status_[gpu_id];
    
    bool was_throttling = status.is_throttling;
    bool is_throttling = status.current_temperature > config_.thermal_throttle_threshold ||
                        status.current_clock_mhz < status.base_clock_mhz * 0.9f;
    
    if (is_throttling && !was_throttling) {
        // New throttling event
        gpu_status_[gpu_id].is_throttling = true;
        
        PerformanceAnomaly anomaly;
        anomaly.type = PerformanceAnomalyType::THERMAL_THROTTLE;
        anomaly.affected_gpu_id = gpu_id;
        anomaly.severity = (status.current_temperature - config_.thermal_throttle_threshold) / 10.0f;
        anomaly.description = "GPU " + std::to_string(gpu_id) + " thermal throttling";
        anomaly.timestamp = std::chrono::steady_clock::now();
        anomaly.metrics["temperature"] = status.current_temperature;
        anomaly.metrics["clock_reduction"] = 1.0f - (status.current_clock_mhz / status.base_clock_mhz);
        
        add_anomaly(anomaly);
        
        // Update metrics
        AdvancedLayerMetrics::instance().thermal_throttle_events->increment();
        
        // Log the event
        OrchestrationLogger::instance().log_thermal_throttle(
            gpu_id, status.current_temperature, 
            status.current_clock_mhz / status.base_clock_mhz);
    } else if (!is_throttling && was_throttling) {
        gpu_status_[gpu_id].is_throttling = false;
        ORCH_LOG_INFO("[Runtime Monitor] GPU %d thermal throttling resolved\n", gpu_id);
    }
}

void RuntimePerformanceMonitor::detect_memory_pressure(int gpu_id) {
    const auto& status = gpu_status_[gpu_id];
    float memory_usage = static_cast<float>(status.used_memory_bytes) / 
                        status.available_memory_bytes;
    
    if (memory_usage > config_.memory_pressure_threshold) {
        PerformanceAnomaly anomaly;
        anomaly.type = PerformanceAnomalyType::MEMORY_PRESSURE;
        anomaly.affected_gpu_id = gpu_id;
        anomaly.severity = (memory_usage - config_.memory_pressure_threshold) / 
                          (1.0f - config_.memory_pressure_threshold);
        anomaly.description = "GPU " + std::to_string(gpu_id) + " memory pressure";
        anomaly.timestamp = std::chrono::steady_clock::now();
        anomaly.metrics["memory_usage_percent"] = memory_usage * 100.0f;
        anomaly.metrics["available_mb"] = (status.available_memory_bytes - 
                                          status.used_memory_bytes) / (1024.0f * 1024.0f);
        
        add_anomaly(anomaly);
    }
}

void RuntimePerformanceMonitor::check_for_anomalies() {
    // Check for load imbalance
    detect_load_imbalance();
    
    // Check for overall performance degradation
    float current_speedup = calculate_current_speedup();
    if (current_speedup < current_distribution_.expected_speedup * 0.8f) {
        PerformanceAnomaly anomaly;
        anomaly.type = PerformanceAnomalyType::SLOW_EXECUTION;
        anomaly.affected_gpu_id = -1;  // System-wide
        anomaly.severity = 1.0f - (current_speedup / current_distribution_.expected_speedup);
        anomaly.description = "Overall performance degradation detected";
        anomaly.timestamp = std::chrono::steady_clock::now();
        anomaly.metrics["current_speedup"] = current_speedup;
        anomaly.metrics["expected_speedup"] = current_distribution_.expected_speedup;
        
        add_anomaly(anomaly);
    }
    
    // Update distribution quality metrics
    auto gpu_loads = get_gpu_load_distribution();
    std::vector<size_t> gpu_memory;
    for (const auto& status : gpu_status_) {
        gpu_memory.push_back(status.used_memory_bytes);
    }
    
    UPDATE_DISTRIBUTION_QUALITY(gpu_loads, gpu_memory, 
                               current_distribution_.expected_speedup, 
                               current_speedup);
}

void RuntimePerformanceMonitor::detect_load_imbalance() {
    auto gpu_loads = get_gpu_load_distribution();
    if (gpu_loads.empty()) return;
    
    float avg_load = std::accumulate(gpu_loads.begin(), gpu_loads.end(), 0.0f) / 
                    gpu_loads.size();
    float max_deviation = 0.0f;
    int most_loaded_gpu = -1;
    
    for (size_t i = 0; i < gpu_loads.size(); ++i) {
        float deviation = std::abs(gpu_loads[i] - avg_load) / avg_load;
        if (deviation > max_deviation) {
            max_deviation = deviation;
            most_loaded_gpu = i;
        }
    }
    
    if (max_deviation > config_.load_imbalance_threshold) {
        PerformanceAnomaly anomaly;
        anomaly.type = PerformanceAnomalyType::LOAD_IMBALANCE;
        anomaly.affected_gpu_id = most_loaded_gpu;
        anomaly.severity = max_deviation / 0.5f;  // 50% deviation = severity 1.0
        anomaly.description = "Significant load imbalance detected";
        anomaly.timestamp = std::chrono::steady_clock::now();
        anomaly.metrics["max_deviation"] = max_deviation;
        anomaly.metrics["avg_load"] = avg_load;
        
        add_anomaly(anomaly);
    }
}

void RuntimePerformanceMonitor::add_anomaly(const PerformanceAnomaly& anomaly) {
    {
        std::lock_guard<std::mutex> lock(anomaly_mutex_);
        anomaly_history_.push_back(anomaly);
        
        // Keep only recent anomalies (last 1000)
        if (anomaly_history_.size() > 1000) {
            anomaly_history_.pop_front();
        }
    }
    
    // Update metrics
    AdvancedLayerMetrics::instance().performance_anomaly_count->increment();
    
    // Log the anomaly
    OrchestrationLogger::instance().log_performance_anomaly(
        anomaly.affected_gpu_id, anomaly.description, anomaly.severity);
    
    // Notify callback if set
    if (anomaly_callback_) {
        anomaly_callback_(anomaly);
    }
}

void RuntimePerformanceMonitor::update_metrics(const LayerPerformanceSample& sample) {
    auto& metrics = AdvancedLayerMetrics::instance();
    
    // Update layer compute time histogram
    if (auto* hist = metrics.get_layer_compute_histogram(sample.layer_id)) {
        hist->observe(sample.execution_time.count());
    }
    
    // Track the execution
    TRACK_LAYER_EXECUTION(sample.layer_id, sample.gpu_id);
    
    // Check for anomalies
    CHECK_PERFORMANCE_ANOMALY(sample.layer_id, sample.gpu_id, sample.execution_time);
}

std::vector<float> RuntimePerformanceMonitor::get_gpu_load_distribution() const {
    std::vector<float> loads(gpu_profiles_.size(), 0.0f);
    std::vector<float> total_time(gpu_profiles_.size(), 0.0f);
    
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    // Calculate total execution time per GPU
    for (const auto& assignment : current_distribution_.assignments) {
        auto it = layer_stats_.find(assignment.layer_index);
        if (it != layer_stats_.end() && it->second.sample_count > 0) {
            total_time[assignment.gpu_id] += it->second.avg_execution_time.count();
        }
    }
    
    // Normalize to percentages
    float max_time = *std::max_element(total_time.begin(), total_time.end());
    if (max_time > 0) {
        for (size_t i = 0; i < loads.size(); ++i) {
            loads[i] = (total_time[i] / max_time) * 100.0f;
        }
    }
    
    return loads;
}

void RuntimePerformanceMonitor::log_performance_summary() const {
    ORCH_LOG_INFO("========== Runtime Performance Summary ==========\n");
    
    // Overall performance
    float speedup = calculate_current_speedup();
    ORCH_LOG_INFO("Current speedup: %.2fx (expected: %.2fx)\n",
                  speedup, current_distribution_.expected_speedup);
    
    // GPU load distribution
    auto loads = get_gpu_load_distribution();
    ORCH_LOG_INFO("\nGPU Load Distribution:\n");
    for (size_t i = 0; i < loads.size(); ++i) {
        ORCH_LOG_INFO("  GPU %zu: %.1f%%\n", i, loads[i]);
    }
    
    // Layer performance statistics
    ORCH_LOG_INFO("\nLayer Performance (top 10 slowest):\n");
    std::vector<std::pair<int, std::chrono::microseconds>> layer_times;
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        for (const auto& [layer_id, stats] : layer_stats_) {
            if (stats.sample_count > 0) {
                layer_times.push_back({layer_id, stats.avg_execution_time});
            }
        }
    }
    
    std::sort(layer_times.begin(), layer_times.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    for (size_t i = 0; i < std::min(size_t(10), layer_times.size()); ++i) {
        ORCH_LOG_INFO("  Layer %d: %.2f ms avg\n",
                      layer_times[i].first, 
                      layer_times[i].second.count() / 1000.0f);
    }
    
    // Anomaly summary
    ORCH_LOG_INFO("\nPerformance Anomalies:\n");
    auto anomalies = get_recent_anomalies(5);
    for (const auto& anomaly : anomalies) {
        ORCH_LOG_INFO("  %s (severity: %.2f)\n",
                      anomaly.description.c_str(), anomaly.severity);
    }
    
    // Advanced metrics summary
    log_advanced_metrics_summary();
    
    ORCH_LOG_INFO("===============================================\n");
}

} // namespace orchestration
} // namespace llama