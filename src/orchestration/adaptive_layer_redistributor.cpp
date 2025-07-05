#include "adaptive_layer_redistributor.h"
#include "llama-impl.h"
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <limits>

namespace llama {
namespace orchestration {

AdaptiveLayerRedistributor::AdaptiveLayerRedistributor() = default;

AdaptiveLayerRedistributor::~AdaptiveLayerRedistributor() {
    disable_adaptive_mode();
}

void AdaptiveLayerRedistributor::initialize(RuntimePerformanceMonitor* monitor,
                                          LayerDistributor* distributor,
                                          GpuProfiler* profiler) {
    performance_monitor_ = monitor;
    layer_distributor_ = distributor;
    gpu_profiler_ = profiler;
    
    // Get initial state
    auto profiling_result = gpu_profiler_->profile_all_gpus();
    current_gpu_profiles_ = profiling_result.gpu_profiles;
    
    ORCH_LOG_INFO("[Adaptive Redistributor] Initialized with %zu GPUs\n",
                  current_gpu_profiles_.size());
}

void AdaptiveLayerRedistributor::enable_adaptive_mode() {
    if (adaptive_mode_enabled_.exchange(true)) {
        return;  // Already enabled
    }
    
    if (!performance_monitor_ || !layer_distributor_ || !gpu_profiler_) {
        ORCH_LOG_INFO("[Adaptive Redistributor] Cannot enable: components not initialized\n");
        adaptive_mode_enabled_ = false;
        return;
    }
    
    monitoring_active_ = true;
    monitor_thread_ = std::thread(&AdaptiveLayerRedistributor::monitor_loop, this);
    
    ORCH_LOG_INFO("[Adaptive Redistributor] Adaptive mode enabled\n");
    OrchestrationLogger::instance().log_redistribution_triggered("Adaptive mode enabled");
}

void AdaptiveLayerRedistributor::disable_adaptive_mode() {
    if (!adaptive_mode_enabled_.exchange(false)) {
        return;  // Already disabled
    }
    
    monitoring_active_ = false;
    monitor_cv_.notify_all();
    
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
    
    ORCH_LOG_INFO("[Adaptive Redistributor] Adaptive mode disabled\n");
}

RedistributionResult AdaptiveLayerRedistributor::trigger_redistribution(
    const RedistributionRequest& request) {
    
    if (redistributing_.exchange(true)) {
        RedistributionResult result;
        result.success = false;
        result.error_message = "Redistribution already in progress";
        return result;
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Check if safe to redistribute
    if (!is_safe_to_redistribute()) {
        redistributing_ = false;
        RedistributionResult result;
        result.success = false;
        result.error_message = "Not safe to redistribute at this time";
        return result;
    }
    
    // Create redistribution plan
    auto plan = create_redistribution_plan(request);
    
    if (!plan.is_valid()) {
        redistributing_ = false;
        RedistributionResult result;
        result.success = false;
        result.error_message = "Could not create valid redistribution plan";
        return result;
    }
    
    // Log the plan
    log_redistribution_plan(plan);
    
    // Execute redistribution
    auto result = execute_redistribution(plan);
    
    // Update timing
    auto end_time = std::chrono::steady_clock::now();
    result.duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    
    // Update statistics
    update_stability_metrics(result);
    
    // Log result
    log_redistribution_result(result);
    OrchestrationLogger::instance().log_redistribution_completed(
        result.success, result.layers_moved, result.duration);
    
    // Update metrics
    if (result.success) {
        AdvancedLayerMetrics::instance().layer_migration_count->increment(result.layers_moved);
    }
    
    // Notify callback
    if (redistribution_callback_) {
        redistribution_callback_(result);
    }
    
    redistributing_ = false;
    last_redistribution_time_ = end_time;
    
    return result;
}

void AdaptiveLayerRedistributor::monitor_loop() {
    while (monitoring_active_.load()) {
        std::unique_lock<std::mutex> lock(monitor_mutex_);
        
        // Wait for next check interval or shutdown
        monitor_cv_.wait_for(lock, std::chrono::seconds(1), 
                           [this] { return !monitoring_active_.load(); });
        
        if (!monitoring_active_.load()) {
            break;
        }
        
        // Check various triggers
        check_performance_triggers();
        check_thermal_triggers();
        check_memory_triggers();
        
        if (config_.enable_predictive_redistribution) {
            check_predictive_triggers();
        }
        
        // Process pending requests
        while (!pending_requests_.empty()) {
            RedistributionRequest request;
            {
                std::lock_guard<std::mutex> req_lock(request_mutex_);
                if (!pending_requests_.empty()) {
                    request = pending_requests_.front();
                    pending_requests_.pop();
                } else {
                    break;
                }
            }
            
            // Execute redistribution
            trigger_redistribution(request);
        }
    }
}

void AdaptiveLayerRedistributor::check_performance_triggers() {
    if (!performance_monitor_ || !performance_monitor_->is_monitoring()) {
        return;
    }
    
    // Get current performance metrics
    float current_speedup = performance_monitor_->calculate_current_speedup();
    float expected_speedup = current_distribution_.expected_speedup;
    
    float degradation = 1.0f - (current_speedup / expected_speedup);
    
    if (degradation > config_.performance_degradation_threshold) {
        // Identify problematic GPUs
        auto gpu_loads = performance_monitor_->get_gpu_load_distribution();
        auto overloaded = identify_overloaded_gpus();
        
        if (!overloaded.empty()) {
            RedistributionRequest request;
            request.type = RedistributionRequest::PERFORMANCE_TRIGGERED;
            request.reason = "Performance degradation detected (degradation: " + 
                           std::to_string(degradation * 100) + "%)";
            request.urgency = std::min(1.0f, degradation / 0.5f);
            request.affected_gpus = overloaded;
            request.timestamp = std::chrono::steady_clock::now();
            
            std::lock_guard<std::mutex> lock(request_mutex_);
            pending_requests_.push(request);
        }
    }
}

void AdaptiveLayerRedistributor::check_thermal_triggers() {
    // Check for thermal throttling
    for (size_t i = 0; i < current_gpu_profiles_.size(); ++i) {
        auto status = performance_monitor_->get_gpu_status(i);
        
        if (status.is_throttling) {
            // Check severity
            float severity = (status.current_temperature - config_.thermal_severity_threshold * 100.0f) / 20.0f;
            severity = std::min(1.0f, std::max(0.0f, severity));
            
            if (severity > config_.thermal_severity_threshold) {
                RedistributionRequest request;
                request.type = RedistributionRequest::THERMAL_TRIGGERED;
                request.reason = "GPU " + std::to_string(i) + " thermal throttling";
                request.urgency = severity;
                request.affected_gpus = {static_cast<int>(i)};
                request.timestamp = std::chrono::steady_clock::now();
                
                std::lock_guard<std::mutex> lock(request_mutex_);
                pending_requests_.push(request);
            }
        }
    }
}

void AdaptiveLayerRedistributor::check_memory_triggers() {
    // Check for memory pressure
    for (size_t i = 0; i < current_gpu_profiles_.size(); ++i) {
        auto status = performance_monitor_->get_gpu_status(i);
        
        float memory_usage = static_cast<float>(status.used_memory_bytes) / 
                           status.available_memory_bytes;
        
        if (memory_usage > config_.memory_pressure_threshold) {
            RedistributionRequest request;
            request.type = RedistributionRequest::MEMORY_TRIGGERED;
            request.reason = "GPU " + std::to_string(i) + " memory pressure (" + 
                           std::to_string(memory_usage * 100) + "% used)";
            request.urgency = (memory_usage - config_.memory_pressure_threshold) / 
                             (1.0f - config_.memory_pressure_threshold);
            request.affected_gpus = {static_cast<int>(i)};
            request.timestamp = std::chrono::steady_clock::now();
            
            std::lock_guard<std::mutex> lock(request_mutex_);
            pending_requests_.push(request);
        }
    }
}

void AdaptiveLayerRedistributor::check_predictive_triggers() {
    // Look for performance degradation trends
    bool degradation_trend = false;
    
    for (const auto& assignment : current_distribution_.assignments) {
        auto stats = performance_monitor_->get_layer_stats(assignment.layer_index);
        if (stats.is_degrading()) {
            degradation_trend = true;
            break;
        }
    }
    
    if (degradation_trend) {
        // Predict future issues and redistribute proactively
        auto overloaded = identify_overloaded_gpus();
        auto underutilized = identify_underutilized_gpus();
        
        if (!overloaded.empty() && !underutilized.empty()) {
            RedistributionRequest request;
            request.type = RedistributionRequest::PERFORMANCE_TRIGGERED;
            request.reason = "Predictive redistribution (performance trend detected)";
            request.urgency = 0.3f;  // Lower urgency for predictive
            request.affected_gpus = overloaded;
            request.timestamp = std::chrono::steady_clock::now();
            
            std::lock_guard<std::mutex> lock(request_mutex_);
            pending_requests_.push(request);
        }
    }
}

RedistributionPlan AdaptiveLayerRedistributor::create_redistribution_plan(
    const RedistributionRequest& request) {
    
    RedistributionPlan plan;
    
    switch (request.type) {
        case RedistributionRequest::PERFORMANCE_TRIGGERED:
            plan = plan_performance_redistribution(request.affected_gpus);
            break;
            
        case RedistributionRequest::THERMAL_TRIGGERED:
            if (!request.affected_gpus.empty()) {
                plan = plan_thermal_redistribution(request.affected_gpus[0]);
            }
            break;
            
        case RedistributionRequest::MEMORY_TRIGGERED:
            if (!request.affected_gpus.empty()) {
                plan = plan_memory_redistribution(request.affected_gpus[0]);
            }
            break;
            
        case RedistributionRequest::MANUAL_TRIGGERED:
            // For manual, try performance redistribution
            plan = plan_performance_redistribution(request.affected_gpus);
            break;
    }
    
    // Optimize the plan
    if (!plan.moves.empty()) {
        optimize_plan(plan);
        plan.risk_score = evaluate_plan_risk(plan);
        plan.expected_overall_improvement = predict_plan_improvement(plan);
    }
    
    return plan;
}

RedistributionPlan AdaptiveLayerRedistributor::plan_performance_redistribution(
    const std::vector<int>& affected_gpus) {
    
    RedistributionPlan plan;
    plan.strategy_description = "Performance-based redistribution";
    
    // Get current load distribution
    auto gpu_loads = performance_monitor_->get_gpu_load_distribution();
    auto underutilized = identify_underutilized_gpus();
    
    // For each overloaded GPU, find layers to move
    for (int gpu_id : affected_gpus) {
        auto movable_layers = find_movable_layers(gpu_id);
        
        // Sort layers by execution time (move slowest first)
        std::sort(movable_layers.begin(), movable_layers.end(),
                  [this](int a, int b) {
                      auto stats_a = performance_monitor_->get_layer_stats(a);
                      auto stats_b = performance_monitor_->get_layer_stats(b);
                      return stats_a.avg_execution_time > stats_b.avg_execution_time;
                  });
        
        // Move layers to underutilized GPUs
        for (int layer_id : movable_layers) {
            if (plan.moves.size() >= static_cast<size_t>(config_.max_layers_per_redistribution)) {
                break;
            }
            
            int target_gpu = find_best_target_gpu(layer_id, underutilized);
            if (target_gpu >= 0 && target_gpu != gpu_id) {
                RedistributionPlan::LayerMove move;
                move.layer_id = layer_id;
                move.from_gpu = gpu_id;
                move.to_gpu = target_gpu;
                
                // Estimate memory size (would be from actual layer profile)
                move.memory_size = 100 * 1024 * 1024;  // 100MB placeholder
                
                // Estimate improvement
                float load_reduction = 1.0f / movable_layers.size();
                move.expected_improvement = load_reduction * 0.2f;  // 20% of load reduction
                
                plan.moves.push_back(move);
            }
        }
    }
    
    return plan;
}

RedistributionPlan AdaptiveLayerRedistributor::plan_thermal_redistribution(
    int throttled_gpu) {
    
    RedistributionPlan plan;
    plan.strategy_description = "Thermal relief redistribution for GPU " + 
                               std::to_string(throttled_gpu);
    
    // Find coolest GPUs
    std::vector<std::pair<int, float>> gpu_temps;
    for (size_t i = 0; i < current_gpu_profiles_.size(); ++i) {
        if (static_cast<int>(i) != throttled_gpu) {
            auto status = performance_monitor_->get_gpu_status(i);
            gpu_temps.push_back({i, status.current_temperature});
        }
    }
    
    // Sort by temperature (coolest first)
    std::sort(gpu_temps.begin(), gpu_temps.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    // Move most compute-intensive layers
    auto movable_layers = find_movable_layers(throttled_gpu);
    std::sort(movable_layers.begin(), movable_layers.end(),
              [this](int a, int b) {
                  auto stats_a = performance_monitor_->get_layer_stats(a);
                  auto stats_b = performance_monitor_->get_layer_stats(b);
                  return stats_a.avg_gpu_utilization > stats_b.avg_gpu_utilization;
              });
    
    // Move to coolest GPUs
    size_t gpu_idx = 0;
    for (int layer_id : movable_layers) {
        if (plan.moves.size() >= 5 || gpu_idx >= gpu_temps.size()) {  // Move at most 5 layers
            break;
        }
        
        RedistributionPlan::LayerMove move;
        move.layer_id = layer_id;
        move.from_gpu = throttled_gpu;
        move.to_gpu = gpu_temps[gpu_idx].first;
        move.memory_size = 100 * 1024 * 1024;  // Placeholder
        move.expected_improvement = 0.15f;  // 15% improvement expected
        
        plan.moves.push_back(move);
        gpu_idx = (gpu_idx + 1) % gpu_temps.size();  // Round-robin
    }
    
    return plan;
}

RedistributionPlan AdaptiveLayerRedistributor::plan_memory_redistribution(
    int pressured_gpu) {
    
    RedistributionPlan plan;
    plan.strategy_description = "Memory pressure relief for GPU " + 
                               std::to_string(pressured_gpu);
    
    // Find GPUs with available memory
    std::vector<std::pair<int, size_t>> gpu_memory;
    for (size_t i = 0; i < current_gpu_profiles_.size(); ++i) {
        if (static_cast<int>(i) != pressured_gpu) {
            auto status = performance_monitor_->get_gpu_status(i);
            size_t available = status.available_memory_bytes - status.used_memory_bytes;
            gpu_memory.push_back({i, available});
        }
    }
    
    // Sort by available memory (most available first)
    std::sort(gpu_memory.begin(), gpu_memory.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Move largest layers first
    auto movable_layers = find_movable_layers(pressured_gpu);
    
    // In real implementation, would sort by actual memory usage
    // For now, just take first few layers
    for (size_t i = 0; i < std::min(size_t(3), movable_layers.size()); ++i) {
        if (i < gpu_memory.size()) {
            RedistributionPlan::LayerMove move;
            move.layer_id = movable_layers[i];
            move.from_gpu = pressured_gpu;
            move.to_gpu = gpu_memory[i].first;
            move.memory_size = 200 * 1024 * 1024;  // 200MB placeholder
            move.expected_improvement = 0.1f;
            
            plan.moves.push_back(move);
        }
    }
    
    return plan;
}

void AdaptiveLayerRedistributor::optimize_plan(RedistributionPlan& plan) {
    if (config_.prefer_minimal_moves) {
        // Remove redundant moves
        std::unordered_map<int, int> layer_final_gpu;
        
        // Track final destination for each layer
        for (const auto& move : plan.moves) {
            layer_final_gpu[move.layer_id] = move.to_gpu;
        }
        
        // Keep only necessary moves
        std::vector<RedistributionPlan::LayerMove> optimized_moves;
        std::unordered_set<int> moved_layers;
        
        for (const auto& move : plan.moves) {
            if (moved_layers.find(move.layer_id) == moved_layers.end()) {
                // Update move to go directly to final destination
                auto final_move = move;
                final_move.to_gpu = layer_final_gpu[move.layer_id];
                optimized_moves.push_back(final_move);
                moved_layers.insert(move.layer_id);
            }
        }
        
        plan.moves = optimized_moves;
    }
    
    // Sort moves by expected improvement (highest first)
    std::sort(plan.moves.begin(), plan.moves.end(),
              [](const auto& a, const auto& b) {
                  return a.expected_improvement > b.expected_improvement;
              });
    
    // Limit to configured maximum
    if (plan.moves.size() > static_cast<size_t>(config_.max_layers_per_redistribution)) {
        plan.moves.resize(config_.max_layers_per_redistribution);
    }
}

float AdaptiveLayerRedistributor::evaluate_plan_risk(const RedistributionPlan& plan) {
    float risk = 0.0f;
    
    // Risk factors:
    // 1. Number of moves (more moves = higher risk)
    float move_risk = static_cast<float>(plan.moves.size()) / 
                     config_.max_layers_per_redistribution;
    risk += move_risk * 0.3f;
    
    // 2. Cross-node moves (if applicable)
    if (!config_.allow_cross_node_moves) {
        // Check if any moves are cross-node (simplified check)
        for (const auto& move : plan.moves) {
            if (std::abs(move.from_gpu - move.to_gpu) > 1) {
                risk += 0.2f;
                break;
            }
        }
    }
    
    // 3. Time since last redistribution
    auto time_since_last = std::chrono::steady_clock::now() - last_redistribution_time_;
    if (time_since_last < config_.min_time_between_redistributions) {
        float time_factor = 1.0f - (time_since_last.count() / 
                                   config_.min_time_between_redistributions.count());
        risk += time_factor * 0.3f;
    }
    
    // 4. Current system stability
    float stability = get_stability_score();
    risk += (1.0f - stability) * 0.2f;
    
    return std::min(1.0f, risk);
}

float AdaptiveLayerRedistributor::predict_plan_improvement(const RedistributionPlan& plan) {
    float total_improvement = 0.0f;
    
    for (const auto& move : plan.moves) {
        total_improvement += move.expected_improvement;
    }
    
    // Apply diminishing returns
    return total_improvement * (1.0f - 0.1f * plan.moves.size() / 10.0f);
}

RedistributionResult AdaptiveLayerRedistributor::execute_redistribution(
    const RedistributionPlan& plan) {
    
    RedistributionResult result;
    result.success = false;
    result.layers_moved = 0;
    
    // Prepare for redistribution
    if (!prepare_redistribution(plan)) {
        result.error_message = "Failed to prepare redistribution";
        return result;
    }
    
    // Execute moves
    for (size_t i = 0; i < plan.moves.size(); ++i) {
        const auto& move = plan.moves[i];
        
        if (migrate_layer(move)) {
            result.layers_moved++;
            result.affected_layers.push_back(move.layer_id);
        } else {
            // Rollback on failure
            rollback_redistribution(plan, i);
            result.error_message = "Failed to migrate layer " + 
                                 std::to_string(move.layer_id);
            return result;
        }
    }
    
    // Update distribution tracking
    {
        std::lock_guard<std::mutex> lock(model_state_mutex_);
        // In real implementation, would update current_distribution_
        // based on the moves executed
    }
    
    result.success = true;
    result.actual_improvement = calculate_actual_improvement();
    
    return result;
}

bool AdaptiveLayerRedistributor::prepare_redistribution(const RedistributionPlan& plan) {
    // In real implementation, would:
    // 1. Pause inference if needed
    // 2. Ensure sufficient memory on target GPUs
    // 3. Prepare data structures for migration
    
    ORCH_LOG_INFO("[Adaptive Redistributor] Preparing to move %zu layers\n", 
                  plan.moves.size());
    
    return true;
}

bool AdaptiveLayerRedistributor::migrate_layer(const RedistributionPlan::LayerMove& move) {
    // In real implementation, would:
    // 1. Copy layer data to target GPU
    // 2. Update layer routing tables
    // 3. Verify layer functionality on new GPU
    
    ORCH_LOG_INFO("[Adaptive Redistributor] Migrating layer %d from GPU %d to GPU %d\n",
                  move.layer_id, move.from_gpu, move.to_gpu);
    
    // Simulate migration time
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    return true;  // Placeholder success
}

void AdaptiveLayerRedistributor::rollback_redistribution(const RedistributionPlan& plan,
                                                        int completed_moves) {
    ORCH_LOG_INFO("[Adaptive Redistributor] Rolling back %d moves\n", completed_moves);
    
    // Reverse completed moves
    for (int i = completed_moves - 1; i >= 0; --i) {
        const auto& original_move = plan.moves[i];
        
        // Create reverse move
        RedistributionPlan::LayerMove reverse_move;
        reverse_move.layer_id = original_move.layer_id;
        reverse_move.from_gpu = original_move.to_gpu;
        reverse_move.to_gpu = original_move.from_gpu;
        reverse_move.memory_size = original_move.memory_size;
        
        migrate_layer(reverse_move);
    }
}

float AdaptiveLayerRedistributor::get_stability_score() const {
    std::lock_guard<std::mutex> lock(history_mutex_);
    
    if (redistribution_history_.empty()) {
        return 1.0f;  // No history = stable
    }
    
    // Calculate stability based on:
    // 1. Success rate of recent redistributions
    // 2. Time since last redistribution
    // 3. Frequency of redistributions
    
    int recent_count = 0;
    int recent_success = 0;
    
    auto now = std::chrono::steady_clock::now();
    for (const auto& result : redistribution_history_) {
        recent_count++;
        if (result.success) {
            recent_success++;
        }
    }
    
    float success_rate = recent_count > 0 ? 
                        static_cast<float>(recent_success) / recent_count : 1.0f;
    
    auto time_since_last = now - last_redistribution_time_;
    float time_factor = std::min(1.0f, 
                                static_cast<float>(time_since_last.count()) / 
                                static_cast<float>(5 * config_.min_time_between_redistributions.count()));
    
    return success_rate * 0.7f + time_factor * 0.3f;
}

bool AdaptiveLayerRedistributor::is_safe_to_redistribute() const {
    // Check safety conditions
    if (!check_redistribution_limits()) {
        return false;
    }
    
    // Check minimum time between redistributions
    auto time_since_last = std::chrono::steady_clock::now() - last_redistribution_time_;
    if (time_since_last < config_.min_time_between_redistributions) {
        return false;
    }
    
    // Check stability score
    if (get_stability_score() < 0.3f) {
        return false;
    }
    
    return true;
}

bool AdaptiveLayerRedistributor::check_redistribution_limits() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    // Check hourly limit
    auto now = std::chrono::steady_clock::now();
    int recent_redistributions = 0;
    
    for (const auto& result : redistribution_history_) {
        auto age = std::chrono::duration_cast<std::chrono::hours>(
            now - stats_.last_redistribution);
        if (age.count() < 1) {
            recent_redistributions++;
        }
    }
    
    return recent_redistributions < config_.max_redistributions_per_hour;
}

std::vector<int> AdaptiveLayerRedistributor::identify_overloaded_gpus() const {
    std::vector<int> overloaded;
    
    auto gpu_loads = performance_monitor_->get_gpu_load_distribution();
    float avg_load = std::accumulate(gpu_loads.begin(), gpu_loads.end(), 0.0f) / 
                    gpu_loads.size();
    
    for (size_t i = 0; i < gpu_loads.size(); ++i) {
        if (gpu_loads[i] > avg_load * (1.0f + config_.load_imbalance_threshold)) {
            overloaded.push_back(i);
        }
    }
    
    return overloaded;
}

std::vector<int> AdaptiveLayerRedistributor::identify_underutilized_gpus() const {
    std::vector<int> underutilized;
    
    auto gpu_loads = performance_monitor_->get_gpu_load_distribution();
    float avg_load = std::accumulate(gpu_loads.begin(), gpu_loads.end(), 0.0f) / 
                    gpu_loads.size();
    
    for (size_t i = 0; i < gpu_loads.size(); ++i) {
        if (gpu_loads[i] < avg_load * (1.0f - config_.load_imbalance_threshold)) {
            underutilized.push_back(i);
        }
    }
    
    return underutilized;
}

std::vector<int> AdaptiveLayerRedistributor::find_movable_layers(int from_gpu) const {
    std::vector<int> movable;
    
    // Find layers assigned to this GPU
    for (const auto& assignment : current_distribution_.assignments) {
        if (assignment.gpu_id == from_gpu) {
            // In real implementation, would check if layer can be moved
            // (e.g., not actively processing, not pinned to GPU)
            movable.push_back(assignment.layer_index);
        }
    }
    
    return movable;
}

int AdaptiveLayerRedistributor::find_best_target_gpu(int layer_id,
                                                    const std::vector<int>& candidates) const {
    if (candidates.empty()) {
        return -1;
    }
    
    // In real implementation, would consider:
    // 1. Architecture affinity for the layer
    // 2. Available memory
    // 3. Current load
    // 4. Communication cost
    
    // For now, return least loaded candidate
    int best_gpu = -1;
    float min_load = std::numeric_limits<float>::max();
    
    auto gpu_loads = performance_monitor_->get_gpu_load_distribution();
    
    for (int gpu_id : candidates) {
        if (gpu_id < static_cast<int>(gpu_loads.size()) && gpu_loads[gpu_id] < min_load) {
            min_load = gpu_loads[gpu_id];
            best_gpu = gpu_id;
        }
    }
    
    return best_gpu;
}

void AdaptiveLayerRedistributor::log_redistribution_plan(const RedistributionPlan& plan) const {
    ORCH_LOG_INFO("[Redistribution Plan] %s\n", plan.strategy_description.c_str());
    ORCH_LOG_INFO("  Moves: %zu layers\n", plan.moves.size());
    ORCH_LOG_INFO("  Expected improvement: %.1f%%\n", 
                  plan.expected_overall_improvement * 100.0f);
    ORCH_LOG_INFO("  Risk score: %.2f\n", plan.risk_score);
    
    for (const auto& move : plan.moves) {
        ORCH_LOG_INFO("  - Layer %d: GPU %d -> GPU %d (%.1f%% improvement)\n",
                      move.layer_id, move.from_gpu, move.to_gpu,
                      move.expected_improvement * 100.0f);
    }
}

void AdaptiveLayerRedistributor::log_redistribution_result(
    const RedistributionResult& result) const {
    
    if (result.success) {
        ORCH_LOG_INFO("[Redistribution Result] Success\n");
        ORCH_LOG_INFO("  Layers moved: %d\n", result.layers_moved);
        ORCH_LOG_INFO("  Duration: %.2f ms\n", result.duration.count() / 1000.0f);
        ORCH_LOG_INFO("  Actual improvement: %.1f%%\n", 
                      result.actual_improvement * 100.0f);
    } else {
        ORCH_LOG_INFO("[Redistribution Result] Failed\n");
        ORCH_LOG_INFO("  Error: %s\n", result.error_message.c_str());
    }
}

float AdaptiveLayerRedistributor::calculate_actual_improvement() const {
    // In real implementation, would compare performance before/after
    // For now, return placeholder
    return 0.15f;  // 15% improvement
}

void AdaptiveLayerRedistributor::update_stability_metrics(const RedistributionResult& result) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    stats_.total_redistributions++;
    if (result.success) {
        stats_.successful_redistributions++;
    } else {
        stats_.failed_redistributions++;
    }
    
    stats_.total_redistribution_time += result.duration;
    stats_.last_redistribution = std::chrono::steady_clock::now();
    
    // Update average improvement
    if (result.success && stats_.successful_redistributions > 0) {
        stats_.average_improvement = 
            ((stats_.average_improvement * (stats_.successful_redistributions - 1)) +
             result.actual_improvement) / stats_.successful_redistributions;
    }
    
    // Add to history
    {
        std::lock_guard<std::mutex> hist_lock(history_mutex_);
        redistribution_history_.push_back(result);
        
        // Keep only recent history
        if (redistribution_history_.size() > 100) {
            redistribution_history_.pop_front();
        }
    }
}

AdaptiveLayerRedistributor::RedistributionStats 
AdaptiveLayerRedistributor::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void AdaptiveLayerRedistributor::reset_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = RedistributionStats();
}

void AdaptiveLayerRedistributor::set_redistribution_callback(
    RedistributionCallback callback) {
    redistribution_callback_ = callback;
}

// Utility functions implementation
namespace redistribution_utils {

bool should_redistribute(const RuntimePerformanceMonitor& monitor,
                       const DistributionResult& current_distribution) {
    // Check if redistribution would be beneficial
    float current_speedup = monitor.calculate_current_speedup();
    float expected_speedup = current_distribution.expected_speedup;
    
    // Redistribute if performance is 20% below expected
    return current_speedup < expected_speedup * 0.8f;
}

RedistributionRequest create_request_from_anomalies(
    const std::vector<PerformanceAnomaly>& anomalies) {
    
    RedistributionRequest request;
    request.timestamp = std::chrono::steady_clock::now();
    
    // Analyze anomalies to determine request type and urgency
    float max_severity = 0.0f;
    std::unordered_set<int> affected_gpus;
    
    for (const auto& anomaly : anomalies) {
        max_severity = std::max(max_severity, anomaly.severity);
        
        if (anomaly.affected_gpu_id >= 0) {
            affected_gpus.insert(anomaly.affected_gpu_id);
        }
        
        // Set request type based on most severe anomaly
        switch (anomaly.type) {
            case PerformanceAnomalyType::THERMAL_THROTTLE:
                request.type = RedistributionRequest::THERMAL_TRIGGERED;
                break;
            case PerformanceAnomalyType::MEMORY_PRESSURE:
                request.type = RedistributionRequest::MEMORY_TRIGGERED;
                break;
            default:
                if (request.type != RedistributionRequest::THERMAL_TRIGGERED &&
                    request.type != RedistributionRequest::MEMORY_TRIGGERED) {
                    request.type = RedistributionRequest::PERFORMANCE_TRIGGERED;
                }
                break;
        }
    }
    
    request.urgency = max_severity;
    request.affected_gpus.assign(affected_gpus.begin(), affected_gpus.end());
    request.reason = "Multiple anomalies detected (count: " + 
                    std::to_string(anomalies.size()) + ")";
    
    return request;
}

} // namespace redistribution_utils

} // namespace orchestration
} // namespace llama