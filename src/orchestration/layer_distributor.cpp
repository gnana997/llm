#include "layer_distributor.h"
#include "metrics.h"
#include "llama-impl.h"
#include "llama-hparams.h"
#include "llama-model.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <queue>
#include <limits>
#include <iostream>

namespace llama {
namespace orchestration {

struct LayerDistributor::Impl {
    GpuProfiler profiler;
    std::chrono::steady_clock::time_point last_cache_cleanup;
    
    Impl() : last_cache_cleanup(std::chrono::steady_clock::now()) {}
    
    void cleanup_old_cache_entries(std::unordered_map<std::string, CacheEntry>& cache) {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::minutes>(now - last_cache_cleanup).count() < 5) {
            return;  // Only cleanup every 5 minutes
        }
        
        auto it = cache.begin();
        while (it != cache.end()) {
            auto age = std::chrono::duration_cast<std::chrono::minutes>(now - it->second.timestamp);
            if (age.count() > CACHE_TTL_MINUTES) {
                it = cache.erase(it);
            } else {
                ++it;
            }
        }
        
        // Also limit cache size
        if (cache.size() > CACHE_MAX_SIZE) {
            // Remove oldest entries
            std::vector<std::pair<std::string, std::chrono::steady_clock::time_point>> entries;
            for (const auto& [key, entry] : cache) {
                entries.push_back({key, entry.timestamp});
            }
            
            std::sort(entries.begin(), entries.end(),
                     [](const auto& a, const auto& b) {
                         return a.second < b.second;
                     });
            
            size_t to_remove = cache.size() - CACHE_MAX_SIZE;
            for (size_t i = 0; i < to_remove; ++i) {
                cache.erase(entries[i].first);
            }
        }
        
        last_cache_cleanup = now;
    }
};

LayerDistributor::LayerDistributor() : 
    pImpl(std::make_unique<Impl>()),
    arch_analyzer_(std::make_unique<GpuArchitectureAnalyzer>()) {}
LayerDistributor::~LayerDistributor() = default;

void LayerDistributor::initialize(const std::vector<GpuProfile>& gpu_profiles) {
    gpu_profiles_ = gpu_profiles;
    
    // Sort GPUs by capability score for consistent ordering
    std::sort(gpu_profiles_.begin(), gpu_profiles_.end(),
              [](const GpuProfile& a, const GpuProfile& b) {
                  return a.get_capability_score() > b.get_capability_score();
              });
    
    LLAMA_LOG_INFO("[Layer Distributor] Initialized with %zu GPUs\n", gpu_profiles_.size());
    for (size_t i = 0; i < gpu_profiles_.size(); ++i) {
        LLAMA_LOG_INFO("  GPU %zu: %s (score: %.1f)\n", 
                i, gpu_profiles_[i].name.c_str(), gpu_profiles_[i].get_capability_score());
    }
}

DistributionResult LayerDistributor::distribute_layers(
    const llama_model& model,
    int total_layers,
    const DistributionConfig& config) {
    
    // Start timing for metrics
    auto& metrics = LayerDistributionMetrics::instance();
    TimerScope timer(metrics.distribution_duration_us);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    DistributionResult result;
    
    // Check cache first
    std::string model_hash = generate_model_hash(model);
    if (config.enable_caching && get_cached_distribution(model_hash, gpu_profiles_, result)) {
        result.cache_hit = true;
        metrics_.cache_hits++;
        metrics.cache_hits->increment();
        LLAMA_LOG_INFO("[Layer Distributor] Using cached distribution for model %s\n", model_hash.c_str());
        
        // Update GPU layer assignment metrics
        for (size_t i = 0; i < gpu_profiles_.size() && i < 8; ++i) {
            metrics.gpu_layers_assigned[i]->set(result.layers_per_gpu[i]);
        }
        
        return result;
    }
    
    metrics_.cache_misses++;
    metrics.cache_misses->increment();
    
    // Profile model layers
    std::vector<LayerProfile> layer_profiles = profile_model_layers(model, total_layers);
    
    // Choose distribution strategy
    std::vector<LayerAssignment> assignments;
    
    switch (config.strategy) {
        case DistributionStrategy::EQUAL:
            assignments = distribute_equal(layer_profiles, gpu_profiles_);
            break;
            
        case DistributionStrategy::OPTIMIZED:
            assignments = distribute_optimized(layer_profiles, gpu_profiles_, config);
            break;
            
        case DistributionStrategy::MANUAL:
            assignments = distribute_manual(layer_profiles, gpu_profiles_, config.manual_splits);
            break;
    }
    
    // Build result
    result.assignments = assignments;
    result.layers_per_gpu.resize(gpu_profiles_.size(), 0);
    result.memory_per_gpu.resize(gpu_profiles_.size(), 0.0f);
    result.compute_load_per_gpu.resize(gpu_profiles_.size(), 0.0f);
    
    // Calculate statistics
    for (const auto& assignment : assignments) {
        result.layers_per_gpu[assignment.gpu_id]++;
        result.memory_per_gpu[assignment.gpu_id] += assignment.memory_bytes / (1024.0f * 1024.0f * 1024.0f);
    }
    
    // Calculate compute load distribution
    float total_compute = 0.0f;
    for (size_t i = 0; i < gpu_profiles_.size(); ++i) {
        result.compute_load_per_gpu[i] = result.layers_per_gpu[i] * gpu_profiles_[i].measured_performance_score;
        total_compute += result.compute_load_per_gpu[i];
    }
    
    if (total_compute > 0) {
        for (size_t i = 0; i < gpu_profiles_.size(); ++i) {
            result.compute_load_per_gpu[i] = (result.compute_load_per_gpu[i] / total_compute) * 100.0f;
        }
    }
    
    // Estimate speedup
    std::vector<LayerAssignment> baseline = distribute_equal(layer_profiles, gpu_profiles_);
    result.expected_speedup = estimate_performance_improvement(assignments, baseline);
    
    // Generate summary
    result.distribution_summary = distribution_utils::format_distribution_summary(result, gpu_profiles_);
    
    // Cache the result
    if (config.enable_caching) {
        cache_distribution(model_hash, result);
    }
    
    // Update metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    metrics_.distribution_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    metrics_.total_distributions++;
    metrics.total_distributions->increment();
    metrics_.average_speedup = (metrics_.average_speedup * (metrics_.total_distributions - 1) + 
                                result.expected_speedup) / metrics_.total_distributions;
    
    // Update GPU layer assignment metrics
    for (size_t i = 0; i < gpu_profiles_.size() && i < 8; ++i) {
        metrics.gpu_layers_assigned[i]->set(result.layers_per_gpu[i]);
    }
    
    log_distribution_result(result);
    
    // Log metrics periodically
    if (metrics_.total_distributions % 10 == 0) {
        MetricsRegistry::instance().log_all();
    }
    
    return result;
}

std::vector<LayerProfile> LayerDistributor::profile_model_layers(
    const llama_model& model,
    int total_layers) {
    
    std::vector<LayerProfile> profiles;
    profiles.reserve(total_layers);
    
    // Get model dimensions
    const auto& hparams = model.hparams;
    size_t hidden_size = hparams.n_embd;
    size_t n_heads = hparams.n_head();  // n_head is a method
    size_t n_ff = hparams.n_ff();      // n_ff is a method
    size_t vocab_size = 32000;          // Default vocab size (n_vocab doesn't exist)
    
    LLAMA_LOG_INFO("[Layer Profiling] Model dimensions: hidden=%zu, heads=%zu, ff=%zu, vocab=%zu\n",
            hidden_size, n_heads, n_ff, vocab_size);
    
    // Typical transformer structure
    for (int i = 0; i < total_layers; ++i) {
        // Attention layer
        LayerProfile attn_profile = pImpl->profiler.estimate_layer_profile(
            "attention", i, hidden_size, 2048, 0);  // Assume 2048 seq length
        profiles.push_back(attn_profile);
        
        // Feedforward layer
        LayerProfile ff_profile = pImpl->profiler.estimate_layer_profile(
            "feedforward", i, hidden_size, 2048, 0);
        profiles.push_back(ff_profile);
    }
    
    // Add embedding and output layers
    LayerProfile embed_profile = pImpl->profiler.estimate_layer_profile(
        "embedding", 0, hidden_size, 1, vocab_size);
    profiles.insert(profiles.begin(), embed_profile);
    
    LayerProfile output_profile = pImpl->profiler.estimate_layer_profile(
        "embedding", total_layers, hidden_size, 1, vocab_size);
    profiles.push_back(output_profile);
    
    return profiles;
}

std::vector<LayerAssignment> LayerDistributor::distribute_equal(
    const std::vector<LayerProfile>& layers,
    const std::vector<GpuProfile>& gpus) {
    
    std::vector<LayerAssignment> assignments;
    size_t layers_per_gpu = (layers.size() + gpus.size() - 1) / gpus.size();
    
    for (size_t i = 0; i < layers.size(); ++i) {
        LayerAssignment assignment;
        assignment.layer_index = i;
        assignment.gpu_id = i / layers_per_gpu;
        assignment.memory_bytes = layers[i].memory_requirement_bytes;
        assignment.expected_load_percentage = 100.0f / gpus.size();
        assignment.rationale = "Equal distribution";
        
        assignments.push_back(assignment);
    }
    
    return assignments;
}

std::vector<LayerAssignment> LayerDistributor::distribute_optimized(
    const std::vector<LayerProfile>& layers,
    const std::vector<GpuProfile>& gpus,
    const DistributionConfig& config) {
    
    const int n_layers = layers.size();
    const int n_gpus = gpus.size();
    
    if (n_gpus == 1) {
        // Single GPU - assign all layers
        return distribute_equal(layers, gpus);
    }
    
    // Get architecture capabilities for all GPUs
    std::vector<ArchitectureCapabilities> arch_caps(n_gpus);
    std::vector<std::vector<LayerAffinityScore>> affinity_scores(n_layers);
    
    if (arch_analyzer_) {
        for (int gpu = 0; gpu < n_gpus; ++gpu) {
            arch_caps[gpu] = arch_analyzer_->analyze_architecture(gpus[gpu]);
        }
        
        // Pre-compute layer affinity scores
        for (int layer = 0; layer < n_layers; ++layer) {
            affinity_scores[layer].resize(n_gpus);
            for (int gpu = 0; gpu < n_gpus; ++gpu) {
                affinity_scores[layer][gpu] = arch_analyzer_->calculate_layer_affinity(
                    gpus[gpu], layers[layer], arch_caps[gpu]);
            }
        }
    }
    
    // Advanced multi-objective dynamic programming
    // State: dp[layer][gpu] = best multi-objective cost to assign layers 0..layer with last layer on gpu
    std::vector<std::vector<DPState>> dp(n_layers, std::vector<DPState>(n_gpus));
    std::vector<std::vector<int>> parent(n_layers, std::vector<int>(n_gpus, -1));
    
    // Initialize DP table
    for (int i = 0; i < n_layers; ++i) {
        for (int j = 0; j < n_gpus; ++j) {
            dp[i][j].total_cost = std::numeric_limits<float>::infinity();
            dp[i][j].gpu_loads.resize(n_gpus, 0.0f);
            dp[i][j].gpu_memory.resize(n_gpus, 0);
        }
    }
    
    // Base case - first layer
    for (int gpu = 0; gpu < n_gpus; ++gpu) {
        dp[0][gpu].total_cost = 0.0f;
        dp[0][gpu].last_gpu = gpu;
        dp[0][gpu].gpu_loads[gpu] = layers[0].compute_intensity;
        dp[0][gpu].gpu_memory[gpu] = layers[0].memory_requirement_bytes;
    }
    
    // Fill DP table
    for (int layer = 1; layer < n_layers; ++layer) {
        for (int gpu = 0; gpu < n_gpus; ++gpu) {
            // Try assigning this layer to each GPU
            for (int prev_gpu = 0; prev_gpu < n_gpus; ++prev_gpu) {
                DPState new_state = dp[layer - 1][prev_gpu];
                
                // Check memory constraint
                if (new_state.gpu_memory[gpu] + layers[layer].memory_requirement_bytes >
                    gpus[gpu].available_memory_bytes * config.memory_safety_margin) {
                    continue;  // Skip if memory exceeded
                }
                
                // Update state
                new_state.gpu_loads[gpu] += layers[layer].compute_intensity / gpus[gpu].measured_performance_score;
                new_state.gpu_memory[gpu] += layers[layer].memory_requirement_bytes;
                
                // Calculate cost
                float cost = calculate_assignment_cost(layers[layer], gpus[gpu], new_state, config);
                new_state.total_cost += cost;
                
                // Update best assignment
                if (new_state.total_cost < dp[layer][gpu].total_cost) {
                    dp[layer][gpu] = new_state;
                    parent[layer][gpu] = prev_gpu;
                }
            }
        }
    }
    
    // Find best final assignment
    int best_final_gpu = 0;
    float best_cost = dp[n_layers - 1][0].total_cost;
    for (int gpu = 1; gpu < n_gpus; ++gpu) {
        if (dp[n_layers - 1][gpu].total_cost < best_cost) {
            best_cost = dp[n_layers - 1][gpu].total_cost;
            best_final_gpu = gpu;
        }
    }
    
    // Reconstruct path
    std::vector<int> gpu_assignments(n_layers);
    int current_gpu = best_final_gpu;
    for (int layer = n_layers - 1; layer >= 0; --layer) {
        gpu_assignments[layer] = current_gpu;
        if (layer > 0) {
            current_gpu = parent[layer][current_gpu];
        }
    }
    
    // Build assignments with enhanced information
    std::vector<LayerAssignment> assignments;
    for (int i = 0; i < n_layers; ++i) {
        LayerAssignment assignment;
        assignment.layer_index = i;
        assignment.gpu_id = gpu_assignments[i];
        assignment.memory_bytes = layers[i].memory_requirement_bytes;
        
        // Enhanced: Analyze compute pattern
        assignment.compute_pattern = analyze_layer_compute_pattern(layers[i]);
        
        // Enhanced: Calculate architecture affinity
        if (arch_analyzer_) {
            auto gpu_arch_caps = arch_caps[assignment.gpu_id];
            auto affinity = affinity_scores[i][assignment.gpu_id];
            
            // Set architecture affinity score
            switch (assignment.compute_pattern) {
                case LayerComputePattern::ATTENTION_PATTERN:
                    assignment.architecture_affinity_score = affinity.attention_score;
                    break;
                case LayerComputePattern::GEMM_HEAVY:
                    assignment.architecture_affinity_score = affinity.feedforward_score;
                    break;
                case LayerComputePattern::EMBEDDING_LOOKUP:
                    assignment.architecture_affinity_score = affinity.embedding_score;
                    break;
                default:
                    assignment.architecture_affinity_score = 1.0f;
            }
            
            // Predict execution time
            assignment.predicted_execution_time_ms = predict_layer_execution_time(
                layers[i], gpus[assignment.gpu_id], gpu_arch_caps);
            
            // Determine if tensor cores will be used
            assignment.uses_tensor_cores = arch_analyzer_->should_use_tensor_cores(
                layers[i], gpu_arch_caps);
            
            // Get optimal kernel configuration
            auto kernel_config = arch_analyzer_->get_optimal_layer_config(
                gpus[assignment.gpu_id], layers[i], gpu_arch_caps);
            assignment.optimal_kernel = kernel_config.kernel_variant;
        } else {
            // Fallback values
            assignment.architecture_affinity_score = 1.0f;
            assignment.predicted_execution_time_ms = 0.0f;
            assignment.uses_tensor_cores = false;
            assignment.optimal_kernel = "standard";
        }
        
        // Calculate expected load percentage based on predicted execution time
        float total_predicted_time = 0.0f;
        for (int j = 0; j < n_layers; ++j) {
            if (arch_analyzer_) {
                total_predicted_time += predict_layer_execution_time(
                    layers[j], gpus[gpu_assignments[j]], arch_caps[gpu_assignments[j]]);
            }
        }
        assignment.expected_load_percentage = 
            (assignment.predicted_execution_time_ms / total_predicted_time) * 100.0f;
        
        // Generate enhanced rationale
        std::ostringstream rationale;
        rationale << "GPU " << assignment.gpu_id << " (" << gpus[assignment.gpu_id].name << ")";
        rationale << " - Affinity: " << std::fixed << std::setprecision(2) 
                  << assignment.architecture_affinity_score;
        
        if (assignment.uses_tensor_cores) {
            rationale << ", using " << assignment.optimal_kernel;
        }
        
        rationale << ", predicted " << std::fixed << std::setprecision(1) 
                  << assignment.predicted_execution_time_ms << "ms";
        
        assignment.rationale = rationale.str();
        assignments.push_back(assignment);
        
        log_distribution_decision(i, assignment.gpu_id, assignment.rationale);
    }
    
    return assignments;
}

std::vector<LayerAssignment> LayerDistributor::distribute_manual(
    const std::vector<LayerProfile>& layers,
    const std::vector<GpuProfile>& gpus,
    const std::vector<float>& splits) {
    
    if (splits.empty() || splits.size() != gpus.size()) {
        LLAMA_LOG_INFO("[Layer Distributor] Invalid manual splits, falling back to equal distribution\n");
        return distribute_equal(layers, gpus);
    }
    
    // Normalize splits
    float total_split = std::accumulate(splits.begin(), splits.end(), 0.0f);
    std::vector<float> normalized_splits;
    for (float split : splits) {
        normalized_splits.push_back(split / total_split);
    }
    
    // Assign layers based on splits
    std::vector<LayerAssignment> assignments;
    size_t layer_idx = 0;
    
    for (size_t gpu_idx = 0; gpu_idx < gpus.size(); ++gpu_idx) {
        size_t layers_for_gpu = std::round(normalized_splits[gpu_idx] * layers.size());
        if (gpu_idx == gpus.size() - 1) {
            // Last GPU gets remaining layers
            layers_for_gpu = layers.size() - layer_idx;
        }
        
        for (size_t i = 0; i < layers_for_gpu && layer_idx < layers.size(); ++i, ++layer_idx) {
            LayerAssignment assignment;
            assignment.layer_index = layer_idx;
            assignment.gpu_id = gpu_idx;
            assignment.memory_bytes = layers[layer_idx].memory_requirement_bytes;
            assignment.expected_load_percentage = normalized_splits[gpu_idx] * 100.0f;
            assignment.rationale = "Manual split: " + std::to_string(splits[gpu_idx]);
            
            assignments.push_back(assignment);
        }
    }
    
    return assignments;
}

float LayerDistributor::calculate_assignment_cost(
    const LayerProfile& layer,
    const GpuProfile& gpu,
    const DPState& current_state,
    const DistributionConfig& config) {
    
    // Multi-objective cost function with architecture awareness
    float total_cost = 0.0f;
    
    // 1. Load balance cost (improved with predicted execution time)
    float predicted_time_ms = 0.0f;
    if (arch_analyzer_) {
        auto arch_caps = arch_analyzer_->analyze_architecture(gpu);
        predicted_time_ms = predict_layer_execution_time(layer, gpu, arch_caps);
    } else {
        // Fallback to simple calculation
        float flops = layer.compute_intensity * layer.memory_requirement_bytes;
        predicted_time_ms = (flops / 1e12f) / gpu.theoretical_tflops * 1000.0f;
    }
    
    // Update load with actual predicted time instead of simple compute intensity
    std::vector<float> predicted_loads = current_state.gpu_loads;
    int gpu_idx = static_cast<int>(&gpu - &gpu_profiles_[0]);  // Get GPU index
    predicted_loads[gpu_idx] += predicted_time_ms;
    
    float load_imbalance = calculate_load_imbalance(predicted_loads);
    float load_cost = load_imbalance * LOAD_BALANCE_WEIGHT;
    
    // 2. Memory balance cost (unchanged)
    float memory_imbalance = calculate_memory_imbalance(current_state.gpu_memory);
    float memory_cost = memory_imbalance * MEMORY_BALANCE_WEIGHT;
    
    // 3. Architecture affinity cost (new)
    float affinity_cost = 0.0f;
    if (arch_analyzer_ && gpu_idx < static_cast<int>(gpu_profiles_.size())) {
        auto arch_caps = arch_analyzer_->analyze_architecture(gpu);
        auto affinity = arch_analyzer_->calculate_layer_affinity(gpu, layer, arch_caps);
        
        // Calculate affinity score based on layer type
        float affinity_score = 1.0f;
        LayerComputePattern pattern = analyze_layer_compute_pattern(layer);
        
        switch (pattern) {
            case LayerComputePattern::ATTENTION_PATTERN:
                affinity_score = affinity.attention_score;
                break;
            case LayerComputePattern::GEMM_HEAVY:
                affinity_score = affinity.feedforward_score;
                break;
            case LayerComputePattern::EMBEDDING_LOOKUP:
                affinity_score = affinity.embedding_score;
                break;
            default:
                affinity_score = (affinity.attention_score + affinity.feedforward_score) / 2.0f;
        }
        
        // Lower affinity means higher cost
        affinity_cost = (2.0f - affinity_score) * 0.3f;  // Weight of 0.3
    }
    
    // 4. Communication cost (for multi-GPU scenarios)
    float comm_cost = 0.0f;
    if (current_state.last_gpu != gpu_idx && arch_analyzer_) {
        // Calculate communication cost for layer transition
        size_t transfer_size = layer.activation_weight * layer.memory_requirement_bytes;
        comm_cost = arch_analyzer_->calculate_communication_cost(
            gpu_profiles_[current_state.last_gpu],
            gpu,
            transfer_size
        ) * 0.001f;  // Convert to normalized cost
    }
    
    // 5. Power efficiency cost (new)
    float power_cost = 0.0f;
    if (arch_analyzer_) {
        auto arch_caps = arch_analyzer_->analyze_architecture(gpu);
        if (arch_caps.efficiency_per_watt > 0) {
            // Normalize power efficiency (0.1 TFLOPS/W is baseline)
            float efficiency_factor = arch_caps.efficiency_per_watt / 0.1f;
            power_cost = (1.0f / efficiency_factor) * 0.1f;  // Weight of 0.1
        }
    }
    
    // 6. Thermal throttling risk (based on current load)
    float thermal_cost = 0.0f;
    if (predicted_loads[gpu_idx] > 1000.0f) {  // If GPU is heavily loaded (>1s of work)
        thermal_cost = (predicted_loads[gpu_idx] / 1000.0f) * 0.1f;
    }
    
    // Combine all costs
    total_cost = load_cost + memory_cost + affinity_cost + comm_cost + power_cost + thermal_cost;
    
    // Apply configuration preferences
    if (config.prefer_memory_balance) {
        memory_cost *= 2.0f;
        load_cost *= 0.5f;
    }
    
    return total_cost;
}

float LayerDistributor::calculate_load_imbalance(const std::vector<float>& loads) const {
    if (loads.empty()) return 0.0f;
    
    float mean = std::accumulate(loads.begin(), loads.end(), 0.0f) / loads.size();
    float variance = 0.0f;
    
    for (float load : loads) {
        variance += (load - mean) * (load - mean);
    }
    
    return std::sqrt(variance / loads.size()) / (mean + 1e-6f);
}

float LayerDistributor::calculate_memory_imbalance(const std::vector<size_t>& memory) const {
    if (memory.empty()) return 0.0f;
    
    std::vector<float> memory_gb;
    for (size_t mem : memory) {
        memory_gb.push_back(mem / (1024.0f * 1024.0f * 1024.0f));
    }
    
    return calculate_load_imbalance(memory_gb);
}

bool LayerDistributor::validate_distribution(
    const std::vector<LayerAssignment>& assignments,
    const std::vector<GpuProfile>& gpus) {
    
    std::vector<size_t> memory_usage(gpus.size(), 0);
    
    for (const auto& assignment : assignments) {
        if (assignment.gpu_id >= static_cast<int>(gpus.size())) {
            LLAMA_LOG_INFO("[Validation] Invalid GPU ID %d for layer %d\n", 
                    assignment.gpu_id, assignment.layer_index);
            return false;
        }
        
        memory_usage[assignment.gpu_id] += assignment.memory_bytes;
    }
    
    for (size_t i = 0; i < gpus.size(); ++i) {
        if (memory_usage[i] > gpus[i].available_memory_bytes * 0.95f) {
            LLAMA_LOG_INFO("[Validation] GPU %zu memory exceeded: %s required, %s available\n",
                    i,
                    profiling_utils::format_memory_size(memory_usage[i]).c_str(),
                    profiling_utils::format_memory_size(gpus[i].available_memory_bytes).c_str());
            return false;
        }
    }
    
    return true;
}

void LayerDistributor::cache_distribution(
    const std::string& model_hash,
    const DistributionResult& result) {
    
    pImpl->cleanup_old_cache_entries(distribution_cache_);
    
    CacheEntry entry;
    entry.result = result;
    entry.timestamp = std::chrono::steady_clock::now();
    
    distribution_cache_[model_hash] = entry;
    
    LLAMA_LOG_INFO("[Cache] Stored distribution for model %s (cache size: %zu)\n",
            model_hash.c_str(), distribution_cache_.size());
}

bool LayerDistributor::get_cached_distribution(
    const std::string& model_hash,
    const std::vector<GpuProfile>& gpus,
    DistributionResult& result) {
    
    auto it = distribution_cache_.find(model_hash);
    if (it == distribution_cache_.end()) {
        return false;
    }
    
    // Check if GPU configuration matches
    if (gpus.size() != gpu_profiles_.size()) {
        return false;
    }
    
    for (size_t i = 0; i < gpus.size(); ++i) {
        if (gpus[i].device_id != gpu_profiles_[i].device_id ||
            std::abs(static_cast<long long>(gpus[i].available_memory_bytes) - 
                     static_cast<long long>(gpu_profiles_[i].available_memory_bytes)) > 
            1024 * 1024 * 1024) {  // 1GB tolerance
            return false;
        }
    }
    
    result = it->second.result;
    return true;
}

void LayerDistributor::clear_cache() {
    distribution_cache_.clear();
    LLAMA_LOG_INFO("[Cache] Cleared distribution cache\n");
}

float LayerDistributor::estimate_performance_improvement(
    const std::vector<LayerAssignment>& optimized,
    const std::vector<LayerAssignment>& baseline) {
    
    // Calculate effective compute distribution
    auto calculate_max_latency = [this](const std::vector<LayerAssignment>& assignments) {
        std::vector<float> gpu_latencies(gpu_profiles_.size(), 0.0f);
        
        for (const auto& assignment : assignments) {
            float layer_compute = 1.0f;  // Normalized compute per layer
            float gpu_performance = gpu_profiles_[assignment.gpu_id].measured_performance_score;
            gpu_latencies[assignment.gpu_id] += layer_compute / gpu_performance;
        }
        
        return *std::max_element(gpu_latencies.begin(), gpu_latencies.end());
    };
    
    float baseline_latency = calculate_max_latency(baseline);
    float optimized_latency = calculate_max_latency(optimized);
    
    if (optimized_latency > 0) {
        return baseline_latency / optimized_latency;
    }
    
    return 1.0f;
}

std::string LayerDistributor::generate_model_hash(const llama_model& model) const {
    std::ostringstream hash;
    const auto& hparams = model.hparams;
    
    hash << "model_" << hparams.n_layer << "_" << hparams.n_embd << "_" 
         << hparams.n_head() << "_" << hparams.n_ff() << "_" << 32000;
    
    return hash.str();
}

void LayerDistributor::log_distribution_result(const DistributionResult& result) const {
    LLAMA_LOG_INFO("\n[Layer Distribution Result]\n");
    LLAMA_LOG_INFO("  Strategy: %s\n", result.cache_hit ? "Cached" : "Computed");
    LLAMA_LOG_INFO("  Expected speedup: %.2fx\n", result.expected_speedup);
    LLAMA_LOG_INFO("  Distribution time: %ld μs\n", metrics_.distribution_time.count());
    LLAMA_LOG_INFO("  Cache statistics: %zu hits, %zu misses (%.1f%% hit rate)\n",
                   metrics_.cache_hits, metrics_.cache_misses,
                   metrics_.cache_hits * 100.0f / (metrics_.cache_hits + metrics_.cache_misses + 0.001f));
    LLAMA_LOG_INFO("  GPU assignment summary:\n");
    
    for (size_t i = 0; i < gpu_profiles_.size(); ++i) {
        float efficiency = (result.layers_per_gpu[i] > 0) ? 
            result.compute_load_per_gpu[i] / (100.0f / gpu_profiles_.size()) : 0.0f;
        
        LLAMA_LOG_INFO("    GPU %zu (%s):\n", i, gpu_profiles_[i].name.c_str());
        LLAMA_LOG_INFO("      - Layers: %d\n", result.layers_per_gpu[i]);
        LLAMA_LOG_INFO("      - Memory: %.2f GB (%.1f%% of available)\n",
                       result.memory_per_gpu[i],
                       result.memory_per_gpu[i] * 100.0f / (gpu_profiles_[i].available_memory_bytes / 1e9f));
        LLAMA_LOG_INFO("      - Compute load: %.1f%%\n", result.compute_load_per_gpu[i]);
        LLAMA_LOG_INFO("      - Efficiency: %.2fx\n", efficiency);
    }
    
    LLAMA_LOG_INFO("\n%s\n", result.distribution_summary.c_str());
    
    // Log decision rationale for first few layers
    LLAMA_LOG_INFO("\n  Layer assignment details (first 10 layers):\n");
    int shown = 0;
    for (const auto& assignment : result.assignments) {
        if (shown++ >= 10) break;
        LLAMA_LOG_INFO("    Layer %d -> GPU %d: %s\n",
                       assignment.layer_index, assignment.gpu_id, assignment.rationale.c_str());
    }
    if (result.assignments.size() > 10) {
        LLAMA_LOG_INFO("    ... (%zu more layers)\n", result.assignments.size() - 10);
    }
}

void LayerDistributor::log_distribution_decision(
    int layer_index,
    int gpu_id,
    const std::string& reason) const {
    
    LLAMA_LOG_INFO("[Distribution] Layer %d -> GPU %d: %s\n", layer_index, gpu_id, reason.c_str());
}

DistributionMetrics LayerDistributor::get_metrics() const {
    return metrics_;
}

void LayerDistributor::reset_metrics() {
    metrics_ = DistributionMetrics();
}

DistributionStrategy LayerDistributor::parse_strategy(const std::string& str) {
    if (str == "equal") return DistributionStrategy::EQUAL;
    if (str == "optimized" || str == "intelligent") return DistributionStrategy::OPTIMIZED;
    if (str == "manual") return DistributionStrategy::MANUAL;
    
    LLAMA_LOG_INFO("[Warning] Unknown distribution strategy '%s', using OPTIMIZED\n", str.c_str());
    return DistributionStrategy::OPTIMIZED;
}

std::string LayerDistributor::strategy_to_string(DistributionStrategy strategy) {
    switch (strategy) {
        case DistributionStrategy::EQUAL: return "equal";
        case DistributionStrategy::OPTIMIZED: return "optimized";
        case DistributionStrategy::MANUAL: return "manual";
    }
    return "unknown";
}

// Utility functions implementation
namespace distribution_utils {

float calculate_theoretical_speedup(
    const std::vector<float>& gpu_capabilities,
    const std::vector<int>& layer_assignments) {
    
    if (gpu_capabilities.empty() || layer_assignments.empty()) return 1.0f;
    
    // Calculate load per GPU
    std::vector<float> gpu_loads(gpu_capabilities.size(), 0.0f);
    for (int gpu_id : layer_assignments) {
        if (gpu_id >= 0 && gpu_id < static_cast<int>(gpu_capabilities.size())) {
            gpu_loads[gpu_id] += 1.0f;
        }
    }
    
    // Calculate completion time (limited by slowest GPU)
    float max_time = 0.0f;
    float equal_time = 0.0f;
    [[maybe_unused]] float total_capability = std::accumulate(gpu_capabilities.begin(), gpu_capabilities.end(), 0.0f);
    
    for (size_t i = 0; i < gpu_capabilities.size(); ++i) {
        if (gpu_loads[i] > 0) {
            float gpu_time = gpu_loads[i] / gpu_capabilities[i];
            max_time = std::max(max_time, gpu_time);
        }
        
        // Equal distribution time
        float equal_load = layer_assignments.size() / static_cast<float>(gpu_capabilities.size());
        equal_time = std::max(equal_time, equal_load / gpu_capabilities[i]);
    }
    
    return equal_time / max_time;
}

std::string format_distribution_summary(
    const DistributionResult& result,
    const std::vector<GpuProfile>& gpus) {
    
    std::ostringstream summary;
    summary << "Layer Distribution Summary:\n";
    summary << "─────────────────────────────────────────\n";
    
    // GPU utilization bars
    for (size_t i = 0; i < gpus.size(); ++i) {
        summary << "GPU " << i << " [" << gpus[i].name << "]\n";
        
        // Memory bar
        float mem_percent = (result.memory_per_gpu[i] / (gpus[i].total_memory_bytes / 1e9f)) * 100.0f;
        int mem_bars = static_cast<int>(mem_percent / 5.0f);
        summary << "  Memory: [";
        for (int j = 0; j < 20; ++j) {
            summary << (j < mem_bars ? "█" : "░");
        }
        summary << "] " << std::fixed << std::setprecision(1) << mem_percent << "%\n";
        
        // Compute bar
        int compute_bars = static_cast<int>(result.compute_load_per_gpu[i] / 5.0f);
        summary << "  Compute:[";
        for (int j = 0; j < 20; ++j) {
            summary << (j < compute_bars ? "█" : "░");
        }
        summary << "] " << std::fixed << std::setprecision(1) << result.compute_load_per_gpu[i] << "%\n";
        
        summary << "  Layers: " << result.layers_per_gpu[i] << "\n\n";
    }
    
    return summary.str();
}

void print_distribution_visualization(
    const DistributionResult& result,
    const std::vector<GpuProfile>& gpus,
    int total_layers) {
    
    std::cout << "\nLayer Distribution Visualization:\n";
    std::cout << "Layer: ";
    for (int i = 0; i < total_layers; ++i) {
        std::cout << std::setw(3) << i;
    }
    std::cout << "\n";
    
    for (size_t gpu_id = 0; gpu_id < gpus.size(); ++gpu_id) {
        std::cout << "GPU " << gpu_id << ": ";
        for (const auto& assignment : result.assignments) {
            if (assignment.gpu_id == static_cast<int>(gpu_id)) {
                std::cout << " ■ ";
            } else {
                std::cout << " · ";
            }
        }
        std::cout << " (" << gpus[gpu_id].name << ")\n";
    }
    std::cout << "\n";
}

LayerPattern analyze_layer_patterns(const std::vector<LayerProfile>& layers) {
    LayerPattern pattern;
    
    size_t attention_count = 0;
    size_t feedforward_count = 0;
    size_t total_memory = 0;
    
    for (const auto& layer : layers) {
        if (layer.layer_type == "attention" || layer.layer_type == "self_attention") {
            attention_count++;
        } else if (layer.layer_type == "feedforward" || layer.layer_type == "mlp") {
            feedforward_count++;
        }
        total_memory += layer.memory_requirement_bytes;
    }
    
    pattern.attention_ratio = attention_count / static_cast<float>(layers.size());
    pattern.feedforward_ratio = feedforward_count / static_cast<float>(layers.size());
    pattern.avg_memory_per_layer = total_memory / static_cast<float>(layers.size());
    
    if (pattern.attention_ratio > 0.6f) {
        pattern.pattern_type = "attention-heavy";
    } else if (pattern.feedforward_ratio > 0.6f) {
        pattern.pattern_type = "feedforward-heavy";
    } else {
        pattern.pattern_type = "balanced";
    }
    
    return pattern;
}

} // namespace distribution_utils

// Integration helper implementation
bool LayerDistributionIntegrator::apply_distribution(
    [[maybe_unused]] llama_model& model,
    const DistributionResult& distribution) {
    
    // This would integrate with the actual llama.cpp model loading
    // For now, we'll log the intended distribution
    LLAMA_LOG_INFO("[Integration] Applying distribution to model with %zu assignments\n",
            distribution.assignments.size());
    
    // TODO: Actual integration with llama_model structure
    // This would modify the tensor placement during model loading
    
    return true;
}

std::vector<int> LayerDistributionIntegrator::get_current_distribution([[maybe_unused]] const llama_model& model) {
    std::vector<int> distribution;
    
    // TODO: Extract current distribution from loaded model
    // This would query the actual tensor placement
    
    return distribution;
}

bool LayerDistributionIntegrator::is_model_compatible(const llama_model& model) {
    // Check if model architecture is supported
    const auto& hparams = model.hparams;
    
    // We support standard transformer architectures
    if (hparams.n_layer > 0 && hparams.n_embd > 0) {
        return true;
    }
    
    return false;
}

// Enhanced layer analysis implementations
LayerComputePattern LayerDistributor::analyze_layer_compute_pattern(const LayerProfile& layer) {
    // Analyze layer type and characteristics
    if (layer.layer_type == "embedding") {
        return LayerComputePattern::EMBEDDING_LOOKUP;
    }
    
    // Calculate arithmetic intensity
    float arithmetic_intensity = layer.compute_intensity;
    
    if (layer.layer_type == "attention" || layer.layer_type == "self_attention") {
        // Attention has specific memory access patterns
        return LayerComputePattern::ATTENTION_PATTERN;
    } else if (layer.layer_type == "feedforward" || layer.layer_type == "mlp") {
        // FFN layers are typically GEMM-heavy
        if (arithmetic_intensity > 50.0f) {
            return LayerComputePattern::GEMM_HEAVY;
        }
    }
    
    // Check if memory-bound based on arithmetic intensity
    if (arithmetic_intensity < 10.0f) {
        return LayerComputePattern::MEMORY_BOUND;
    }
    
    return LayerComputePattern::MIXED;
}

float LayerDistributor::calculate_layer_memory_bandwidth_requirement(
    const LayerProfile& layer,
    size_t sequence_length) {
    
    // Base calculation from layer profile
    float base_bandwidth = layer.memory_bandwidth_requirement_gbps;
    
    // Adjust for actual sequence length
    float seq_factor = sequence_length / 2048.0f;  // Normalized to 2K sequence length
    
    // Different patterns have different bandwidth requirements
    LayerComputePattern pattern = analyze_layer_compute_pattern(layer);
    
    switch (pattern) {
        case LayerComputePattern::ATTENTION_PATTERN:
            // Attention bandwidth scales quadratically with sequence length
            base_bandwidth *= (seq_factor * seq_factor);
            break;
            
        case LayerComputePattern::EMBEDDING_LOOKUP:
            // Embedding is mostly random access
            base_bandwidth *= 1.5f;  // Account for poor cache utilization
            break;
            
        case LayerComputePattern::GEMM_HEAVY:
            // GEMM has good memory access patterns
            base_bandwidth *= 0.8f;  // Better cache utilization
            break;
            
        default:
            base_bandwidth *= seq_factor;
    }
    
    return base_bandwidth;
}

bool LayerDistributor::can_fuse_layers(
    const LayerProfile& layer1,
    const LayerProfile& layer2,
    const GpuProfile& gpu) {
    
    if (!arch_analyzer_) {
        return false;
    }
    
    // Get architecture capabilities
    auto arch_caps = arch_analyzer_->analyze_architecture(gpu);
    
    // Use architecture analyzer for fusion decision
    return arch_analyzer_->should_fuse_layers(layer1, layer2, arch_caps);
}

float LayerDistributor::predict_layer_execution_time(
    const LayerProfile& layer,
    const GpuProfile& gpu,
    const ArchitectureCapabilities& arch_caps) {
    
    if (!arch_analyzer_) {
        // Fallback calculation
        float flops = layer.compute_intensity * layer.memory_requirement_bytes;
        return (flops / 1e9f) / gpu.theoretical_tflops;  // milliseconds
    }
    
    // Get estimated performance for this layer on this GPU
    float effective_tflops = arch_analyzer_->estimate_layer_performance(gpu, layer, arch_caps);
    
    // Calculate compute time
    float compute_flops = layer.compute_intensity * layer.memory_requirement_bytes;
    float compute_time_ms = (compute_flops / 1e12f) / effective_tflops * 1000.0f;
    
    // Calculate memory transfer time
    float memory_time_ms = 0.0f;
    LayerComputePattern pattern = analyze_layer_compute_pattern(layer);
    
    if (pattern == LayerComputePattern::MEMORY_BOUND || pattern == LayerComputePattern::EMBEDDING_LOOKUP) {
        float bandwidth_gb = gpu.memory_bandwidth_gbps * arch_caps.memory_bandwidth_efficiency;
        memory_time_ms = (layer.memory_requirement_bytes / 1e9f) / bandwidth_gb * 1000.0f;
    }
    
    // Return the maximum of compute and memory time
    return std::max(compute_time_ms, memory_time_ms);
}

} // namespace orchestration
} // namespace llama