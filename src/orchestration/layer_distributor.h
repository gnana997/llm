#pragma once

#include "gpu_profiler.h"
#include "gpu_architecture_analyzer.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <functional>
#include <chrono>

// Forward declarations
struct llama_model;
struct ggml_tensor;

namespace llama {
namespace orchestration {

// Distribution strategy types
enum class DistributionStrategy {
    EQUAL,      // Traditional equal distribution
    OPTIMIZED,  // Intelligent distribution based on GPU capabilities
    MANUAL      // User-specified tensor split
};

// Enhanced layer characterization
enum class LayerComputePattern {
    GEMM_HEAVY,        // Dense matrix multiplication
    MEMORY_BOUND,      // Limited by memory bandwidth
    ATTENTION_PATTERN, // Attention-specific access patterns
    EMBEDDING_LOOKUP,  // Sparse memory access
    MIXED              // Combination of patterns
};

// Layer assignment result
struct LayerAssignment {
    int layer_index;
    int gpu_id;
    float expected_load_percentage;  // Expected load on this GPU
    size_t memory_bytes;            // Memory required for this layer
    std::string rationale;          // Explanation for assignment
    
    // Enhanced fields
    LayerComputePattern compute_pattern;
    float architecture_affinity_score;  // How well suited this GPU is for this layer
    float predicted_execution_time_ms;  // Predicted execution time
    bool uses_tensor_cores;            // Whether tensor/matrix cores will be used
    std::string optimal_kernel;        // Optimal kernel variant for this layer/GPU combo
};

// Distribution result
struct DistributionResult {
    std::vector<LayerAssignment> assignments;
    std::vector<int> layers_per_gpu;     // Count of layers per GPU
    std::vector<float> memory_per_gpu;   // Memory usage per GPU (GB)
    std::vector<float> compute_load_per_gpu;  // Relative compute load per GPU (%)
    float expected_speedup;              // Expected speedup vs equal distribution
    bool cache_hit;                      // Whether result was from cache
    std::string distribution_summary;    // Human-readable summary
};

// Configuration for layer distribution
struct DistributionConfig {
    DistributionStrategy strategy = DistributionStrategy::OPTIMIZED;
    float memory_safety_margin = 0.9f;   // Use only 90% of available memory
    bool enable_caching = true;
    bool prefer_memory_balance = false;  // Balance memory over compute
    int max_layers_per_gpu = -1;         // -1 = no limit
    std::vector<float> manual_splits;    // For manual strategy
};

// Metrics tracking
struct DistributionMetrics {
    std::chrono::microseconds distribution_time;
    size_t cache_hits = 0;
    size_t cache_misses = 0;
    size_t total_distributions = 0;
    float average_speedup = 0.0f;
};

class LayerDistributor {
public:
    LayerDistributor();
    ~LayerDistributor();
    
    // Initialize with GPU profiles
    void initialize(const std::vector<GpuProfile>& gpu_profiles);
    
    // Main distribution function
    DistributionResult distribute_layers(
        const llama_model& model,
        int total_layers,
        const DistributionConfig& config = {}
    );
    
    // Profile model layers
    std::vector<LayerProfile> profile_model_layers(
        const llama_model& model,
        int total_layers
    );
    
    // Calculate optimal distribution using dynamic programming
    std::vector<int> calculate_optimal_distribution(
        const std::vector<GpuProfile>& gpus,
        const std::vector<LayerProfile>& layers,
        const DistributionConfig& config
    );
    
    // Validate distribution feasibility
    bool validate_distribution(
        const std::vector<LayerAssignment>& assignments,
        const std::vector<GpuProfile>& gpus
    );
    
    // Cache management
    void cache_distribution(
        const std::string& model_hash,
        const DistributionResult& result
    );
    
    bool get_cached_distribution(
        const std::string& model_hash,
        const std::vector<GpuProfile>& gpus,
        DistributionResult& result
    );
    
    void clear_cache();
    
    // Performance estimation
    float estimate_performance_improvement(
        const std::vector<LayerAssignment>& optimized,
        const std::vector<LayerAssignment>& baseline
    );
    
    // Logging and debugging
    void log_distribution_result(const DistributionResult& result) const;
    void log_distribution_decision(
        int layer_index,
        int gpu_id,
        const std::string& reason
    ) const;
    
    // Metrics
    DistributionMetrics get_metrics() const;
    void reset_metrics();
    
    // Strategy selection helpers
    static DistributionStrategy parse_strategy(const std::string& str);
    static std::string strategy_to_string(DistributionStrategy strategy);
    
    // Enhanced layer analysis methods
    LayerComputePattern analyze_layer_compute_pattern(const LayerProfile& layer);
    
    float calculate_layer_memory_bandwidth_requirement(
        const LayerProfile& layer,
        size_t sequence_length
    );
    
    bool can_fuse_layers(
        const LayerProfile& layer1,
        const LayerProfile& layer2,
        const GpuProfile& gpu
    );
    
    float predict_layer_execution_time(
        const LayerProfile& layer,
        const GpuProfile& gpu,
        const ArchitectureCapabilities& arch_caps
    );
    
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
    
    // Internal methods
    std::vector<LayerAssignment> distribute_equal(
        const std::vector<LayerProfile>& layers,
        const std::vector<GpuProfile>& gpus
    );
    
    std::vector<LayerAssignment> distribute_optimized(
        const std::vector<LayerProfile>& layers,
        const std::vector<GpuProfile>& gpus,
        const DistributionConfig& config
    );
    
    std::vector<LayerAssignment> distribute_manual(
        const std::vector<LayerProfile>& layers,
        const std::vector<GpuProfile>& gpus,
        const std::vector<float>& splits
    );
    
    // Dynamic programming state for optimization
    struct DPState {
        float total_cost;
        int last_gpu;
        std::vector<float> gpu_loads;
        std::vector<size_t> gpu_memory;
    };
    
    // Cost function for layer assignment
    float calculate_assignment_cost(
        const LayerProfile& layer,
        const GpuProfile& gpu,
        const DPState& current_state,
        const DistributionConfig& config
    );
    
    // Helper to generate model hash for caching
    std::string generate_model_hash(const llama_model& model) const;
    
    // Load balancing helpers
    float calculate_load_imbalance(const std::vector<float>& loads) const;
    float calculate_memory_imbalance(const std::vector<size_t>& memory) const;
    
    // GPU profiles storage
    std::vector<GpuProfile> gpu_profiles_;
    
    // Architecture analyzer
    std::unique_ptr<GpuArchitectureAnalyzer> arch_analyzer_;
    
    // Distribution cache
    struct CacheEntry {
        DistributionResult result;
        std::chrono::steady_clock::time_point timestamp;
    };
    std::unordered_map<std::string, CacheEntry> distribution_cache_;
    
    // Metrics tracking
    mutable DistributionMetrics metrics_;
    
    // Constants for optimization
    static constexpr float LOAD_BALANCE_WEIGHT = 0.4f;
    static constexpr float MEMORY_BALANCE_WEIGHT = 0.3f;
    static constexpr float EFFICIENCY_WEIGHT = 0.3f;
    static constexpr size_t CACHE_MAX_SIZE = 100;
    static constexpr int CACHE_TTL_MINUTES = 30;
};

// Helper functions
namespace distribution_utils {
    // Calculate theoretical speedup from distribution
    float calculate_theoretical_speedup(
        const std::vector<float>& gpu_capabilities,
        const std::vector<int>& layer_assignments
    );
    
    // Format distribution summary
    std::string format_distribution_summary(
        const DistributionResult& result,
        const std::vector<GpuProfile>& gpus
    );
    
    // Visualize distribution (for debugging)
    void print_distribution_visualization(
        const DistributionResult& result,
        const std::vector<GpuProfile>& gpus,
        int total_layers
    );
    
    // Analyze layer patterns
    struct LayerPattern {
        std::string pattern_type;  // "attention-heavy", "feedforward-heavy", etc.
        float attention_ratio;
        float feedforward_ratio;
        float avg_memory_per_layer;
    };
    
    LayerPattern analyze_layer_patterns(const std::vector<LayerProfile>& layers);
}

// Integration helper for llama.cpp
class LayerDistributionIntegrator {
public:
    // Apply distribution to model during loading
    static bool apply_distribution(
        llama_model& model,
        const DistributionResult& distribution
    );
    
    // Get current distribution from loaded model
    static std::vector<int> get_current_distribution(const llama_model& model);
    
    // Validate model compatibility
    static bool is_model_compatible(const llama_model& model);
};

} // namespace orchestration
} // namespace llama