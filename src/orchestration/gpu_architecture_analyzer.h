#pragma once

#include "gpu_profiler_interface.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace llama {
namespace orchestration {

// Architecture-specific capabilities and optimizations
struct ArchitectureCapabilities {
    // Compute capabilities
    bool has_tensor_cores = false;
    bool has_matrix_cores = false;
    bool has_xmx_engines = false;
    int tensor_core_generation = 0;  // 1=Volta, 2=Turing, 3=Ampere, 4=Hopper, 5=Blackwell
    
    // Memory capabilities
    float memory_bandwidth_efficiency = 0.8f;  // Typical efficiency
    bool supports_async_copy = false;
    bool supports_memory_compression = false;
    
    // Optimization hints
    int optimal_gemm_tile_size = 128;
    int optimal_block_size = 256;
    float gemm_efficiency = 0.85f;  // Fraction of theoretical peak
    float attention_efficiency = 0.75f;
    
    // Power and thermal
    float tdp_watts = 350.0f;
    float efficiency_per_watt = 0.0f;  // TFLOPS/W
};

// Layer affinity scores for different architectures
struct LayerAffinityScore {
    float attention_score = 1.0f;      // How well this GPU handles attention
    float feedforward_score = 1.0f;    // How well this GPU handles FFN
    float embedding_score = 1.0f;      // How well this GPU handles embeddings
    float communication_penalty = 0.0f; // Penalty for cross-GPU communication
};

// Architecture-specific optimization rules
class GpuArchitectureAnalyzer {
public:
    GpuArchitectureAnalyzer();
    ~GpuArchitectureAnalyzer();
    
    // Analyze GPU architecture and capabilities
    ArchitectureCapabilities analyze_architecture(const GpuProfile& gpu);
    
    // Calculate layer affinity scores for a GPU
    LayerAffinityScore calculate_layer_affinity(
        const GpuProfile& gpu,
        const LayerProfile& layer,
        const ArchitectureCapabilities& arch_caps
    );
    
    // Get optimal configuration for layer on specific architecture
    struct LayerOptimalConfig {
        int block_size;
        int tile_size;
        bool use_tensor_cores;
        bool use_async_copy;
        std::string kernel_variant;  // "tensor_core", "matrix_core", "standard"
    };
    
    LayerOptimalConfig get_optimal_layer_config(
        const GpuProfile& gpu,
        const LayerProfile& layer,
        const ArchitectureCapabilities& arch_caps
    );
    
    // Estimate performance for specific layer on GPU
    float estimate_layer_performance(
        const GpuProfile& gpu,
        const LayerProfile& layer,
        const ArchitectureCapabilities& arch_caps
    );
    
    // Calculate cross-GPU communication cost
    float calculate_communication_cost(
        const GpuProfile& src_gpu,
        const GpuProfile& dst_gpu,
        size_t data_size_bytes
    );
    
    // Architecture-specific optimizations
    bool should_use_tensor_cores(
        const LayerProfile& layer,
        const ArchitectureCapabilities& arch_caps
    );
    
    bool should_fuse_layers(
        const LayerProfile& layer1,
        const LayerProfile& layer2,
        const ArchitectureCapabilities& arch_caps
    );
    
    // Memory bandwidth vs compute analysis
    enum class LayerBottleneck {
        COMPUTE_BOUND,
        MEMORY_BOUND,
        BALANCED
    };
    
    LayerBottleneck analyze_layer_bottleneck(
        const LayerProfile& layer,
        const ArchitectureCapabilities& arch_caps
    );
    
    // Get architecture name and generation
    std::string get_architecture_name(const GpuProfile& gpu);
    int get_architecture_generation(const GpuProfile& gpu);
    
private:
    // NVIDIA architecture detection
    ArchitectureCapabilities analyze_nvidia_architecture(const GpuProfile& gpu);
    
    // AMD architecture detection
    ArchitectureCapabilities analyze_amd_architecture(const GpuProfile& gpu);
    
    // Intel architecture detection
    ArchitectureCapabilities analyze_intel_architecture(const GpuProfile& gpu);
    
    // Generic/fallback architecture
    ArchitectureCapabilities analyze_generic_architecture(const GpuProfile& gpu);
    
    // Helper functions
    float calculate_tensor_core_efficiency(
        const LayerProfile& layer,
        int tensor_core_generation
    );
    
    float calculate_matrix_core_efficiency(
        const LayerProfile& layer,
        const std::string& amd_architecture
    );
    
    // Architecture knowledge base
    struct ArchitectureKnowledge {
        std::unordered_map<std::string, ArchitectureCapabilities> known_architectures;
        std::unordered_map<std::string, LayerAffinityScore> architecture_affinities;
    };
    
    void initialize_architecture_knowledge();
    
    // Cache for analyzed architectures
    mutable std::unordered_map<std::string, ArchitectureCapabilities> architecture_cache_;
    
    // Knowledge base
    ArchitectureKnowledge knowledge_base_;
};

// Helper namespace for architecture-specific calculations
namespace architecture_utils {
    // NVIDIA specific
    bool is_nvidia_gpu(const GpuProfile& gpu);
    bool supports_nvidia_tensor_cores(int compute_major, int compute_minor);
    int get_nvidia_tensor_core_generation(int compute_major, int compute_minor);
    
    // AMD specific
    bool is_amd_gpu(const GpuProfile& gpu);
    bool supports_amd_matrix_cores(const std::string& architecture);
    std::string detect_amd_architecture(const GpuProfile& gpu);
    
    // Intel specific
    bool is_intel_gpu(const GpuProfile& gpu);
    bool supports_intel_xmx(const std::string& architecture);
    
    // Communication topology
    enum class GpuInterconnect {
        PCIE,
        NVLINK,
        INFINITY_FABRIC,
        XE_LINK,
        UNKNOWN
    };
    
    GpuInterconnect detect_interconnect(const GpuProfile& gpu1, const GpuProfile& gpu2);
    float get_interconnect_bandwidth(GpuInterconnect type);
}

} // namespace orchestration
} // namespace llama