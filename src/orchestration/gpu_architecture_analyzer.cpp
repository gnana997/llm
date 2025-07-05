#include "gpu_architecture_analyzer.h"
#include "gpu_profiler.h"  // For GpuProfile and LayerProfile
#include "llama-impl.h"
#include <algorithm>
#include <cmath>
#include <regex>

namespace llama {
namespace orchestration {

GpuArchitectureAnalyzer::GpuArchitectureAnalyzer() {
    initialize_architecture_knowledge();
}

GpuArchitectureAnalyzer::~GpuArchitectureAnalyzer() = default;

void GpuArchitectureAnalyzer::initialize_architecture_knowledge() {
    // NVIDIA Architectures
    ArchitectureCapabilities volta_caps;
    volta_caps.has_tensor_cores = true;
    volta_caps.tensor_core_generation = 1;
    volta_caps.memory_bandwidth_efficiency = 0.75f;
    volta_caps.supports_async_copy = false;
    volta_caps.optimal_gemm_tile_size = 128;
    volta_caps.gemm_efficiency = 0.80f;
    volta_caps.attention_efficiency = 0.70f;
    volta_caps.tdp_watts = 300.0f;
    volta_caps.efficiency_per_watt = 0.05f;
    knowledge_base_.known_architectures["Volta"] = volta_caps;
    
    ArchitectureCapabilities turing_caps;
    turing_caps.has_tensor_cores = true;
    turing_caps.tensor_core_generation = 2;
    turing_caps.memory_bandwidth_efficiency = 0.78f;
    turing_caps.supports_async_copy = false;
    turing_caps.optimal_gemm_tile_size = 128;
    turing_caps.gemm_efficiency = 0.82f;
    turing_caps.attention_efficiency = 0.72f;
    turing_caps.tdp_watts = 280.0f;
    turing_caps.efficiency_per_watt = 0.07f;
    knowledge_base_.known_architectures["Turing"] = turing_caps;
    
    ArchitectureCapabilities ampere_caps;
    ampere_caps.has_tensor_cores = true;
    ampere_caps.tensor_core_generation = 3;
    ampere_caps.memory_bandwidth_efficiency = 0.82f;
    ampere_caps.supports_async_copy = true;
    ampere_caps.supports_memory_compression = true;
    ampere_caps.optimal_gemm_tile_size = 256;
    ampere_caps.gemm_efficiency = 0.87f;
    ampere_caps.attention_efficiency = 0.78f;
    ampere_caps.tdp_watts = 350.0f;
    ampere_caps.efficiency_per_watt = 0.10f;
    knowledge_base_.known_architectures["Ampere"] = ampere_caps;
    
    ArchitectureCapabilities hopper_caps;
    hopper_caps.has_tensor_cores = true;
    hopper_caps.tensor_core_generation = 4;
    hopper_caps.memory_bandwidth_efficiency = 0.85f;
    hopper_caps.supports_async_copy = true;
    hopper_caps.supports_memory_compression = true;
    hopper_caps.optimal_gemm_tile_size = 256;
    hopper_caps.gemm_efficiency = 0.90f;
    hopper_caps.attention_efficiency = 0.85f;
    hopper_caps.tdp_watts = 700.0f;
    hopper_caps.efficiency_per_watt = 0.14f;
    knowledge_base_.known_architectures["Hopper"] = hopper_caps;
    
    // AMD Architectures
    ArchitectureCapabilities cdna1_caps;
    cdna1_caps.has_matrix_cores = true;
    cdna1_caps.memory_bandwidth_efficiency = 0.80f;
    cdna1_caps.supports_async_copy = true;
    cdna1_caps.optimal_gemm_tile_size = 128;
    cdna1_caps.gemm_efficiency = 0.83f;
    cdna1_caps.attention_efficiency = 0.75f;
    cdna1_caps.tdp_watts = 300.0f;
    cdna1_caps.efficiency_per_watt = 0.08f;
    knowledge_base_.known_architectures["CDNA1"] = cdna1_caps;
    
    ArchitectureCapabilities cdna2_caps;
    cdna2_caps.has_matrix_cores = true;
    cdna2_caps.memory_bandwidth_efficiency = 0.83f;
    cdna2_caps.supports_async_copy = true;
    cdna2_caps.optimal_gemm_tile_size = 256;
    cdna2_caps.gemm_efficiency = 0.86f;
    cdna2_caps.attention_efficiency = 0.80f;
    cdna2_caps.tdp_watts = 500.0f;
    cdna2_caps.efficiency_per_watt = 0.12f;
    knowledge_base_.known_architectures["CDNA2"] = cdna2_caps;
    
    ArchitectureCapabilities cdna3_caps;
    cdna3_caps.has_matrix_cores = true;
    cdna3_caps.memory_bandwidth_efficiency = 0.87f;
    cdna3_caps.supports_async_copy = true;
    cdna3_caps.supports_memory_compression = true;
    cdna3_caps.optimal_gemm_tile_size = 256;
    cdna3_caps.gemm_efficiency = 0.89f;
    cdna3_caps.attention_efficiency = 0.84f;
    cdna3_caps.tdp_watts = 750.0f;
    cdna3_caps.efficiency_per_watt = 0.15f;
    knowledge_base_.known_architectures["CDNA3"] = cdna3_caps;
    
    // RDNA architectures (gaming GPUs)
    ArchitectureCapabilities rdna2_caps;
    rdna2_caps.has_matrix_cores = false;
    rdna2_caps.memory_bandwidth_efficiency = 0.75f;
    rdna2_caps.optimal_gemm_tile_size = 64;
    rdna2_caps.gemm_efficiency = 0.70f;
    rdna2_caps.attention_efficiency = 0.65f;
    rdna2_caps.tdp_watts = 300.0f;
    rdna2_caps.efficiency_per_watt = 0.05f;
    knowledge_base_.known_architectures["RDNA2"] = rdna2_caps;
    
    ArchitectureCapabilities rdna3_caps;
    rdna3_caps.has_matrix_cores = false;  // Gaming GPUs don't have matrix cores
    rdna3_caps.memory_bandwidth_efficiency = 0.78f;
    rdna3_caps.optimal_gemm_tile_size = 128;
    rdna3_caps.gemm_efficiency = 0.75f;
    rdna3_caps.attention_efficiency = 0.70f;
    rdna3_caps.tdp_watts = 350.0f;
    rdna3_caps.efficiency_per_watt = 0.07f;
    knowledge_base_.known_architectures["RDNA3"] = rdna3_caps;
}

ArchitectureCapabilities GpuArchitectureAnalyzer::analyze_architecture(const GpuProfile& gpu) {
    // Check cache first
    auto cache_key = gpu.name + "_" + gpu.backend_type;
    auto it = architecture_cache_.find(cache_key);
    if (it != architecture_cache_.end()) {
        return it->second;
    }
    
    ArchitectureCapabilities caps;
    
    // Determine GPU vendor and analyze
    if (gpu.backend_type == "CUDA" || architecture_utils::is_nvidia_gpu(gpu)) {
        caps = analyze_nvidia_architecture(gpu);
    } else if (gpu.backend_type == "ROCm" || gpu.backend_type == "HIP" || 
               architecture_utils::is_amd_gpu(gpu)) {
        caps = analyze_amd_architecture(gpu);
    } else if (architecture_utils::is_intel_gpu(gpu)) {
        caps = analyze_intel_architecture(gpu);
    } else {
        caps = analyze_generic_architecture(gpu);
    }
    
    // Cache the result
    architecture_cache_[cache_key] = caps;
    
    LLAMA_LOG_INFO("[Architecture Analyzer] GPU %s (%s): tensor_cores=%d, matrix_cores=%d, "
                   "gemm_eff=%.2f, attn_eff=%.2f, mem_bw_eff=%.2f\n",
                   gpu.name.c_str(), gpu.backend_type.c_str(),
                   caps.has_tensor_cores, caps.has_matrix_cores,
                   caps.gemm_efficiency, caps.attention_efficiency,
                   caps.memory_bandwidth_efficiency);
    
    return caps;
}

ArchitectureCapabilities GpuArchitectureAnalyzer::analyze_nvidia_architecture(const GpuProfile& gpu) {
    ArchitectureCapabilities caps;
    
    // Detect architecture from compute capability
    int cc_major = gpu.compute_capability_major;
    int cc_minor = gpu.compute_capability_minor;
    
    if (cc_major >= 9) {
        // Blackwell or newer
        caps = knowledge_base_.known_architectures["Hopper"];  // Use Hopper as baseline
        caps.tensor_core_generation = 5;
        caps.gemm_efficiency = 0.92f;
        caps.attention_efficiency = 0.88f;
    } else if (cc_major == 8 && cc_minor >= 9) {
        // Ada Lovelace
        caps = knowledge_base_.known_architectures["Hopper"];
        caps.tdp_watts = 450.0f;  // Lower TDP than H100
    } else if (cc_major == 8 && cc_minor >= 6) {
        // Ampere
        caps = knowledge_base_.known_architectures["Ampere"];
    } else if (cc_major == 8 && cc_minor == 0) {
        // A100 (special Ampere)
        caps = knowledge_base_.known_architectures["Ampere"];
        caps.memory_bandwidth_efficiency = 0.85f;  // HBM2e
        caps.tdp_watts = 400.0f;
    } else if (cc_major == 7 && cc_minor >= 5) {
        // Turing
        caps = knowledge_base_.known_architectures["Turing"];
    } else if (cc_major == 7 && cc_minor == 0) {
        // Volta
        caps = knowledge_base_.known_architectures["Volta"];
    } else {
        // Older architectures without tensor cores
        caps.has_tensor_cores = false;
        caps.gemm_efficiency = 0.65f;
        caps.attention_efficiency = 0.60f;
    }
    
    // Check for specific features
    caps.has_tensor_cores = architecture_utils::supports_nvidia_tensor_cores(cc_major, cc_minor);
    if (caps.has_tensor_cores) {
        caps.tensor_core_generation = architecture_utils::get_nvidia_tensor_core_generation(cc_major, cc_minor);
    }
    
    // Adjust for specific GPU models
    if (gpu.name.find("H100") != std::string::npos) {
        caps.memory_bandwidth_efficiency = 0.88f;  // HBM3
        caps.gemm_efficiency = 0.91f;
        caps.attention_efficiency = 0.87f;
        caps.efficiency_per_watt = 0.14f;
    } else if (gpu.name.find("A100") != std::string::npos) {
        caps.memory_bandwidth_efficiency = 0.85f;  // HBM2e
        caps.efficiency_per_watt = 0.125f;
    }
    
    return caps;
}

ArchitectureCapabilities GpuArchitectureAnalyzer::analyze_amd_architecture(const GpuProfile& gpu) {
    ArchitectureCapabilities caps;
    
    // Detect AMD architecture from name or properties
    std::string arch = architecture_utils::detect_amd_architecture(gpu);
    
    if (arch.find("gfx940") != std::string::npos || arch.find("gfx941") != std::string::npos ||
        arch.find("gfx942") != std::string::npos) {
        // MI300 series (CDNA3)
        caps = knowledge_base_.known_architectures["CDNA3"];
    } else if (arch.find("gfx90a") != std::string::npos) {
        // MI200 series (CDNA2)
        caps = knowledge_base_.known_architectures["CDNA2"];
    } else if (arch.find("gfx908") != std::string::npos) {
        // MI100 (CDNA1)
        caps = knowledge_base_.known_architectures["CDNA1"];
    } else if (arch.find("gfx1100") != std::string::npos || arch.find("gfx1101") != std::string::npos) {
        // RDNA3
        caps = knowledge_base_.known_architectures["RDNA3"];
    } else if (arch.find("gfx1030") != std::string::npos || arch.find("gfx1031") != std::string::npos) {
        // RDNA2
        caps = knowledge_base_.known_architectures["RDNA2"];
    } else {
        // Generic AMD GPU
        caps = analyze_generic_architecture(gpu);
    }
    
    // Check for matrix cores
    caps.has_matrix_cores = architecture_utils::supports_amd_matrix_cores(arch);
    
    // Specific model adjustments
    if (gpu.name.find("MI300") != std::string::npos) {
        caps.memory_bandwidth_efficiency = 0.90f;  // HBM3
        caps.gemm_efficiency = 0.90f;
        caps.efficiency_per_watt = 0.16f;
    } else if (gpu.name.find("MI250") != std::string::npos) {
        caps.memory_bandwidth_efficiency = 0.85f;  // HBM2e
        caps.efficiency_per_watt = 0.13f;
    }
    
    return caps;
}

ArchitectureCapabilities GpuArchitectureAnalyzer::analyze_intel_architecture(const GpuProfile& gpu) {
    ArchitectureCapabilities caps = analyze_generic_architecture(gpu);
    
    // Intel Arc and Data Center GPU Max series
    if (gpu.name.find("Arc") != std::string::npos || gpu.name.find("GPU Max") != std::string::npos) {
        caps.has_xmx_engines = true;
        caps.memory_bandwidth_efficiency = 0.75f;
        caps.gemm_efficiency = 0.78f;
        caps.attention_efficiency = 0.72f;
        caps.optimal_gemm_tile_size = 128;
    }
    
    return caps;
}

ArchitectureCapabilities GpuArchitectureAnalyzer::analyze_generic_architecture(const GpuProfile& gpu) {
    ArchitectureCapabilities caps;
    
    // Conservative estimates for unknown GPUs
    caps.memory_bandwidth_efficiency = 0.70f;
    caps.gemm_efficiency = 0.65f;
    caps.attention_efficiency = 0.60f;
    caps.optimal_gemm_tile_size = 64;
    caps.optimal_block_size = 128;
    
    // Estimate based on memory size
    float memory_gb = gpu.total_memory_bytes / (1024.0f * 1024.0f * 1024.0f);
    if (memory_gb >= 80.0f) {
        // Likely a high-end datacenter GPU
        caps.gemm_efficiency = 0.80f;
        caps.attention_efficiency = 0.75f;
        caps.tdp_watts = 500.0f;
    } else if (memory_gb >= 40.0f) {
        // High-end consumer/prosumer
        caps.gemm_efficiency = 0.75f;
        caps.attention_efficiency = 0.70f;
        caps.tdp_watts = 350.0f;
    } else {
        caps.tdp_watts = 250.0f;
    }
    
    // Estimate efficiency per watt
    if (gpu.theoretical_tflops > 0 && caps.tdp_watts > 0) {
        caps.efficiency_per_watt = gpu.theoretical_tflops / caps.tdp_watts;
    }
    
    return caps;
}

LayerAffinityScore GpuArchitectureAnalyzer::calculate_layer_affinity(
    const GpuProfile& gpu,
    const LayerProfile& layer,
    const ArchitectureCapabilities& arch_caps) {
    
    LayerAffinityScore score;
    
    // Base scores on architecture capabilities
    if (layer.layer_type == "attention" || layer.layer_type == "self_attention") {
        // Attention layers benefit from tensor/matrix cores
        if (arch_caps.has_tensor_cores || arch_caps.has_matrix_cores) {
            score.attention_score = 1.2f + (arch_caps.tensor_core_generation * 0.05f);
        } else {
            score.attention_score = 0.8f;
        }
        
        // Attention efficiency
        score.attention_score *= arch_caps.attention_efficiency;
        
        // Memory bandwidth is important for attention
        if (gpu.memory_bandwidth_gbps > 1000.0f) {
            score.attention_score *= 1.1f;
        }
        
    } else if (layer.layer_type == "feedforward" || layer.layer_type == "mlp") {
        // Feedforward layers are compute-heavy
        if (arch_caps.has_tensor_cores || arch_caps.has_matrix_cores) {
            score.feedforward_score = 1.3f + (arch_caps.tensor_core_generation * 0.08f);
        } else {
            score.feedforward_score = 0.9f;
        }
        
        // GEMM efficiency is critical
        score.feedforward_score *= arch_caps.gemm_efficiency;
        
    } else if (layer.layer_type == "embedding") {
        // Embedding layers are memory-bound
        score.embedding_score = arch_caps.memory_bandwidth_efficiency;
        
        // Large embeddings benefit from high memory capacity
        if (layer.memory_requirement_bytes > 1024 * 1024 * 1024) {  // > 1GB
            float memory_ratio = gpu.available_memory_bytes / (float)layer.memory_requirement_bytes;
            score.embedding_score *= std::min(1.5f, memory_ratio / 2.0f);
        }
    }
    
    // Analyze if layer is compute or memory bound
    LayerBottleneck bottleneck = analyze_layer_bottleneck(layer, arch_caps);
    
    // Adjust scores based on bottleneck analysis
    if (bottleneck == LayerBottleneck::MEMORY_BOUND) {
        float bandwidth_factor = gpu.memory_bandwidth_gbps / 1000.0f;  // Normalize to TB/s
        score.attention_score *= (0.8f + 0.2f * bandwidth_factor);
        score.feedforward_score *= (0.9f + 0.1f * bandwidth_factor);
        score.embedding_score *= (0.7f + 0.3f * bandwidth_factor);
    } else if (bottleneck == LayerBottleneck::COMPUTE_BOUND) {
        float compute_factor = gpu.theoretical_tflops / 100.0f;  // Normalize to 100 TFLOPS
        score.attention_score *= (0.9f + 0.1f * compute_factor);
        score.feedforward_score *= (0.8f + 0.2f * compute_factor);
    }
    
    // Power efficiency considerations
    if (arch_caps.efficiency_per_watt > 0.1f) {
        float efficiency_bonus = 1.0f + (arch_caps.efficiency_per_watt - 0.1f);
        score.attention_score *= efficiency_bonus;
        score.feedforward_score *= efficiency_bonus;
        score.embedding_score *= efficiency_bonus;
    }
    
    return score;
}

GpuArchitectureAnalyzer::LayerOptimalConfig GpuArchitectureAnalyzer::get_optimal_layer_config(
    const GpuProfile& gpu [[maybe_unused]],
    const LayerProfile& layer,
    const ArchitectureCapabilities& arch_caps) {
    
    LayerOptimalConfig config;
    
    // Set optimal block and tile sizes
    config.block_size = arch_caps.optimal_block_size;
    config.tile_size = arch_caps.optimal_gemm_tile_size;
    
    // Determine if we should use tensor cores
    config.use_tensor_cores = should_use_tensor_cores(layer, arch_caps);
    
    // Async copy for large layers
    config.use_async_copy = arch_caps.supports_async_copy && 
                           layer.memory_requirement_bytes > 100 * 1024 * 1024;  // > 100MB
    
    // Select kernel variant
    if (config.use_tensor_cores && arch_caps.has_tensor_cores) {
        config.kernel_variant = "tensor_core";
        if (arch_caps.tensor_core_generation >= 3) {
            config.kernel_variant += "_ampere";  // Use optimized Ampere+ kernels
        }
    } else if (arch_caps.has_matrix_cores) {
        config.kernel_variant = "matrix_core";
    } else if (arch_caps.has_xmx_engines) {
        config.kernel_variant = "xmx";
    } else {
        config.kernel_variant = "standard";
    }
    
    // Adjust for layer type
    if (layer.layer_type == "attention") {
        // Attention needs different tile sizes for Q, K, V
        config.tile_size = std::min(config.tile_size, 64);
    } else if (layer.layer_type == "embedding") {
        // Embeddings don't benefit from tensor cores
        config.use_tensor_cores = false;
        config.kernel_variant = "standard";
    }
    
    return config;
}

float GpuArchitectureAnalyzer::estimate_layer_performance(
    const GpuProfile& gpu,
    const LayerProfile& layer,
    const ArchitectureCapabilities& arch_caps) {
    
    // Base performance from theoretical TFLOPS
    float base_tflops = gpu.theoretical_tflops;
    
    // Apply architecture efficiency
    float efficiency = 1.0f;
    if (layer.layer_type == "attention") {
        efficiency = arch_caps.attention_efficiency;
    } else if (layer.layer_type == "feedforward") {
        efficiency = arch_caps.gemm_efficiency;
    } else {
        efficiency = 0.7f;  // Conservative for other layers
    }
    
    // Apply tensor core speedup if applicable
    if (should_use_tensor_cores(layer, arch_caps)) {
        float tc_speedup = calculate_tensor_core_efficiency(layer, arch_caps.tensor_core_generation);
        efficiency *= tc_speedup;
    }
    
    // Consider memory bandwidth limitations
    LayerBottleneck bottleneck = analyze_layer_bottleneck(layer, arch_caps);
    if (bottleneck == LayerBottleneck::MEMORY_BOUND) {
        // Performance limited by memory bandwidth
        float required_bandwidth = layer.memory_bandwidth_requirement_gbps;
        float available_bandwidth = gpu.memory_bandwidth_gbps * arch_caps.memory_bandwidth_efficiency;
        float bandwidth_limit = available_bandwidth / required_bandwidth;
        efficiency *= std::min(1.0f, bandwidth_limit);
    }
    
    return base_tflops * efficiency;
}

float GpuArchitectureAnalyzer::calculate_communication_cost(
    const GpuProfile& src_gpu,
    const GpuProfile& dst_gpu,
    size_t data_size_bytes) {
    
    // Same GPU, no transfer needed
    if (src_gpu.device_id == dst_gpu.device_id) {
        return 0.0f;
    }
    
    // Detect interconnect type
    auto interconnect = architecture_utils::detect_interconnect(src_gpu, dst_gpu);
    float bandwidth = architecture_utils::get_interconnect_bandwidth(interconnect);
    
    // Calculate transfer time in milliseconds
    float transfer_time_ms = (data_size_bytes / (bandwidth * 1024.0f * 1024.0f * 1024.0f)) * 1000.0f;
    
    // Add latency overhead
    float latency_ms = 0.0f;
    switch (interconnect) {
        case architecture_utils::GpuInterconnect::NVLINK:
            latency_ms = 0.01f;  // Very low latency
            break;
        case architecture_utils::GpuInterconnect::INFINITY_FABRIC:
            latency_ms = 0.02f;
            break;
        case architecture_utils::GpuInterconnect::PCIE:
            latency_ms = 0.1f;   // Higher latency
            break;
        default:
            latency_ms = 0.15f;  // Conservative estimate
    }
    
    return transfer_time_ms + latency_ms;
}

bool GpuArchitectureAnalyzer::should_use_tensor_cores(
    const LayerProfile& layer,
    const ArchitectureCapabilities& arch_caps) {
    
    if (!arch_caps.has_tensor_cores && !arch_caps.has_matrix_cores) {
        return false;
    }
    
    // Check if layer type benefits from tensor cores
    if (layer.layer_type != "attention" && layer.layer_type != "feedforward") {
        return false;
    }
    
    // Check minimum size requirements
    // Tensor cores need sufficiently large matrices
    if (layer.memory_requirement_bytes < 10 * 1024 * 1024) {  // < 10MB
        return false;
    }
    
    // Check compute intensity
    if (layer.compute_intensity < 10.0f) {  // Low compute intensity
        return false;
    }
    
    return true;
}

bool GpuArchitectureAnalyzer::should_fuse_layers(
    const LayerProfile& layer1,
    const LayerProfile& layer2,
    const ArchitectureCapabilities& arch_caps) {
    
    // Check if layers are consecutive and compatible
    if (std::abs(layer1.layer_index - layer2.layer_index) != 1) {
        return false;
    }
    
    // Common fusion patterns
    if (layer1.layer_type == "attention" && layer2.layer_type == "feedforward") {
        // Can fuse attention output with FFN input
        return arch_caps.supports_async_copy;
    }
    
    // Check memory requirements
    size_t combined_memory = layer1.memory_requirement_bytes + layer2.memory_requirement_bytes;
    if (combined_memory > static_cast<size_t>(arch_caps.optimal_block_size) * 1024 * 1024) {
        return false;  // Too large to fuse efficiently
    }
    
    return false;
}

GpuArchitectureAnalyzer::LayerBottleneck GpuArchitectureAnalyzer::analyze_layer_bottleneck(
    const LayerProfile& layer,
    const ArchitectureCapabilities& arch_caps) {
    
    // Calculate arithmetic intensity (FLOPs per byte)
    float arithmetic_intensity = layer.compute_intensity;
    
    // Architecture-specific thresholds
    float compute_threshold = 50.0f;   // FLOPs/byte for compute bound
    float memory_threshold = 10.0f;    // FLOPs/byte for memory bound
    
    // Adjust thresholds based on architecture
    if (arch_caps.has_tensor_cores || arch_caps.has_matrix_cores) {
        compute_threshold *= 2.0f;  // Higher compute capability
    }
    
    if (arithmetic_intensity > compute_threshold) {
        return LayerBottleneck::COMPUTE_BOUND;
    } else if (arithmetic_intensity < memory_threshold) {
        return LayerBottleneck::MEMORY_BOUND;
    } else {
        return LayerBottleneck::BALANCED;
    }
}

float GpuArchitectureAnalyzer::calculate_tensor_core_efficiency(
    const LayerProfile& layer,
    int tensor_core_generation) {
    
    float base_speedup = 1.0f;
    
    // Generation-specific speedups
    switch (tensor_core_generation) {
        case 1:  // Volta
            base_speedup = 4.0f;
            break;
        case 2:  // Turing
            base_speedup = 6.0f;
            break;
        case 3:  // Ampere
            base_speedup = 8.0f;
            break;
        case 4:  // Hopper
            base_speedup = 12.0f;
            break;
        case 5:  // Blackwell+
            base_speedup = 16.0f;
            break;
        default:
            base_speedup = 1.0f;
    }
    
    // Adjust based on layer characteristics
    if (layer.layer_type == "feedforward") {
        // FFN layers get full benefit
        return base_speedup * 0.9f;  // 90% efficiency
    } else if (layer.layer_type == "attention") {
        // Attention is more complex, lower efficiency
        return base_speedup * 0.7f;  // 70% efficiency
    }
    
    return base_speedup * 0.5f;  // Conservative default
}

float GpuArchitectureAnalyzer::calculate_matrix_core_efficiency(
    const LayerProfile& layer,
    const std::string& amd_architecture) {
    
    float base_speedup = 1.0f;
    
    if (amd_architecture.find("CDNA3") != std::string::npos) {
        base_speedup = 10.0f;  // MI300 series
    } else if (amd_architecture.find("CDNA2") != std::string::npos) {
        base_speedup = 8.0f;   // MI200 series
    } else if (amd_architecture.find("CDNA1") != std::string::npos) {
        base_speedup = 6.0f;   // MI100
    }
    
    // Similar efficiency adjustments as tensor cores
    if (layer.layer_type == "feedforward") {
        return base_speedup * 0.85f;
    } else if (layer.layer_type == "attention") {
        return base_speedup * 0.65f;
    }
    
    return base_speedup * 0.5f;
}

std::string GpuArchitectureAnalyzer::get_architecture_name(const GpuProfile& gpu) {
    if (gpu.backend_type == "CUDA") {
        int cc_major = gpu.compute_capability_major;
        int cc_minor = gpu.compute_capability_minor;
        
        if (cc_major >= 9) return "Blackwell";
        if (cc_major == 8 && cc_minor >= 9) return "Ada Lovelace";
        if (cc_major == 8 && cc_minor >= 6) return "Ampere";
        if (cc_major == 8 && cc_minor == 0) return "Ampere (A100)";
        if (cc_major == 7 && cc_minor >= 5) return "Turing";
        if (cc_major == 7 && cc_minor == 0) return "Volta";
        return "Pre-Volta";
    } else if (gpu.backend_type == "ROCm" || gpu.backend_type == "HIP") {
        std::string arch = architecture_utils::detect_amd_architecture(gpu);
        if (arch.find("gfx94") != std::string::npos) return "CDNA3";
        if (arch.find("gfx90a") != std::string::npos) return "CDNA2";
        if (arch.find("gfx908") != std::string::npos) return "CDNA1";
        if (arch.find("gfx11") != std::string::npos) return "RDNA3";
        if (arch.find("gfx103") != std::string::npos) return "RDNA2";
        return arch;
    }
    
    return "Unknown";
}

int GpuArchitectureAnalyzer::get_architecture_generation(const GpuProfile& gpu) {
    if (gpu.backend_type == "CUDA") {
        return gpu.compute_capability_major;
    } else if (gpu.backend_type == "ROCm") {
        std::string arch = architecture_utils::detect_amd_architecture(gpu);
        if (arch.find("gfx94") != std::string::npos) return 3;  // CDNA3
        if (arch.find("gfx90a") != std::string::npos) return 2;  // CDNA2
        if (arch.find("gfx908") != std::string::npos) return 1;  // CDNA1
    }
    return 0;
}

// Architecture utility functions
namespace architecture_utils {

bool is_nvidia_gpu(const GpuProfile& gpu) {
    return gpu.backend_type == "CUDA" || 
           gpu.name.find("NVIDIA") != std::string::npos ||
           gpu.name.find("GeForce") != std::string::npos ||
           gpu.name.find("RTX") != std::string::npos ||
           gpu.name.find("GTX") != std::string::npos ||
           gpu.name.find("Tesla") != std::string::npos ||
           gpu.name.find("Quadro") != std::string::npos;
}

bool supports_nvidia_tensor_cores(int compute_major, int compute_minor [[maybe_unused]]) {
    return compute_major >= 7;  // Volta and newer
}

int get_nvidia_tensor_core_generation(int compute_major, int compute_minor) {
    if (compute_major >= 9) return 5;      // Blackwell
    if (compute_major == 8) {
        if (compute_minor >= 9) return 4;  // Ada Lovelace (consumer Hopper)
        if (compute_minor >= 6) return 3;  // Ampere
        if (compute_minor == 0) return 4;  // Hopper (H100)
    }
    if (compute_major == 7) {
        if (compute_minor >= 5) return 2;  // Turing
        if (compute_minor == 0) return 1;  // Volta
    }
    return 0;
}

bool is_amd_gpu(const GpuProfile& gpu) {
    return gpu.backend_type == "ROCm" || 
           gpu.backend_type == "HIP" ||
           gpu.name.find("AMD") != std::string::npos ||
           gpu.name.find("Radeon") != std::string::npos ||
           gpu.name.find("MI") != std::string::npos;
}

bool supports_amd_matrix_cores(const std::string& architecture) {
    // Only CDNA architectures have matrix cores
    return architecture.find("gfx908") != std::string::npos ||  // MI100
           architecture.find("gfx90a") != std::string::npos ||  // MI200
           architecture.find("gfx940") != std::string::npos ||  // MI300
           architecture.find("gfx941") != std::string::npos ||
           architecture.find("gfx942") != std::string::npos;
}

std::string detect_amd_architecture(const GpuProfile& gpu) {
    // Try to extract architecture from GPU name
    std::regex gfx_regex("gfx[0-9a-f]+");
    std::smatch match;
    
    if (std::regex_search(gpu.name, match, gfx_regex)) {
        return match.str();
    }
    
    // Fallback to known GPU models
    if (gpu.name.find("MI300") != std::string::npos) return "gfx942";
    if (gpu.name.find("MI250") != std::string::npos) return "gfx90a";
    if (gpu.name.find("MI200") != std::string::npos) return "gfx90a";
    if (gpu.name.find("MI100") != std::string::npos) return "gfx908";
    if (gpu.name.find("RX 7900") != std::string::npos) return "gfx1100";
    if (gpu.name.find("RX 6900") != std::string::npos) return "gfx1030";
    
    return "unknown";
}

bool is_intel_gpu(const GpuProfile& gpu) {
    return gpu.name.find("Intel") != std::string::npos ||
           gpu.name.find("Arc") != std::string::npos ||
           gpu.name.find("Xe") != std::string::npos ||
           gpu.name.find("GPU Max") != std::string::npos;
}

bool supports_intel_xmx(const std::string& architecture) {
    return architecture.find("Arc") != std::string::npos ||
           architecture.find("Xe-HP") != std::string::npos ||
           architecture.find("Xe-HPC") != std::string::npos;
}

GpuInterconnect detect_interconnect(const GpuProfile& gpu1, const GpuProfile& gpu2) {
    // Same backend type increases chance of specialized interconnect
    if (gpu1.backend_type != gpu2.backend_type) {
        return GpuInterconnect::PCIE;  // Cross-vendor always uses PCIe
    }
    
    if (gpu1.backend_type == "CUDA") {
        // Check for NVLink support (simplified)
        if ((gpu1.compute_capability_major >= 7 && gpu2.compute_capability_major >= 7) &&
            (gpu1.name.find("V100") != std::string::npos ||
             gpu1.name.find("A100") != std::string::npos ||
             gpu1.name.find("H100") != std::string::npos)) {
            return GpuInterconnect::NVLINK;
        }
    } else if (gpu1.backend_type == "ROCm" || gpu1.backend_type == "HIP") {
        // Check for Infinity Fabric
        if (gpu1.name.find("MI") != std::string::npos &&
            gpu2.name.find("MI") != std::string::npos) {
            return GpuInterconnect::INFINITY_FABRIC;
        }
    }
    
    return GpuInterconnect::PCIE;
}

float get_interconnect_bandwidth(GpuInterconnect type) {
    switch (type) {
        case GpuInterconnect::NVLINK:
            return 600.0f;  // GB/s for NVLink 3.0
        case GpuInterconnect::INFINITY_FABRIC:
            return 400.0f;  // GB/s for Infinity Fabric 3.0
        case GpuInterconnect::XE_LINK:
            return 300.0f;  // GB/s estimate
        case GpuInterconnect::PCIE:
            return 32.0f;   // GB/s for PCIe 4.0 x16
        default:
            return 16.0f;   // Conservative estimate
    }
}

} // namespace architecture_utils

} // namespace orchestration
} // namespace llama