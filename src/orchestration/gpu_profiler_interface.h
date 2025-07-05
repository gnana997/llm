#pragma once

#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <unordered_map>

namespace llama {
namespace orchestration {

// Forward declarations
struct GpuProfile;
struct LayerProfile;
struct ProfilingResult;

/**
 * Abstract interface for GPU profiling across different backends
 * 
 * This interface provides a unified API for profiling GPUs from various
 * vendors and APIs (CUDA, ROCm, Metal, SYCL, Vulkan, etc.)
 */
class IGpuProfiler {
public:
    virtual ~IGpuProfiler() = default;
    
    // Core profiling interface
    
    /**
     * Profile all available GPUs for this backend
     * @return Profiling results including all discovered GPUs
     */
    virtual ProfilingResult profile_all_gpus() = 0;
    
    /**
     * Profile a specific GPU by device ID
     * @param device_id Backend-specific device identifier
     * @return Profile information for the specified GPU
     */
    virtual GpuProfile profile_gpu(int device_id) = 0;
    
    /**
     * Check if this backend supports multi-GPU configurations
     * @return true if multiple GPUs can be used together
     */
    virtual bool multi_gpu_supported() const = 0;
    
    /**
     * Get the backend type name (e.g., "CUDA", "ROCm", "Metal")
     * @return Backend identifier string
     */
    virtual std::string get_backend_type() const = 0;
    
    // Performance measurement
    
    /**
     * Run performance benchmark on specified GPU
     * @param device_id Device to benchmark
     * @return Performance score (e.g., TFLOPS)
     */
    virtual float benchmark_gpu_performance(int device_id) = 0;
    
    // Layer profiling
    
    /**
     * Estimate computational requirements for a model layer
     * @param layer_type Type of layer (attention, feedforward, etc.)
     * @param layer_index Index of the layer in the model
     * @param hidden_size Hidden dimension size
     * @param sequence_length Sequence length for computation
     * @param vocab_size Vocabulary size (for embedding layers)
     * @return Estimated layer profile
     */
    virtual LayerProfile estimate_layer_profile(
        const std::string& layer_type,
        int layer_index,
        size_t hidden_size,
        size_t sequence_length,
        size_t vocab_size = 0
    ) = 0;
    
    // Utility methods
    
    /**
     * Calculate normalized GPU capability score (0-100)
     * @param profile GPU profile to score
     * @return Normalized capability score
     */
    virtual float calculate_gpu_score(const GpuProfile& profile) const = 0;
    
    /**
     * Compare relative performance between two GPUs
     * @param gpu1 First GPU profile
     * @param gpu2 Second GPU profile
     * @return Relative performance ratio (gpu1/gpu2)
     */
    virtual float calculate_relative_performance(
        const GpuProfile& gpu1, 
        const GpuProfile& gpu2
    ) const = 0;
    
    // Optional: Backend-specific feature detection
    
    /**
     * Check if the backend supports a specific feature
     * @param feature_name Name of the feature to check
     * @return true if feature is supported
     */
    virtual bool supports_feature([[maybe_unused]] const std::string& feature_name) const {
        return false;  // Default: no special features
    }
    
    /**
     * Get backend-specific capabilities as key-value pairs
     * @return Map of capability names to values
     */
    virtual std::unordered_map<std::string, std::string> get_capabilities() const {
        return {};  // Default: no special capabilities
    }
};

/**
 * Factory function type for creating backend-specific profilers
 */
using ProfilerFactory = std::unique_ptr<IGpuProfiler>(*)();

} // namespace orchestration
} // namespace llama