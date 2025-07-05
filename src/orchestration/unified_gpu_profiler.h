#pragma once

#include "gpu_profiler_interface.h"
#include "gpu_profiler.h"  // For complete type definitions
#include "ggml-backend.h"
#include <unordered_map>
#include <mutex>
#include <set>

namespace llama {
namespace orchestration {

/**
 * Unified GPU profiler that manages multiple backend profilers
 * 
 * This class provides a single interface for profiling GPUs across
 * different backends (CUDA, ROCm, Metal, etc.). It automatically
 * detects available backends and delegates to appropriate profilers.
 */
class UnifiedGpuProfiler : public IGpuProfiler {
public:
    UnifiedGpuProfiler();
    ~UnifiedGpuProfiler() override = default;
    
    // IGpuProfiler interface implementation
    ProfilingResult profile_all_gpus() override;
    GpuProfile profile_gpu(int device_id) override;
    bool multi_gpu_supported() const override;
    std::string get_backend_type() const override { return "Unified"; }
    
    // Performance measurement
    float benchmark_gpu_performance(int device_id) override;
    
    // Layer profiling
    LayerProfile estimate_layer_profile(
        const std::string& layer_type,
        int layer_index,
        size_t hidden_size,
        size_t sequence_length,
        size_t vocab_size = 0
    ) override;
    
    // Utility methods
    float calculate_gpu_score(const GpuProfile& profile) const override;
    float calculate_relative_performance(
        const GpuProfile& gpu1, 
        const GpuProfile& gpu2
    ) const override;
    
    // Extended functionality
    bool supports_feature(const std::string& feature_name) const override;
    std::unordered_map<std::string, std::string> get_capabilities() const override;
    
    /**
     * Get profiles grouped by backend type
     */
    std::unordered_map<std::string, std::vector<GpuProfile>> get_profiles_by_backend();
    
    /**
     * Get the number of GPUs for a specific backend
     */
    size_t get_gpu_count_for_backend(const std::string& backend_name) const;
    
    /**
     * Check if heterogeneous GPU setup is detected
     */
    bool is_heterogeneous_setup() const;

private:
    // Device mapping: global device ID -> (backend_name, backend_device_id)
    struct DeviceMapping {
        std::string backend_name;
        int backend_device_id;
        ggml_backend_dev_t ggml_device;
    };
    
    // Get or create profiler for a backend
    IGpuProfiler& get_or_create_profiler(const std::string& backend_name);
    
    // Build device mappings
    void build_device_mappings();
    
    // Get mapping for global device ID
    DeviceMapping get_device_mapping(int global_device_id) const;
    
    // Convert backend-specific profile to global
    GpuProfile convert_to_global_profile(
        const GpuProfile& backend_profile,
        const std::string& backend_name,
        int global_device_id
    ) const;

private:
    // Backend profilers
    mutable std::unordered_map<std::string, std::unique_ptr<IGpuProfiler>> backend_profilers_;
    mutable std::mutex profilers_mutex_;
    
    // Device mappings
    std::vector<DeviceMapping> device_mappings_;
    mutable std::mutex mappings_mutex_;
    
    // Cached results
    mutable std::unordered_map<int, GpuProfile> profile_cache_;
    mutable std::mutex cache_mutex_;
    
    // Detected backends
    std::set<std::string> detected_backends_;
    
    // Flags
    bool mappings_built_ = false;
};

} // namespace orchestration
} // namespace llama