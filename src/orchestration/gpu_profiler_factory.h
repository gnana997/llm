#pragma once

#include "gpu_profiler_interface.h"
#include "ggml-backend.h"
#include <memory>
#include <functional>
#include <unordered_map>
#include <mutex>
#include <vector>

namespace llama {
namespace orchestration {

/**
 * Factory for creating GPU profilers based on backend type
 * 
 * This factory supports runtime registration of profiler implementations
 * and automatic backend detection for creating appropriate profilers.
 */
class GpuProfilerFactory {
public:
    // Function type for creating profiler instances
    using ProfilerCreator = std::function<std::unique_ptr<IGpuProfiler>()>;
    
    /**
     * Register a profiler implementation for a specific backend
     * @param backend_name Name of the backend (e.g., "CUDA", "ROCm", "Metal")
     * @param creator Function that creates profiler instances
     * @param priority Priority for automatic selection (higher = preferred)
     */
    static void register_backend(
        const std::string& backend_name,
        ProfilerCreator creator,
        int priority = 0
    );
    
    /**
     * Unregister a backend (mainly for testing)
     */
    static void unregister_backend(const std::string& backend_name);
    
    /**
     * Create a profiler for a specific backend
     * @param backend_name Name of the backend
     * @return Profiler instance or nullptr if backend not found
     */
    static std::unique_ptr<IGpuProfiler> create_for_backend(
        const std::string& backend_name
    );
    
    /**
     * Create the best available profiler for a specific device
     * @param device GGML backend device
     * @return Profiler instance or generic profiler as fallback
     */
    static std::unique_ptr<IGpuProfiler> create_for_device(
        ggml_backend_dev_t device
    );
    
    /**
     * Create a unified profiler that supports all registered backends
     * @return Unified profiler instance
     */
    static std::unique_ptr<IGpuProfiler> create_unified();
    
    /**
     * Get list of registered backend names
     */
    static std::vector<std::string> get_registered_backends();
    
    /**
     * Check if a backend is registered
     */
    static bool is_backend_registered(const std::string& backend_name);
    
    /**
     * Initialize all available backends
     * This is called automatically but can be called manually for testing
     */
    static void initialize_backends();

private:
    struct BackendInfo {
        ProfilerCreator creator;
        int priority;
    };
    
    // Registry singleton
    struct Registry {
        std::unordered_map<std::string, BackendInfo> backends;
        std::mutex mutex;
        bool initialized = false;
    };
    
    static Registry& get_registry();
    
    // Helper to get backend name from device
    static std::string get_backend_name_from_device(ggml_backend_dev_t device);
};

/**
 * RAII helper for backend registration
 * 
 * Usage:
 * namespace {
 *     ProfilerRegistrar cuda_registrar("CUDA", []() { 
 *         return std::make_unique<CudaGpuProfiler>(); 
 *     }, 100);
 * }
 */
class ProfilerRegistrar {
public:
    ProfilerRegistrar(
        const std::string& backend_name,
        GpuProfilerFactory::ProfilerCreator creator,
        int priority = 0
    ) : backend_name_(backend_name) {
        GpuProfilerFactory::register_backend(backend_name, creator, priority);
    }
    
    ~ProfilerRegistrar() {
        // Optionally unregister on destruction
        // GpuProfilerFactory::unregister_backend(backend_name_);
    }

private:
    std::string backend_name_;
};

} // namespace orchestration
} // namespace llama