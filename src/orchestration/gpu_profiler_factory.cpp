#include "gpu_profiler_factory.h"
#include "cuda_gpu_profiler.h"
#include "rocm_gpu_profiler.h"
#include "generic_gpu_profiler.h"
#include "unified_gpu_profiler.h"
#include "llama-impl.h"
#include <algorithm>

namespace llama {
namespace orchestration {

// Static registry instance
GpuProfilerFactory::Registry& GpuProfilerFactory::get_registry() {
    static Registry registry;
    return registry;
}

void GpuProfilerFactory::register_backend(
    const std::string& backend_name,
    ProfilerCreator creator,
    int priority) {
    
    auto& registry = get_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    
    LLAMA_LOG_INFO("[GPU Profiler Factory] Registering backend: %s (priority: %d)\n", 
                   backend_name.c_str(), priority);
    
    registry.backends[backend_name] = {creator, priority};
}

void GpuProfilerFactory::unregister_backend(const std::string& backend_name) {
    auto& registry = get_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    
    registry.backends.erase(backend_name);
}

std::unique_ptr<IGpuProfiler> GpuProfilerFactory::create_for_backend(
    const std::string& backend_name) {
    
    auto& registry = get_registry();
    
    // Initialize backends on first use
    if (!registry.initialized) {
        initialize_backends();
    }
    
    std::lock_guard<std::mutex> lock(registry.mutex);
    
    auto it = registry.backends.find(backend_name);
    if (it != registry.backends.end()) {
        try {
            return it->second.creator();
        } catch (const std::exception& e) {
            LLAMA_LOG_INFO("[GPU Profiler Factory] Failed to create %s profiler: %s\n",
                          backend_name.c_str(), e.what());
        }
    }
    
    return nullptr;
}

std::unique_ptr<IGpuProfiler> GpuProfilerFactory::create_for_device(
    ggml_backend_dev_t device) {
    
    if (!device) {
        return nullptr;
    }
    
    // Get device backend type
    std::string backend_name = get_backend_name_from_device(device);
    
    // Try to create specific profiler
    auto profiler = create_for_backend(backend_name);
    if (profiler) {
        return profiler;
    }
    
    // Fallback to generic profiler
    LLAMA_LOG_INFO("[GPU Profiler Factory] Using generic profiler for %s backend\n",
                   backend_name.c_str());
    return std::make_unique<GenericGpuProfiler>();
}

std::unique_ptr<IGpuProfiler> GpuProfilerFactory::create_unified() {
    initialize_backends();
    return std::make_unique<UnifiedGpuProfiler>();
}

std::vector<std::string> GpuProfilerFactory::get_registered_backends() {
    auto& registry = get_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    
    std::vector<std::string> backends;
    backends.reserve(registry.backends.size());
    
    for (const auto& [name, info] : registry.backends) {
        backends.push_back(name);
    }
    
    // Sort by priority (descending)
    std::sort(backends.begin(), backends.end(),
              [&registry](const std::string& a, const std::string& b) {
                  return registry.backends[a].priority > registry.backends[b].priority;
              });
    
    return backends;
}

bool GpuProfilerFactory::is_backend_registered(const std::string& backend_name) {
    auto& registry = get_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    
    return registry.backends.find(backend_name) != registry.backends.end();
}

void GpuProfilerFactory::initialize_backends() {
    auto& registry = get_registry();
    
    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        if (registry.initialized) {
            return;
        }
        registry.initialized = true;
    }
    
    LLAMA_LOG_INFO("[GPU Profiler Factory] Initializing backends...\n");
    
    // Load all GGML backends
    ggml_backend_load_all();
    
    // Backends will self-register via static initializers
    // But we can also explicitly register them here if needed
    
#ifdef GGML_USE_CUDA
    if (!is_backend_registered("CUDA")) {
        register_backend("CUDA", []() {
            return std::make_unique<CudaGpuProfiler>();
        }, 100); // High priority for CUDA
    }
#endif

#ifdef GGML_USE_HIP
    if (!is_backend_registered("ROCm") && !is_backend_registered("HIP")) {
        register_backend("HIP", []() {
            return std::make_unique<RocmGpuProfiler>();
        }, 90); // High priority for ROCm/HIP
        
        // Also register as "ROCm" for compatibility
        register_backend("ROCm", []() {
            return std::make_unique<RocmGpuProfiler>();
        }, 90);
    }
#endif

#ifdef GGML_USE_METAL
    if (!is_backend_registered("Metal")) {
        register_backend("Metal", []() {
            // Will be implemented later
            return std::make_unique<GenericGpuProfiler>();
        }, 80);
    }
#endif

#ifdef GGML_USE_SYCL
    if (!is_backend_registered("SYCL")) {
        register_backend("SYCL", []() {
            // Will be implemented later
            return std::make_unique<GenericGpuProfiler>();
        }, 70);
    }
#endif

#ifdef GGML_USE_VULKAN
    if (!is_backend_registered("Vulkan")) {
        register_backend("Vulkan", []() {
            // Will be implemented later
            return std::make_unique<GenericGpuProfiler>();
        }, 60);
    }
#endif

    // Always register generic fallback
    if (!is_backend_registered("Generic")) {
        register_backend("Generic", []() {
            return std::make_unique<GenericGpuProfiler>();
        }, 0); // Lowest priority
    }
    
    auto backends = get_registered_backends();
    LLAMA_LOG_INFO("[GPU Profiler Factory] Registered backends: ");
    for (const auto& backend : backends) {
        LLAMA_LOG_INFO("%s ", backend.c_str());
    }
    LLAMA_LOG_INFO("\n");
}

std::string GpuProfilerFactory::get_backend_name_from_device(ggml_backend_dev_t device) {
    if (!device) {
        return "Unknown";
    }
    
    // Get backend registry
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    if (!reg) {
        return "Unknown";
    }
    
    // Get backend name
    const char* name = ggml_backend_reg_name(reg);
    return name ? name : "Unknown";
}

// Static registrars for GPU backends
#ifdef GGML_USE_CUDA
namespace {
    ProfilerRegistrar cuda_registrar("CUDA", []() {
        return std::make_unique<CudaGpuProfiler>();
    }, 100);
}
#endif

#ifdef GGML_USE_HIP
namespace {
    ProfilerRegistrar hip_registrar("HIP", []() {
        return std::make_unique<RocmGpuProfiler>();
    }, 90);
    
    ProfilerRegistrar rocm_registrar("ROCm", []() {
        return std::make_unique<RocmGpuProfiler>();
    }, 90);
}
#endif

} // namespace orchestration
} // namespace llama