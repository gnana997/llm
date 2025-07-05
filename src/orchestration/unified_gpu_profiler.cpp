#include "unified_gpu_profiler.h"
#include "gpu_profiler_factory.h"
#include "llama-impl.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace llama {
namespace orchestration {

UnifiedGpuProfiler::UnifiedGpuProfiler() {
    LLAMA_LOG_INFO("[Unified GPU Profiler] Initializing...\n");
    
    // Ensure GGML backends are loaded
    ggml_backend_load_all();
    
    // Build device mappings
    build_device_mappings();
}

void UnifiedGpuProfiler::build_device_mappings() {
    std::lock_guard<std::mutex> lock(mappings_mutex_);
    
    if (mappings_built_) {
        return;
    }
    
    device_mappings_.clear();
    detected_backends_.clear();
    
    size_t total_devices = ggml_backend_dev_count();
    int global_gpu_index = 0;
    
    // Map of backend -> local device index
    std::unordered_map<std::string, int> backend_device_counts;
    
    for (size_t i = 0; i < total_devices; ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (!device) continue;
        
        // Get device properties
        struct ggml_backend_dev_props props;
        ggml_backend_dev_get_props(device, &props);
        
        // Only interested in GPU devices
        if (props.type != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        
        // Get backend type
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        if (!reg) continue;
        
        const char* backend_name = ggml_backend_reg_name(reg);
        if (!backend_name) continue;
        
        std::string backend_str(backend_name);
        detected_backends_.insert(backend_str);
        
        // Create mapping
        DeviceMapping mapping;
        mapping.backend_name = backend_str;
        mapping.backend_device_id = backend_device_counts[backend_str]++;
        mapping.ggml_device = device;
        
        device_mappings_.push_back(mapping);
        
        LLAMA_LOG_INFO("[Unified GPU Profiler] Mapped device %d -> %s GPU %d\n",
                       global_gpu_index, backend_str.c_str(), mapping.backend_device_id);
        
        global_gpu_index++;
    }
    
    mappings_built_ = true;
    
    // Log summary
    LLAMA_LOG_INFO("[Unified GPU Profiler] Found %zu GPU(s) across %zu backend(s):\n",
                   device_mappings_.size(), detected_backends_.size());
    for (const auto& backend : detected_backends_) {
        LLAMA_LOG_INFO("  %s: %d device(s)\n", 
                       backend.c_str(), backend_device_counts[backend]);
    }
}

ProfilingResult UnifiedGpuProfiler::profile_all_gpus() {
    ProfilingResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Ensure mappings are built
    build_device_mappings();
    
    std::lock_guard<std::mutex> lock(mappings_mutex_);
    
    if (device_mappings_.empty()) {
        result.success = false;
        result.error_message = "No GPU devices found";
        return result;
    }
    
    // Profile each device
    for (size_t i = 0; i < device_mappings_.size(); ++i) {
        try {
            GpuProfile profile = profile_gpu(static_cast<int>(i));
            result.gpu_profiles.push_back(profile);
        } catch (const std::exception& e) {
            LLAMA_LOG_INFO("[Unified GPU Profiler] Failed to profile device %zu: %s\n",
                          i, e.what());
        }
    }
    
    result.success = !result.gpu_profiles.empty();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.profiling_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    
    // Log summary
    LLAMA_LOG_INFO("\n[Unified GPU Profiler] Profiling Summary:\n");
    LLAMA_LOG_INFO("  Total GPUs profiled: %zu\n", result.gpu_profiles.size());
    LLAMA_LOG_INFO("  Profiling duration: %ld us\n", result.profiling_duration.count());
    
    if (is_heterogeneous_setup()) {
        LLAMA_LOG_INFO("  Heterogeneous GPU setup detected\n");
        
        auto profiles_by_backend = get_profiles_by_backend();
        for (const auto& [backend, profiles] : profiles_by_backend) {
            LLAMA_LOG_INFO("    %s: %zu GPU(s)\n", backend.c_str(), profiles.size());
        }
    }
    
    return result;
}

GpuProfile UnifiedGpuProfiler::profile_gpu(int device_id) {
    // Check cache first
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = profile_cache_.find(device_id);
        if (it != profile_cache_.end()) {
            return it->second;
        }
    }
    
    // Get device mapping
    DeviceMapping mapping = get_device_mapping(device_id);
    
    // Get or create backend profiler
    auto& profiler = get_or_create_profiler(mapping.backend_name);
    
    // Profile using backend-specific profiler
    GpuProfile backend_profile = profiler.profile_gpu(mapping.backend_device_id);
    
    // Convert to global profile
    GpuProfile global_profile = convert_to_global_profile(
        backend_profile, mapping.backend_name, device_id);
    
    // Cache result
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        profile_cache_[device_id] = global_profile;
    }
    
    return global_profile;
}

bool UnifiedGpuProfiler::multi_gpu_supported() const {
    std::lock_guard<std::mutex> lock(mappings_mutex_);
    return device_mappings_.size() > 1;
}

float UnifiedGpuProfiler::benchmark_gpu_performance(int device_id) {
    DeviceMapping mapping = get_device_mapping(device_id);
    auto& profiler = get_or_create_profiler(mapping.backend_name);
    return profiler.benchmark_gpu_performance(mapping.backend_device_id);
}

LayerProfile UnifiedGpuProfiler::estimate_layer_profile(
    const std::string& layer_type,
    int layer_index,
    size_t hidden_size,
    size_t sequence_length,
    size_t vocab_size) {
    
    // Use the first available backend profiler
    // In the future, this could be made backend-specific
    if (!backend_profilers_.empty()) {
        return backend_profilers_.begin()->second->estimate_layer_profile(
            layer_type, layer_index, hidden_size, sequence_length, vocab_size);
    }
    
    // Fallback to creating a generic profiler
    auto profiler = GpuProfilerFactory::create_for_backend("Generic");
    return profiler->estimate_layer_profile(
        layer_type, layer_index, hidden_size, sequence_length, vocab_size);
}

float UnifiedGpuProfiler::calculate_gpu_score(const GpuProfile& profile) const {
    // Delegate to first available profiler or use default calculation
    if (!backend_profilers_.empty()) {
        return backend_profilers_.begin()->second->calculate_gpu_score(profile);
    }
    
    // Default calculation (same as base class)
    return profile.get_capability_score();
}

float UnifiedGpuProfiler::calculate_relative_performance(
    const GpuProfile& gpu1, 
    const GpuProfile& gpu2) const {
    
    if (gpu2.measured_performance_score == 0) return 1.0f;
    return gpu1.measured_performance_score / gpu2.measured_performance_score;
}

bool UnifiedGpuProfiler::supports_feature(const std::string& feature_name) const {
    if (feature_name == "multi_backend") return true;
    if (feature_name == "heterogeneous_gpu") return true;
    
    // Check if any backend supports the feature
    std::lock_guard<std::mutex> lock(profilers_mutex_);
    for (const auto& [backend, profiler] : backend_profilers_) {
        if (profiler->supports_feature(feature_name)) {
            return true;
        }
    }
    
    return false;
}

std::unordered_map<std::string, std::string> UnifiedGpuProfiler::get_capabilities() const {
    std::unordered_map<std::string, std::string> caps;
    
    caps["backend"] = "Unified";
    caps["multi_backend_support"] = "true";
    caps["detected_backends"] = std::accumulate(
        detected_backends_.begin(), detected_backends_.end(), std::string(),
        [](const std::string& a, const std::string& b) {
            return a.empty() ? b : a + ", " + b;
        });
    caps["total_gpus"] = std::to_string(device_mappings_.size());
    caps["heterogeneous_support"] = is_heterogeneous_setup() ? "true" : "false";
    
    return caps;
}

std::unordered_map<std::string, std::vector<GpuProfile>> 
UnifiedGpuProfiler::get_profiles_by_backend() {
    std::unordered_map<std::string, std::vector<GpuProfile>> result;
    
    // Profile all GPUs if not already done
    auto all_profiles = profile_all_gpus();
    
    // Group by backend
    for (const auto& profile : all_profiles.gpu_profiles) {
        result[profile.backend_type].push_back(profile);
    }
    
    return result;
}

size_t UnifiedGpuProfiler::get_gpu_count_for_backend(const std::string& backend_name) const {
    std::lock_guard<std::mutex> lock(mappings_mutex_);
    
    return std::count_if(device_mappings_.begin(), device_mappings_.end(),
                        [&backend_name](const DeviceMapping& mapping) {
                            return mapping.backend_name == backend_name;
                        });
}

bool UnifiedGpuProfiler::is_heterogeneous_setup() const {
    std::lock_guard<std::mutex> lock(mappings_mutex_);
    return detected_backends_.size() > 1;
}

IGpuProfiler& UnifiedGpuProfiler::get_or_create_profiler(const std::string& backend_name) {
    std::lock_guard<std::mutex> lock(profilers_mutex_);
    
    auto it = backend_profilers_.find(backend_name);
    if (it != backend_profilers_.end()) {
        return *it->second;
    }
    
    // Create new profiler for this backend
    auto profiler = GpuProfilerFactory::create_for_backend(backend_name);
    if (!profiler) {
        // Fallback to generic profiler
        profiler = GpuProfilerFactory::create_for_backend("Generic");
    }
    
    backend_profilers_[backend_name] = std::move(profiler);
    return *backend_profilers_[backend_name];
}

UnifiedGpuProfiler::DeviceMapping 
UnifiedGpuProfiler::get_device_mapping(int global_device_id) const {
    std::lock_guard<std::mutex> lock(mappings_mutex_);
    
    if (global_device_id < 0 || 
        global_device_id >= static_cast<int>(device_mappings_.size())) {
        throw std::runtime_error("Invalid device ID: " + std::to_string(global_device_id));
    }
    
    return device_mappings_[global_device_id];
}

GpuProfile UnifiedGpuProfiler::convert_to_global_profile(
    const GpuProfile& backend_profile,
    const std::string& backend_name,
    int global_device_id) const {
    
    GpuProfile global_profile = backend_profile;
    
    // Update device ID to global ID
    global_profile.device_id = global_device_id;
    
    // Ensure backend type is set
    if (global_profile.backend_type.empty()) {
        global_profile.backend_type = backend_name;
    }
    
    // Add global device identifier to name
    if (!global_profile.name.empty()) {
        global_profile.name = "[GPU " + std::to_string(global_device_id) + "] " + 
                             global_profile.name;
    }
    
    return global_profile;
}

} // namespace orchestration
} // namespace llama