// Demonstration of extended logging usage
#include "../src/log/log-ex.h"
#include "log.h"
#include <thread>
#include <chrono>

// Example function that demonstrates performance logging
void example_model_loading() {
    // Use RAII timer for automatic timing
    PERF_TIMER(PERF_EVENT_MODEL_LOAD_START, "model_7b");
    
    LOG_PERF("Starting model load for model_7b\n");
    LOG_CAT(COMMON_LOG_CAT_MODEL, GGML_LOG_LEVEL_INFO, "Loading model weights...\n");
    
    // Simulate loading
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Log memory allocation
    common_perf_event alloc_event;
    alloc_event.type = PERF_EVENT_GPU_ALLOC;
    alloc_event.timestamp = std::chrono::high_resolution_clock::now();
    alloc_event.size = 7 * 1024 * 1024 * 1024ULL; // 7GB
    alloc_event.name = "model_weights";
    alloc_event.metadata["device"] = "cuda:0";
    common_log_ex::instance().log_perf_event(alloc_event);
    
    LOG_CAT(COMMON_LOG_CAT_MEMORY | COMMON_LOG_CAT_GPU, GGML_LOG_LEVEL_INFO, 
            "Allocated 7GB on GPU for model weights\n");
}

// Example batch processing with detailed tracing
void example_batch_processing(int batch_size) {
    LOG_PERF_START(PERF_EVENT_BATCH_START, "batch_" + std::to_string(batch_size));
    
    LOG_TRACE("Processing batch of size %d\n", batch_size);
    
    for (int i = 0; i < batch_size; i++) {
        LOG_PERF_START(PERF_EVENT_TOKEN_START, "token_" + std::to_string(i));
        
        // Simulate token processing
        std::this_thread::sleep_for(std::chrono::microseconds(500));
        
        LOG_TRACEV(4, "Token %d processed\n", i);
        LOG_PERF_END(PERF_EVENT_TOKEN_END, "token_" + std::to_string(i));
        
        // Simulate cache behavior
        if (i % 4 == 0) {
            common_perf_event cache_event;
            cache_event.type = PERF_EVENT_CACHE_HIT;
            cache_event.timestamp = std::chrono::high_resolution_clock::now();
            cache_event.name = "kv_cache";
            common_log_ex::instance().log_perf_event(cache_event);
        } else {
            common_perf_event cache_event;
            cache_event.type = PERF_EVENT_CACHE_MISS;
            cache_event.timestamp = std::chrono::high_resolution_clock::now();
            cache_event.name = "kv_cache";
            common_log_ex::instance().log_perf_event(cache_event);
        }
    }
    
    LOG_PERF_END(PERF_EVENT_BATCH_END, "batch_" + std::to_string(batch_size));
    LOG_CAT(COMMON_LOG_CAT_BATCH, GGML_LOG_LEVEL_INFO, "Batch processing completed\n");
}

// Example of using the extended logging system
int main() {
    // Initialize common logging
    common_log_set_prefix(common_log_main(), true);
    common_log_set_timestamps(common_log_main(), true);
    common_log_set_colors(common_log_main(), true);
    
    // Configure extended logging
    auto& logger = common_log_ex::instance();
    
    // Set up log rotation
    common_log_rotation_config rotation;
    rotation.base_path = "llm_perf.log";
    rotation.max_file_size = 10 * 1024 * 1024; // 10MB
    rotation.max_files = 3;
    logger.set_rotation_config(rotation);
    
    // Enable JSON format for structured logging
    logger.set_format(COMMON_LOG_FORMAT_JSON);
    
    // Set category filter - only log GPU, Model, and Performance categories
    logger.set_category_filter(COMMON_LOG_CAT_GPU | COMMON_LOG_CAT_MODEL | COMMON_LOG_CAT_PERF);
    
    // Add regex filter to exclude certain messages
    logger.add_regex_filter("debug.*internal", false); // Exclude internal debug messages
    
    // Enable performance summary
    logger.enable_performance_summary(true);
    
    // Set verbosity for performance and trace logging
    common_log_set_verbosity_thold(3); // Enable TRACE level
    
    LOG_INF("Starting extended logging demonstration\n");
    
    // Demonstrate different logging features
    example_model_loading();
    
    // Process multiple batches
    for (int batch_size : {8, 16, 32}) {
        example_batch_processing(batch_size);
    }
    
    // Log some GPU operations
    LOG_CAT(COMMON_LOG_CAT_GPU, GGML_LOG_LEVEL_INFO, "GPU utilization: 85%%\n");
    LOG_PERFV(2, "Kernel execution time: 1.2ms\n");
    
    // Force log rotation
    logger.force_rotate();
    
    // Print performance summary
    logger.print_performance_summary();
    
    // Get JSON summary
    std::string json_summary = logger.get_performance_summary_json();
    LOG_INF("Performance summary (JSON): %s\n", json_summary.c_str());
    
    return 0;
}