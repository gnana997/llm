#include "metrics.h"
#include "metrics-gpu.h"
#include "metrics-aggregator.h"
#include "../src/log/log-ex.h"
#include <iostream>
#include <thread>
#include <random>
#include <cmath>

using namespace llama::metrics;

// Simulate model loading
void simulate_model_load(const std::string& model_name) {
    std::cout << "Loading model: " << model_name << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    track_model_load_start(model_name);
    
    // Simulate loading time
    std::this_thread::sleep_for(std::chrono::milliseconds(500 + rand() % 1000));
    
    // Simulate memory allocation
    size_t model_size = 1024 * 1024 * (100 + rand() % 400); // 100-500 MB
    track_memory_allocation(model_size);
    update_memory_usage(model_size);
    
    auto duration = std::chrono::high_resolution_clock::now() - start;
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    
    track_model_load_end(model_name, microseconds);
    std::cout << "Model loaded in " << microseconds / 1000 << " ms" << std::endl;
}

// Simulate token generation
void simulate_token_generation(int num_tokens) {
    std::cout << "\nGenerating " << num_tokens << " tokens..." << std::endl;
    
    track_generation_request_start();
    auto request_start = std::chrono::high_resolution_clock::now();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> token_time(50, 20); // 50ms average, 20ms stddev
    
    for (int i = 0; i < num_tokens; ++i) {
        // Simulate token generation time
        auto token_duration = std::max(10.0, token_time(gen));
        
        {
            METRIC_TIMER("token_generation_duration_us");
            std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(token_duration * 1000)));
        }
        
        track_tokens_generated(1);
        
        // Simulate cache access
        if (rand() % 100 < 70) { // 70% cache hit rate
            track_cache_hit();
        } else {
            track_cache_miss();
        }
        
        // Progress indicator
        if ((i + 1) % 10 == 0) {
            std::cout << "." << std::flush;
        }
    }
    
    auto request_duration = std::chrono::high_resolution_clock::now() - request_start;
    auto request_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(request_duration).count();
    
    track_generation_request_end(request_microseconds);
    std::cout << "\nGenerated " << num_tokens << " tokens in " 
              << request_microseconds / 1000 << " ms" << std::endl;
}

// Simulate batch processing
void simulate_batch_processing(size_t batch_size) {
    std::cout << "\nProcessing batch of size " << batch_size << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Simulate batch processing
    std::this_thread::sleep_for(std::chrono::milliseconds(100 + batch_size * 10));
    
    auto duration = std::chrono::high_resolution_clock::now() - start;
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    
    track_batch_processed(batch_size, microseconds);
}

// Simulate GPU operations
void simulate_gpu_operations() {
    std::cout << "\nSimulating GPU operations..." << std::endl;
    
    // Initialize GPU metrics
    gpu::gpu_metrics_manager::instance().initialize();
    
    // Simulate memory allocation
    {
        size_t alloc_size = 256 * 1024 * 1024; // 256 MB
        METRIC_GPU_ALLOC(0, alloc_size);
        std::cout << "GPU: Allocated " << alloc_size / (1024 * 1024) << " MB" << std::endl;
    }
    
    // Simulate H2D transfer
    {
        size_t transfer_size = 64 * 1024 * 1024; // 64 MB
        GPU_METRIC_H2D_TIMER(0, transfer_size);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::cout << "GPU: H2D transfer " << transfer_size / (1024 * 1024) << " MB" << std::endl;
    }
    
    // Simulate kernel execution
    {
        GPU_METRIC_KERNEL_TIMER(0, "matrix_multiply");
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::cout << "GPU: Executed kernel 'matrix_multiply'" << std::endl;
    }
    
    // Simulate D2H transfer
    {
        size_t transfer_size = 32 * 1024 * 1024; // 32 MB
        GPU_METRIC_D2H_TIMER(0, transfer_size);
        std::this_thread::sleep_for(std::chrono::milliseconds(8));
        std::cout << "GPU: D2H transfer " << transfer_size / (1024 * 1024) << " MB" << std::endl;
    }
    
    // Update GPU metrics
    METRIC_GPU_UPDATE();
}

// Test custom metrics
void test_custom_metrics() {
    std::cout << "\nTesting custom metrics..." << std::endl;
    
    // Register custom counter
    auto& registry = metric_registry::instance();
    auto custom_counter = registry.register_metric<counter>("demo_operations_total", 
                                                           "Total demo operations");
    
    // Register custom histogram with custom buckets
    std::vector<double> latency_buckets = {1, 5, 10, 25, 50, 100, 250, 500, 1000};
    auto custom_histogram = registry.register_metric<histogram>("demo_operation_latency_ms",
                                                               "Demo operation latency in milliseconds",
                                                               latency_buckets);
    
    // Simulate operations
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> latency_dist(100, 50);
    
    for (int i = 0; i < 100; ++i) {
        custom_counter->increment();
        
        double latency = std::max(1.0, latency_dist(gen));
        custom_histogram->observe(latency);
    }
    
    std::cout << "Performed 100 custom operations" << std::endl;
}

// Test aggregations
void test_aggregations() {
    std::cout << "\nTesting metric aggregations..." << std::endl;
    
    // Setup aggregator
    auto& aggregator = metrics_aggregator::instance();
    
    // Register custom rate
    aggregator.register_rate("demo_operations_total", "demo_operations_per_second", 
                           std::chrono::seconds(10));
    
    // Update aggregations
    aggregator.update();
    
    // Calculate statistics
    auto stats = aggregator.calculate_stats("token_generation_duration_us", std::chrono::seconds(60));
    std::cout << "Token generation stats (last 60s):" << std::endl;
    std::cout << "  Count: " << stats.count << std::endl;
    std::cout << "  Min: " << stats.min / 1000 << " ms" << std::endl;
    std::cout << "  Max: " << stats.max / 1000 << " ms" << std::endl;
    std::cout << "  Avg: " << stats.avg / 1000 << " ms" << std::endl;
    std::cout << "  StdDev: " << stats.stddev / 1000 << " ms" << std::endl;
}

// Display metrics summary
void display_metrics_summary() {
    std::cout << "\n=== Metrics Summary ===" << std::endl;
    
    // Log snapshot
    log_metrics_snapshot();
    
    // Log detailed metrics
    log_metrics_detailed();
    
    // GPU metrics
    gpu::gpu_metrics_manager::instance().log_summary();
    
    // Export formats
    auto& aggregator = metrics_aggregator::instance();
    
    // Prometheus format
    std::cout << "\n=== Prometheus Export Sample ===" << std::endl;
    auto prometheus = aggregator.export_prometheus();
    // Show first few lines
    size_t pos = 0;
    for (int i = 0; i < 10 && pos < prometheus.size(); ++i) {
        size_t end = prometheus.find('\n', pos);
        if (end == std::string::npos) break;
        std::cout << prometheus.substr(pos, end - pos + 1);
        pos = end + 1;
    }
    std::cout << "..." << std::endl;
    
    // JSON format
    std::cout << "\n=== JSON Export Sample ===" << std::endl;
    auto json = aggregator.export_json();
    // Show first few lines
    pos = 0;
    for (int i = 0; i < 10 && pos < json.size(); ++i) {
        size_t end = json.find('\n', pos);
        if (end == std::string::npos) break;
        std::cout << json.substr(pos, end - pos + 1);
        pos = end + 1;
    }
    std::cout << "..." << std::endl;
}

int main(int argc, char** argv) {
    // Initialize logging
    common_log_set_verbosity_thold(LOG_DEFAULT_DEBUG);
    
    std::cout << "=== LLaMA.cpp Metrics Framework Demo ===" << std::endl;
    std::cout << "This demo showcases the metrics collection capabilities\n" << std::endl;
    
    // Start aggregation updater
    aggregation_updater updater(std::chrono::milliseconds(100));
    updater.start();
    
    // Run simulations
    simulate_model_load("llama-7b-v2");
    simulate_token_generation(50);
    simulate_batch_processing(16);
    simulate_batch_processing(32);
    simulate_batch_processing(64);
    simulate_gpu_operations();
    test_custom_metrics();
    
    // Simulate some errors
    track_error("demo_error");
    track_warning("demo_warning");
    
    // Let aggregations run for a bit
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    // Test aggregations
    test_aggregations();
    
    // Display summary
    display_metrics_summary();
    
    // Stop updater
    updater.stop();
    
    std::cout << "\n=== Demo Complete ===" << std::endl;
    std::cout << "The metrics framework provides:" << std::endl;
    std::cout << "- Counter, Gauge, Histogram, and Summary metric types" << std::endl;
    std::cout << "- GPU-specific metrics collection" << std::endl;
    std::cout << "- Automatic rate calculations and aggregations" << std::endl;
    std::cout << "- Multiple export formats (Prometheus, JSON, CSV)" << std::endl;
    std::cout << "- Integration with the extended logging system" << std::endl;
    
    return 0;
}