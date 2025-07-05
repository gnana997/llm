# Metrics Collection Framework Guide

## Overview

The metrics collection framework provides comprehensive monitoring and observability for llama.cpp, featuring:
- Multiple metric types (Counter, Gauge, Histogram, Summary)
- GPU-specific metrics collection
- Automatic aggregation and rate calculations
- Multiple export formats (Prometheus, JSON, CSV)
- Integration with the extended logging system

## Architecture

### Core Components

1. **Metric Types** (`metrics.h`)
   - **Counter**: Monotonically increasing values (e.g., total requests)
   - **Gauge**: Values that can go up/down (e.g., memory usage)
   - **Histogram**: Distribution of values with configurable buckets
   - **Summary**: Time-windowed percentile calculations

2. **GPU Metrics** (`metrics-gpu.h`)
   - Device memory usage and availability
   - GPU utilization and performance
   - Transfer and kernel execution tracking
   - Multi-GPU support

3. **Aggregator** (`metrics-aggregator.h`)
   - Rate calculations from counters
   - Derived metrics and ratios
   - Time-based aggregations
   - Alert conditions

## Usage

### Basic Metrics

```cpp
#include "metrics.h"
using namespace llama::metrics;

// Increment a counter
METRIC_INCREMENT("requests_total", 1);

// Set a gauge value
METRIC_SET("memory_used_bytes", current_memory);

// Record a histogram observation
METRIC_OBSERVE("request_latency_us", latency_microseconds);

// Use scoped timer for automatic timing
{
    METRIC_TIMER("operation_duration_us");
    // ... operation to time ...
}
```

### GPU Metrics

```cpp
#include "metrics-gpu.h"

// Track GPU memory allocation
METRIC_GPU_ALLOC(device_id, bytes);
METRIC_GPU_FREE(device_id, bytes);

// Track data transfers with timing
{
    GPU_METRIC_H2D_TIMER(device_id, bytes);
    // ... host to device transfer ...
}

// Track kernel execution
{
    GPU_METRIC_KERNEL_TIMER(device_id, "matrix_multiply");
    // ... kernel execution ...
}

// Update all GPU metrics
METRIC_GPU_UPDATE();
```

### Custom Metrics

```cpp
// Register custom counter
auto& registry = metric_registry::instance();
auto my_counter = registry.register_metric<counter>(
    "my_operations_total", 
    "Total custom operations"
);

// Register histogram with custom buckets
std::vector<double> buckets = {10, 50, 100, 500, 1000};
auto my_histogram = registry.register_metric<histogram>(
    "my_operation_latency_ms",
    "Operation latency in milliseconds",
    buckets
);

// Use the metrics
my_counter->increment();
my_histogram->observe(latency_ms);
```

### Aggregations and Rates

```cpp
#include "metrics-aggregator.h"

auto& aggregator = metrics_aggregator::instance();

// Register rate calculation
aggregator.register_rate(
    "tokens_generated_total",      // Counter name
    "tokens_per_second",           // Rate metric name
    std::chrono::seconds(60)       // Window size
);

// Register derived metric
aggregator.register_derived(
    "cache_hit_rate_percent",
    "Cache hit rate percentage",
    []() {
        auto hits = get_counter("cache_hits_total");
        auto misses = get_counter("cache_misses_total");
        if (!hits || !misses) return 0.0;
        
        double total = hits->get() + misses->get();
        return total > 0 ? (hits->get() / total) * 100.0 : 0;
    }
);

// Update all aggregations
METRIC_UPDATE_AGGREGATIONS();
```

## Common Metrics

The framework automatically registers common metrics:

### Model Metrics
- `model_loads_total`: Total models loaded
- `model_load_duration_us`: Model loading time
- `models_loaded`: Currently loaded models

### Generation Metrics
- `tokens_generated_total`: Total tokens generated
- `token_generation_duration_us`: Token generation time
- `active_generation_requests`: Active requests

### Batch Processing
- `batches_processed_total`: Total batches processed
- `batch_size`: Batch size distribution
- `batch_processing_duration_us`: Batch processing time

### Memory Metrics
- `memory_used_bytes`: Memory in use
- `memory_allocated_bytes`: Memory allocated
- `memory_allocations_total`: Total allocations
- `memory_deallocations_total`: Total deallocations

### Cache Metrics
- `cache_hits_total`: Cache hits
- `cache_misses_total`: Cache misses
- `cache_size_bytes`: Cache size
- `cache_lookup_duration_us`: Lookup time

### GPU Metrics (per device)
- `gpu_memory_used_bytes_gpu{N}`: GPU memory used
- `gpu_memory_free_bytes_gpu{N}`: GPU memory free
- `gpu_utilization_percent_gpu{N}`: GPU utilization
- `gpu_allocations_total_gpu{N}`: GPU allocations
- `gpu_kernel_launches_total_gpu{N}`: Kernel launches

## Export Formats

### Prometheus Format

```cpp
auto prometheus_data = aggregator.export_prometheus();
// Outputs:
// # HELP requests_total Total number of requests
// # TYPE requests_total counter
// requests_total 12345
```

### JSON Format

```cpp
auto json_data = aggregator.export_json();
// Outputs:
// {
//   "timestamp": 1642512345678,
//   "metrics": {
//     "requests_total": 12345,
//     "memory_used_bytes": 1073741824
//   }
// }
```

### CSV Format

```cpp
auto csv_data = aggregator.export_csv();
// Outputs time series data in CSV format
```

## Integration with Logging

The metrics framework integrates with the extended logging system:

```cpp
// Log metrics snapshot
log_metrics_snapshot();

// Log detailed metrics report
log_metrics_detailed();

// Log GPU metrics summary
gpu::gpu_metrics_manager::instance().log_summary();
```

## Automatic Updates

Use the aggregation updater for automatic metric updates:

```cpp
// Start automatic updates every second
aggregation_updater updater(std::chrono::seconds(1));
updater.start();

// ... your application code ...

// Stop updates
updater.stop();
```

## Alerts

Configure alerts based on metric values:

```cpp
alert_condition high_memory{
    "memory_utilization_percent",           // Metric to monitor
    [](double value) { return value > 90; }, // Condition
    "Memory usage above 90%",               // Message
    std::chrono::seconds(300)               // Cooldown
};

aggregator.register_alert("high_memory_usage", high_memory);

// Check alerts periodically
METRIC_CHECK_ALERTS();
```

## Best Practices

1. **Use appropriate metric types**
   - Counters for cumulative values
   - Gauges for current state
   - Histograms for distributions
   - Summaries for percentiles over time

2. **Choose meaningful buckets**
   - Use `histogram::default_latency_buckets()` for time metrics
   - Use `histogram::default_memory_buckets()` for memory metrics
   - Define custom buckets for domain-specific metrics

3. **Minimize overhead**
   - Batch metric updates when possible
   - Use scoped timers for automatic timing
   - Avoid excessive histogram observations in hot paths

4. **GPU metrics**
   - Initialize GPU metrics once at startup
   - Update GPU metrics periodically, not on every operation
   - Use GPU timers for accurate kernel timing

5. **Aggregations**
   - Register rates for important counters
   - Create derived metrics for ratios and percentages
   - Use appropriate time windows for aggregations

## Performance Considerations

- Metric operations use atomic operations for thread safety
- Histograms use lock-free updates where possible
- GPU metrics collection has minimal overhead
- Aggregations run in a separate thread

## Example: Complete Integration

```cpp
#include "metrics.h"
#include "metrics-gpu.h"
#include "metrics-aggregator.h"
#include "log-ex.h"

int main() {
    // Initialize logging
    common_log_set_verbosity_thold(LOG_DEFAULT_DEBUG);
    
    // Start metric updates
    aggregation_updater updater;
    updater.start();
    
    // Initialize GPU metrics
    gpu::gpu_metrics_manager::instance().initialize();
    
    // Your application logic
    while (running) {
        // Track request
        track_generation_request_start();
        
        {
            METRIC_TIMER("request_processing_us");
            
            // Process request
            process_tokens();
            
            // Update metrics
            METRIC_INCREMENT("requests_processed", 1);
            METRIC_SET("queue_depth", get_queue_size());
        }
        
        track_generation_request_end(duration);
        
        // Periodic updates
        if (should_update_metrics()) {
            METRIC_GPU_UPDATE();
            log_metrics_snapshot();
        }
    }
    
    // Final report
    log_metrics_detailed();
    gpu::gpu_metrics_manager::instance().log_detailed();
    
    return 0;
}
```

## Troubleshooting

### Metrics not updating
- Ensure `aggregation_updater` is started
- Check that metrics are registered before use
- Verify counter increments are positive

### High memory usage
- Limit histogram bucket count
- Reduce aggregation history retention
- Use sampling for high-frequency metrics

### GPU metrics unavailable
- Check CUDA/ROCm installation
- Verify GPU device initialization
- Ensure proper permissions for GPU access

## Future Extensions

- OpenTelemetry export support
- Distributed metrics aggregation
- Custom metric dashboards
- Machine learning on metrics
- Predictive alerting