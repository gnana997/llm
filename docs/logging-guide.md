# Extended Logging System Guide

## Overview

The extended logging system provides advanced logging capabilities for llama.cpp, including performance tracking, structured logging, log rotation, and category-based filtering.

## Features

### 1. Additional Log Levels
- **PERF (2)**: Performance metrics and timing information
- **TRACE (3)**: Detailed trace logging for debugging

### 2. Structured Logging
- JSON output format for machine parsing
- Timestamped events with microsecond precision
- Performance event tracking with duration calculation

### 3. Category-Based Filtering
Available categories:
- `general`: General application logs
- `perf`: Performance-related logs
- `memory`: Memory allocation/deallocation logs
- `gpu`: GPU operations and statistics
- `inference`: Inference-specific logs
- `model`: Model loading and management
- `batch`: Batch processing logs
- `cache`: Cache operations
- `network`: Network-related logs (for server mode)

### 4. Log Rotation
- Automatic file rotation based on size
- Configurable rotation parameters
- Old log file management

### 5. Performance Tracking
- Automatic timing of operations
- Performance summary generation
- Statistical analysis (avg, median, P95, P99)

## Usage

### Command Line Options

```bash
# Set log level
llm --log-level trace generate --model model.gguf --prompt "Hello"

# Enable performance logging with JSON output
llm --log-perf true --log-format json generate --model model.gguf

# Filter by categories
llm --log-categories gpu,inference,perf generate --model model.gguf

# Set up log rotation
llm --log-file app.log --log-rotate-size 50 --log-rotate-count 3 generate

# Filter messages with regex
llm --log-filter "batch.*processing" generate --model model.gguf
```

### In Code

```cpp
#include "common/log-ex.h"

// Basic performance logging
LOG_PERF("Processing batch of size %d\n", batch_size);

// Trace logging
LOG_TRACE("Detailed token processing: token_id=%d\n", token_id);

// Category-based logging
LOG_CAT(COMMON_LOG_CAT_GPU, GGML_LOG_LEVEL_INFO, "GPU memory: %zu MB\n", gpu_mem);

// Performance timers
{
    PERF_TIMER(PERF_EVENT_MODEL_LOAD_START, "model_name");
    // ... model loading code ...
}

// Manual performance events
LOG_PERF_START(PERF_EVENT_BATCH_START, "batch_32");
// ... processing ...
LOG_PERF_END(PERF_EVENT_BATCH_END, "batch_32");

// Custom performance events
common_perf_event event;
event.type = PERF_EVENT_CUSTOM;
event.timestamp = std::chrono::high_resolution_clock::now();
event.name = "custom_operation";
event.metadata["tokens"] = "512";
common_log_ex::instance().log_perf_event(event);
```

### Configuration Examples

#### High-Performance Inference Logging
```bash
llm --log-level perf \
    --log-perf true \
    --log-categories perf,gpu,inference \
    --log-format json \
    --log-file inference.log \
    generate --model model.gguf
```

#### Debug Mode with Full Tracing
```bash
llm --log-level trace \
    --log-categories all \
    --log-file debug.log \
    --log-rotate-size 200 \
    generate --model model.gguf
```

#### Production Logging
```bash
llm --log-level info \
    --log-categories general,error \
    --log-file /var/log/llm/app.log \
    --log-rotate-size 100 \
    --log-rotate-count 7 \
    generate --model model.gguf
```

## Performance Summary Output

When `--log-perf true` is enabled, a performance summary is generated:

```
=== Performance Summary ===
MODEL_LOAD_START:
  Count: 1
  Avg: 2500000 us
  Median: 2500000 us
  P95: 2500000 us
  P99: 2500000 us
TOKEN_GENERATION:
  Count: 256
  Avg: 15625 us
  Median: 15000 us
  P95: 18000 us
  P99: 20000 us
```

## JSON Output Format

With `--log-format json`, logs are output in structured JSON:

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "category": "INFERENCE",
  "message": "Starting text generation"
}

{
  "type": "perf_event",
  "event_type": "TOKEN_START",
  "timestamp": 1705315845123456,
  "name": "token_42",
  "metadata": {
    "batch_id": "batch_1",
    "position": "42"
  }
}
```

## Best Practices

1. **Use appropriate log levels**: Reserve TRACE for detailed debugging, PERF for performance analysis
2. **Use categories**: Tag logs with appropriate categories for easier filtering
3. **Performance timers**: Use RAII timers for automatic timing of scoped operations
4. **Log rotation**: Always configure rotation for production deployments
5. **JSON format**: Use JSON format when logs need to be processed by other tools

## Integration with Monitoring

The JSON output format makes it easy to integrate with monitoring solutions:
- Ship logs to Elasticsearch for analysis
- Use Prometheus log exporters for metrics
- Process with Logstash for transformation
- Visualize performance data in Grafana

## Metrics Integration

The extended logging system integrates seamlessly with the metrics collection framework, providing comprehensive observability:

### Automatic Metric Emission

Performance events logged through the logging system automatically update corresponding metrics:

```cpp
// This performance timer...
{
    PERF_TIMER(PERF_EVENT_MODEL_LOAD_START, "model_name");
    // ... model loading code ...
}

// ...automatically updates these metrics:
// - model_loads_total (counter)
// - model_load_duration_us (histogram)
// - models_loaded (gauge)
```

### Combined Usage

```cpp
#include "common/log-ex.h"
#include "common/metrics.h"

// Log and metric together
LOG_PERF("Starting batch processing: size=%d\n", batch_size);
METRIC_OBSERVE("batch_size", batch_size);

// Performance tracking with both systems
{
    PERF_TIMER(PERF_EVENT_BATCH_START, "batch_processing");
    METRIC_TIMER("batch_processing_duration_us");
    
    // Process batch...
    
    track_batch_processed(batch_size, duration_us);
}
```

### Unified Reporting

```cpp
// Log metrics snapshot alongside performance logs
log_metrics_snapshot();

// Get combined performance summary
common_log_ex::instance().print_performance_summary();
log_metrics_detailed();
```

### Benefits of Integration

1. **Correlation**: Performance logs provide context for metric changes
2. **Debugging**: Trace logs help explain metric anomalies
3. **Completeness**: Logs capture events, metrics track trends
4. **Flexibility**: Choose appropriate tool for each use case

### Example: Full Observability

```bash
# Enable both logging and metrics
llm --log-level perf \
    --log-perf true \
    --log-categories gpu,inference,perf \
    --log-format json \
    --log-file inference.log \
    generate --model model.gguf

# Metrics are automatically collected and can be exported
# Check metrics endpoint or use aggregator export functions
```

See the [Metrics Collection Framework Guide](metrics-guide.md) for detailed metrics documentation.