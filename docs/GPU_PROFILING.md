# GPU Profiling and Capability Detection

This document describes the GPU profiling and capability detection features added to the llama.cpp CUDA backend.

## Overview

The GPU profiling system provides comprehensive information about CUDA devices including:
- Extended compute capabilities detection
- Memory bandwidth profiling
- GPU topology mapping (NVLink/PCIe connections)
- Detailed hardware feature detection
- Performance metrics tracking

## Environment Variables

### `GGML_CUDA_PROFILE_BANDWIDTH`
When set to any value, enables memory bandwidth profiling during initialization. This will run a bandwidth test kernel to measure actual memory throughput.

Example:
```bash
export GGML_CUDA_PROFILE_BANDWIDTH=1
```

### `GGML_CUDA_LOG_TOPOLOGY`
When set to any value, enables detailed GPU topology logging showing peer-to-peer connections and hardware capabilities.

Example:
```bash
export GGML_CUDA_LOG_TOPOLOGY=1
```

## Features Detected

### Compute Capabilities
- FP16 support (Pascal and later)
- INT8 DP4A instructions
- Tensor Core availability
- BF16 support (Ampere and later)

### Memory Hierarchy
- Total GPU memory
- L1/L2 cache sizes
- Constant memory size
- Texture memory limits
- Measured memory bandwidth

### Connectivity
- PCIe generation and link width
- NVLink detection and peer mapping
- Peer-to-peer capability matrix

### Performance Characteristics
- Maximum clock speeds
- Number of SMs and cores
- Maximum threads per block
- Register limits

## Metrics Tracked

The following metrics are automatically tracked when the metrics system is available:

### Global Metrics
- `gpu_initialization_duration_us` - Time taken to initialize all GPUs
- `gpu_count_total` - Total number of CUDA devices

### Per-Device Metrics (suffixed with _gpu{N})
- `gpu_memory_total_bytes_gpu{N}` - Total memory for device N
- `gpu_compute_capability_gpu{N}` - Compute capability (major.minor * 10)
- `gpu_memory_bandwidth_gbps_gpu{N}` - Measured memory bandwidth
- `gpu_l2_cache_size_kb_gpu{N}` - L2 cache size in KB
- `gpu_pcie_generation_gpu{N}` - PCIe generation
- `gpu_has_nvlink_gpu{N}` - NVLink availability (0/1)

## Example Output

When both environment variables are set, you'll see output like:

```
=== GPU Topology ===
GPU 0: NVIDIA GeForce RTX 4090
  Compute Capability: 8.9 (Ada Lovelace)
  Memory: 24.00 GB (23.99 GB available)
  Memory Bandwidth: 1008.0 GB/s (theoretical), 952.3 GB/s (measured)
  SM Count: 128, Max Threads/Block: 1024
  L2 Cache: 72 MB
  PCIe: Gen4 x16
  Features: FP16 INT8-DP4A TensorCores BF16
  
GPU 1: NVIDIA GeForce RTX 4090
  Compute Capability: 8.9 (Ada Lovelace)
  Memory: 24.00 GB (23.99 GB available)
  Memory Bandwidth: 1008.0 GB/s (theoretical), 948.7 GB/s (measured)
  SM Count: 128, Max Threads/Block: 1024
  L2 Cache: 72 MB
  PCIe: Gen4 x16
  Features: FP16 INT8-DP4A TensorCores BF16
  Peer Connections:
    GPU 0 <-> GPU 1: PCIe P2P

GPU Topology Summary:
  GPU 0 <-> GPU 1: PCIe P2P (~32 GB/s bidirectional)
```

## Implementation Details

### Files Added
- `ggml/src/ggml-cuda/gpu-profiler.cuh` - Header with profiling function declarations
- `ggml/src/ggml-cuda/gpu-profiler.cu` - Implementation of profiling functions

### Files Modified
- `ggml/src/ggml-cuda/common.cuh` - Extended cuda_device_info structure
- `ggml/src/ggml-cuda/ggml-cuda.cu` - Integration in ggml_cuda_init()

### Key Functions
- `profile_memory_bandwidth()` - Measures actual memory bandwidth
- `detect_gpu_topology()` - Maps peer-to-peer connections
- `query_device_capabilities()` - Queries extended device attributes
- `log_gpu_topology()` - Outputs human-readable topology information

## Performance Impact

The profiling features have minimal performance impact:
- Basic capability detection adds < 1ms per GPU
- Memory bandwidth profiling (when enabled) adds ~100-500ms per GPU
- Topology detection adds < 10ms for multi-GPU systems

## Future Enhancements

Potential improvements include:
- NVML integration for real-time power/temperature monitoring
- Peer-to-peer bandwidth measurement
- Automatic optimal GPU assignment based on topology
- Runtime profiling of kernel performance
- Integration with performance prediction models