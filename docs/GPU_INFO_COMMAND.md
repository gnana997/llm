# GPU Info Command for LLM CLI

The `gpu-info` command displays comprehensive information about available CUDA GPUs in your system.

## Usage

### Basic Usage
```bash
llm gpu-info
# or use the alias
llm gpuinfo
```

### Options
- `--verbose` or `-v`: Show extended GPU information including cache sizes, PCIe details, and more
- `--json`: Output GPU information in JSON format for scripting
- `--benchmark` or `--bench`: Run memory bandwidth benchmark (adds ~100-500ms per GPU)

## Examples

### Basic GPU Information
```bash
$ llm gpu-info

GPU Information:
================

Device 0: NVIDIA GeForce RTX 4090
  Compute Capability: 8.9 (Ada Lovelace)
  Memory: 24.00 GB
  SMs: 128, Cores: 16384
  Clock: 2520 MHz
  L2 Cache: 72 MB
  Features: FP16 INT8 TensorCores BF16

Total GPUs: 1
Total Memory: 24.00 GB
```

### Verbose Output
```bash
$ llm gpu-info --verbose

GPU Information (Verbose):
==========================

Device 0: NVIDIA GeForce RTX 4090
  Architecture:
    Compute Capability: 8.9
    Warp Size: 32
    Integrated: No
  Memory:
    Total: 24.00 GB
    Bandwidth: 952.3 GB/s (measured)
    Clock: 10501 MHz
  Compute:
    SMs: 128
    Max Clock: 2520 MHz
    Max Threads/Block: 1024
    Max Blocks/SM: 32
    Max Registers/Block: 65536
  Cache:
    L1/Shared: 128 KB
    L2: 72 MB
    Constant: 64 KB
    Texture: 65536 KB
  PCIe:
    Generation: 4
    Link Width: x16
  Features:
    FP16: Yes
    INT8 DP4A: Yes
    Tensor Cores: Yes
    BF16: Yes
    Virtual Memory: Yes
    VMM Granularity: 2048 KB
```

### JSON Output
```bash
$ llm gpu-info --json
{
  "device_count": 1,
  "devices": [
    {
      "id": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "compute_capability": 89,
      "compute_capability_major": 8,
      "compute_capability_minor": 9,
      "memory_gb": 24.00,
      "memory_bytes": 25757220864,
      "sms": 128,
      "max_clock_mhz": 2520,
      "l2_cache_mb": 72,
      "features": ["fp16", "int8", "tensor_cores", "bf16"],
      "integrated": false,
      "has_nvlink": false
    }
  ],
  "total_memory_gb": 24.00
}
```

### With Bandwidth Benchmark
```bash
$ llm gpu-info --benchmark

GPU Information:
================

Device 0: NVIDIA GeForce RTX 4090
  Compute Capability: 8.9 (Ada Lovelace)
  Memory: 24.00 GB
  Memory Bandwidth: 952.3 GB/s (measured)
  SMs: 128, Cores: 16384
  Clock: 2520 MHz
  L2 Cache: 72 MB
  Features: FP16 INT8 TensorCores BF16

Total GPUs: 1
Total Memory: 24.00 GB
```

## Multi-GPU Systems

On systems with multiple GPUs, the command will also display topology information:

```bash
GPU Topology:
=============
  GPU 0 <-> GPU 1: NVLink
  GPU 0 <-> GPU 2: PCIe P2P
  GPU 1 <-> GPU 2: PCIe P2P
```

## Use Cases

1. **System Verification**: Quickly check if CUDA is available and working
2. **Performance Analysis**: Understand GPU capabilities before running models
3. **Multi-GPU Setup**: Verify GPU topology and peer connections
4. **Scripting**: Use JSON output to programmatically query GPU capabilities
5. **Benchmarking**: Measure actual memory bandwidth vs theoretical

## Requirements

- CUDA-enabled build of llama.cpp (`GGML_CUDA=ON`)
- NVIDIA GPU with CUDA support
- Appropriate CUDA drivers installed

## Notes

- The memory bandwidth benchmark is optional as it takes 100-500ms per GPU
- Features detected are based on compute capability
- PCIe information may show default values on some systems
- NVLink detection is heuristic-based unless bandwidth profiling is enabled