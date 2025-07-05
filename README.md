# LLM CLI - Enhanced Command-Line Interface for Large Language Models

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/gnana997/llm/actions/workflows/llm-cli-ci.yml/badge.svg)](https://github.com/gnana997/llm/actions/workflows/llm-cli-ci.yml)
[![Release](https://img.shields.io/github/v/release/gnana997/llm)](https://github.com/gnana997/llm/releases)

**LLM CLI** is an enhanced command-line interface built on top of [llama.cpp](https://github.com/ggml-org/llama.cpp), providing a unified, production-ready CLI tool for local LLM inference with advanced features like GPU profiling, comprehensive metrics, and a modular command framework.

## Key Features

- 🚀 **Unified CLI Interface** - Single `llm` command for all operations
- 🎯 **GPU Profiling & Monitoring** - Advanced GPU capability detection and bandwidth profiling
- 📊 **Comprehensive Metrics** - Built-in performance tracking and monitoring
- 🔧 **Modular Command Framework** - Easily extensible architecture for new commands
- 🖥️ **Multi-Platform Support** - Windows, macOS, Linux with CUDA, Metal, and CPU backends
- ⚡ **Production-Ready** - Enhanced logging, configuration management, and error handling

## Quick Start

### Installation

#### Build from Source

```bash
# Clone the repository
git clone https://github.com/gnana997/llm.git
cd llm

# Build with CUDA support (for NVIDIA GPUs)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j

# The llm binary will be in build/bin/
./build/bin/llm --help
```

#### Download Pre-built Binaries

Download the latest release for your platform from the [releases page](https://github.com/gnana997/llm/releases).

### Basic Usage

```bash
# Show GPU information
llm gpu-info
llm gpu-info --verbose    # Detailed GPU capabilities
llm gpu-info --json       # JSON output for scripting
llm gpu-info --benchmark  # Include bandwidth benchmarks

# Generate text (requires a model file)
llm generate -m model.gguf -p "Hello, world!"

# Show version
llm version

# Start API server
llm server -m model.gguf
```

## Enhanced Features

### GPU Information Command

The `gpu-info` command provides comprehensive information about available CUDA GPUs:

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

### Advanced CLI Framework

The LLM CLI uses a modular command framework that makes it easy to add new functionality:

- **Command Registry** - Automatic command discovery and registration
- **Configuration Management** - Support for config files and environment variables
- **Enhanced Logging** - Structured logging with multiple output formats
- **Batch Processing** - Process multiple prompts efficiently

## Configuration

LLM CLI supports multiple configuration methods:

1. **Configuration Files**: `.llmrc.json` or `.llmrc.yaml`
2. **Environment Variables**: `LLM_*` prefix
3. **Command-line Arguments**: Highest precedence

Example configuration file (`.llmrc.json`):
```json
{
  "model": "/path/to/model.gguf",
  "context_size": 4096,
  "gpu_layers": 35,
  "log_level": "info"
}
```

## Building with Different Backends

### CUDA (NVIDIA GPUs)
```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

### Metal (Apple Silicon)
```bash
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release
```

### CPU-only
```bash
cmake -B build
cmake --build build --config Release
```

## Project Structure

```
llm/
├── tools/llm/           # Main CLI application
├── common/
│   └── cli-framework/   # Reusable CLI framework
│       ├── commands/    # Command implementations
│       ├── core/        # Core framework components
│       └── utils/       # Utility functions
├── ggml/                # Core tensor library (from llama.cpp)
├── examples/            # Example applications
└── tests/               # Test suite
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
# Ubuntu/Debian
sudo apt-get install cmake ninja-build ccache

# macOS
brew install cmake ninja ccache

# Run tests
cmake -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features and development timeline.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is built on top of the excellent [llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and contributors. We maintain compatibility with upstream llama.cpp while adding enhanced CLI features.

## Links

- [Documentation](docs/README.md)
- [API Reference](docs/api/README.md)
- [GPU Profiling Guide](GPU_PROFILING.md)
- [CLI Command Reference](docs/cli-reference.md)