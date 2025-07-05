# LLaMA.cpp Unified CLI Framework

This is the new unified CLI framework for llama.cpp, providing a single entry point for all llama.cpp functionality.

## Features

- **Unified Interface**: Single `llm` command for all operations
- **Subcommand Architecture**: Organized commands like `llm generate`, `llm server`, `llm quantize`
- **Configuration Management**: Support for YAML/JSON config files and environment variables
- **Extensible Design**: Easy to add new commands via plugin architecture
- **Backward Compatibility**: Existing tools continue to work

## Usage

```bash
# Text generation
llm generate --model path/to/model.gguf --prompt "Hello, world"

# Start API server
llm server --model path/to/model.gguf --port 8080

# Quantize a model
llm quantize input.gguf output.gguf q4_0

# Run benchmarks
llm benchmark --model path/to/model.gguf

# Show version
llm version

# Get help
llm --help
llm generate --help
```

## Configuration

The CLI supports configuration through multiple sources (in order of precedence):

1. Command line arguments
2. Environment variables (prefix with `LLM_`)
3. Configuration files (`.llmrc`, `.llmrc.json`, `.llmrc.yaml`)
4. Default values

### Configuration File Locations

The CLI looks for configuration files in:
1. Current directory: `.llmrc`
2. Home directory: `~/.llmrc`
3. Config directory: `~/.config/llm/config`

### Example Configuration

**YAML format** (`.llmrc.yaml`):
```yaml
model:
  path: /path/to/model.gguf
  gpu_layers: 32

generation:
  temperature: 0.8
  top_k: 40
```

**JSON format** (`.llmrc.json`):
```json
{
  "model.path": "/path/to/model.gguf",
  "model.gpu_layers": "32",
  "generation.temperature": "0.8",
  "generation.top_k": "40"
}
```

**Environment variables**:
```bash
export LLM_MODEL_PATH=/path/to/model.gguf
export LLM_MODEL_GPU_LAYERS=32
```

## Architecture

The CLI framework consists of:

- **Command Interface**: Base classes for implementing commands
- **Registry System**: Dynamic command registration and discovery
- **Configuration Manager**: Handles config files and environment variables
- **Application Core**: Main entry point and command dispatcher

## Adding New Commands

To add a new command:

1. Create a new class inheriting from `Command`
2. Implement required methods: `name()`, `description()`, `usage()`, `execute()`
3. Register the command in `main.cpp`

Example:
```cpp
class MyCommand : public Command {
public:
    std::string name() const override { return "mycommand"; }
    std::string description() const override { return "My custom command"; }
    std::string usage() const override { return "mycommand [options]"; }
    
    int execute(CommandContext& ctx) override {
        // Implementation
        return 0;
    }
};
```

## Build Instructions

The CLI framework is built as part of the standard llama.cpp build:

```bash
mkdir build
cd build
cmake ..
make llm
```

## Migration Path

The unified CLI is designed to coexist with existing tools:

1. **Phase 1**: Unified CLI wraps existing tools
2. **Phase 2**: Gradual migration of functionality into unified framework
3. **Phase 3**: Legacy tools become symlinks to unified CLI

## Future Enhancements

- Plugin system for external commands
- Shell completion generation
- Interactive mode improvements
- Remote model management
- Integrated model marketplace