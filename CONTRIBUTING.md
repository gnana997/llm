# Contributing to LLM CLI

Thank you for your interest in contributing to LLM CLI! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Adding New CLI Commands](#adding-new-cli-commands)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/llm.git
   cd llm
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/gnana997/llm.git
   ```

## Development Setup

### Prerequisites

- CMake 3.14 or higher
- C++17 compatible compiler
- CUDA Toolkit 11.0+ (optional, for GPU support)
- Git

### Building the Project

```bash
# Standard build
cmake -B build -DGGML_CUDA=ON  # Enable CUDA if available
cmake --build build --config Release -j

# Development build with tests
cmake -B build-dev -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-dev -j

# Run tests
cd build-dev && ctest
```

### Running the CLI

```bash
./build/bin/llm --help
```

## How to Contribute

### Areas for Contribution

1. **New CLI Commands** - Add useful commands to the CLI
2. **GPU Features** - Enhance GPU profiling and monitoring
3. **Documentation** - Improve docs and examples
4. **Bug Fixes** - Fix issues and improve stability
5. **Performance** - Optimize performance and resource usage
6. **Tests** - Add test coverage for new and existing features

### Reporting Issues

- Use the issue templates provided
- Include system information (OS, GPU, CUDA version)
- Provide minimal reproducible examples
- Check existing issues before creating new ones

## Adding New CLI Commands

The LLM CLI uses a modular command framework. Here's how to add a new command:

### 1. Create Command Header

Create `common/cli-framework/commands/your_command.h`:

```cpp
#pragma once
#include "../core/command.h"

namespace llama {
namespace cli {

class YourCommand : public Command {
public:
    std::string name() const override { return "your-command"; }
    std::string description() const override { return "Brief description"; }
    std::string usage() const override { return "your-command [options]"; }
    std::vector<std::string> aliases() const override { return {"yc"}; }
    
    int execute(CommandContext& ctx) override;
};

} // namespace cli
} // namespace llama
```

### 2. Implement Command

Create `common/cli-framework/commands/your_command.cpp`:

```cpp
#include "your_command.h"
#include <iostream>

namespace llama {
namespace cli {

int YourCommand::execute(CommandContext& ctx) {
    // Parse options
    bool verbose = ctx.hasOption("verbose") || ctx.hasOption("v");
    
    // Your command logic here
    std::cout << "Executing your command..." << std::endl;
    
    // Return 0 for success, non-zero for error
    return 0;
}

} // namespace cli
} // namespace llama
```

### 3. Register Command

Add to `tools/llm/main.cpp`:

```cpp
#include "common/cli-framework/commands/your_command.h"

// In main():
app.addCommand(std::make_unique<YourCommand>());
```

### 4. Update CMakeLists.txt

Add your source files to `common/cli-framework/CMakeLists.txt`.

## Testing Guidelines

### Unit Tests

- Write tests for new functionality
- Place tests in appropriate test files
- Use the existing test framework

### Integration Tests

- Test command-line interface behavior
- Verify output formats (text, JSON)
- Test error conditions

### GPU Testing

For GPU-specific features:

1. Test on systems without GPUs (graceful fallback)
2. Test with single GPU
3. Test with multiple GPUs (if possible)
4. Document GPU requirements

### Manual Testing Checklist

Before submitting a PR:

- [ ] Build succeeds on your platform
- [ ] All tests pass
- [ ] Command works as expected
- [ ] Help text is clear and accurate
- [ ] Error messages are helpful
- [ ] Code follows project style

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow coding standards
   - Add tests
   - Update documentation

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "cli: add new feature X"
   ```
   
   Commit message format:
   - `cli:` for CLI-specific changes
   - `gpu:` for GPU-related changes
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `test:` for test additions

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Use the PR template
   - Link related issues
   - Provide clear description
   - Include test results

### PR Review Process

- PRs require at least one review
- Address review feedback promptly
- Keep PRs focused and small
- Update PR description as needed

## Coding Standards

### C++ Guidelines

Follow the existing codebase style:

- Use 4 spaces for indentation
- Opening braces on same line
- Use `snake_case` for functions and variables
- Use `PascalCase` for classes
- Add meaningful comments

### Example Code Style

```cpp
namespace llama {
namespace cli {

class ExampleCommand : public Command {
private:
    int count_ = 0;
    
public:
    int execute(CommandContext& ctx) override {
        // Parse arguments
        auto args = ctx.args();
        
        if (args.empty()) {
            std::cerr << "Error: No arguments provided\n";
            return 1;
        }
        
        // Process each argument
        for (const auto& arg : args) {
            process_argument(arg);
        }
        
        return 0;
    }
    
private:
    void process_argument(const std::string& arg) {
        // Implementation here
        count_++;
    }
};

} // namespace cli
} // namespace llama
```

### Documentation

- Add comments for complex logic
- Document public APIs
- Update README for new features
- Add examples for new commands

## Syncing with Upstream

Keep your fork up to date:

```bash
git fetch upstream
git checkout master
git merge upstream/master
git push origin master
```

For detailed upstream sync instructions, see [.github/UPSTREAM_SYNC.md](.github/UPSTREAM_SYNC.md).

## Questions?

- Open a [GitHub Discussion](https://github.com/gnana997/llm/discussions)
- Check existing issues and PRs
- Join our community chat (coming soon)

Thank you for contributing to LLM CLI!