#!/bin/bash

# Test script for LLM CLI gpu-info command
# This script tests various GPU configurations and output formats

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default binary path
LLM_BIN="${LLM_BIN:-./build/bin/llm}"

# Check if binary exists
if [ ! -f "$LLM_BIN" ]; then
    echo -e "${RED}Error: LLM binary not found at $LLM_BIN${NC}"
    echo "Please build the project first or set LLM_BIN environment variable"
    exit 1
fi

echo "==================================="
echo "LLM CLI GPU Info Test Suite"
echo "==================================="
echo "Binary: $LLM_BIN"
echo ""

# Function to run test
run_test() {
    local test_name="$1"
    local command="$2"
    local expected_exit_code="${3:-0}"
    
    echo -e "${YELLOW}Test: $test_name${NC}"
    echo "Command: $command"
    
    set +e
    output=$($command 2>&1)
    exit_code=$?
    set -e
    
    if [ $exit_code -eq $expected_exit_code ]; then
        echo -e "${GREEN}✓ Exit code: $exit_code (expected)${NC}"
    else
        echo -e "${RED}✗ Exit code: $exit_code (expected: $expected_exit_code)${NC}"
        echo "Output:"
        echo "$output"
        return 1
    fi
    
    echo "Output preview:"
    echo "$output" | head -n 10
    if [ $(echo "$output" | wc -l) -gt 10 ]; then
        echo "... (truncated)"
    fi
    echo ""
    
    return 0
}

# Test basic functionality
echo "=== Basic Tests ==="
run_test "Help command" "$LLM_BIN gpu-info --help"
run_test "Basic GPU info" "$LLM_BIN gpu-info"
run_test "GPU info with alias" "$LLM_BIN gpuinfo"

# Test output formats
echo "=== Output Format Tests ==="
run_test "Verbose output" "$LLM_BIN gpu-info --verbose"
run_test "Verbose output (short)" "$LLM_BIN gpu-info -v"
run_test "JSON output" "$LLM_BIN gpu-info --json"

# Test JSON validity
echo "=== JSON Validation ==="
if command -v python3 &> /dev/null; then
    echo "Validating JSON output..."
    if $LLM_BIN gpu-info --json | python3 -m json.tool > /dev/null 2>&1; then
        echo -e "${GREEN}✓ JSON output is valid${NC}"
    else
        echo -e "${RED}✗ JSON output is invalid${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Python3 not found, skipping JSON validation${NC}"
fi
echo ""

# Test benchmark mode
echo "=== Benchmark Tests ==="
echo -e "${YELLOW}Note: Benchmark mode may take longer (100-500ms per GPU)${NC}"
run_test "Benchmark mode" "$LLM_BIN gpu-info --benchmark"
run_test "Benchmark mode (short)" "$LLM_BIN gpu-info --bench"

# Test environment variables
echo "=== Environment Variable Tests ==="
export GGML_CUDA_PROFILE_BANDWIDTH=1
run_test "With bandwidth profiling env var" "$LLM_BIN gpu-info"
unset GGML_CUDA_PROFILE_BANDWIDTH

export GGML_CUDA_LOG_TOPOLOGY=1
run_test "With topology logging env var" "$LLM_BIN gpu-info"
unset GGML_CUDA_LOG_TOPOLOGY

# Test combined options
echo "=== Combined Options Tests ==="
run_test "Verbose + JSON" "$LLM_BIN gpu-info --verbose --json"
run_test "Benchmark + JSON" "$LLM_BIN gpu-info --benchmark --json"
run_test "All options" "$LLM_BIN gpu-info --verbose --json --benchmark"

# System information
echo "=== System Information ==="
echo "OS: $(uname -s)"
echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "NVIDIA Driver Information:"
    nvidia-smi --query-gpu=driver_version,cuda_version --format=csv,noheader || true
    
    echo ""
    echo "GPU Count: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
    echo "GPUs:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
else
    echo -e "${YELLOW}nvidia-smi not found - CUDA may not be available${NC}"
fi

# Performance comparison
if [ -f "$LLM_BIN" ]; then
    echo ""
    echo "=== Performance Timing ==="
    echo "Timing basic gpu-info command..."
    time_output=$( { time $LLM_BIN gpu-info > /dev/null 2>&1; } 2>&1 )
    echo "$time_output"
    
    echo ""
    echo "Timing gpu-info with benchmark..."
    time_output=$( { time $LLM_BIN gpu-info --benchmark > /dev/null 2>&1; } 2>&1 )
    echo "$time_output"
fi

echo ""
echo "==================================="
echo "Test Summary"
echo "==================================="
echo -e "${GREEN}✓ All basic tests completed${NC}"
echo ""
echo "To test on a multi-GPU system:"
echo "1. Run this script on a machine with multiple GPUs"
echo "2. Check that NVLink connections are detected"
echo "3. Verify bandwidth measurements are reasonable"
echo ""
echo "For cloud testing (e.g., RunPod):"
echo "1. Copy the binary to the cloud instance"
echo "2. Run: $LLM_BIN gpu-info --verbose --benchmark"
echo "3. Compare results across different GPU configurations"