#!/bin/bash

# Test script for intelligent layer distribution
# Tests the performance improvement of optimized GPU distribution

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODEL_PATH="${MODEL_PATH:-models/llama-2-7b.gguf}"
PROMPT="${PROMPT:-"The quick brown fox jumps over the lazy dog. This is a test of the intelligent layer distribution system."}"
N_PREDICT="${N_PREDICT:-100}"
BUILD_DIR="${BUILD_DIR:-build}"

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Intelligent Layer Distribution Test${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Check if binary exists
if [ ! -f "$BUILD_DIR/bin/llm" ]; then
    echo -e "${RED}Error: LLM binary not found at $BUILD_DIR/bin/llm${NC}"
    echo "Please build the project first with CUDA support"
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${YELLOW}Warning: Model not found at $MODEL_PATH${NC}"
    echo "Using a dummy test mode instead"
    TEST_MODE="dummy"
fi

# Function to run test
run_test() {
    local strategy=$1
    local description=$2
    local output_file="test_${strategy}_$(date +%s).log"
    
    echo -e "\n${YELLOW}Testing: $description${NC}"
    echo "Strategy: $strategy"
    echo "Output: $output_file"
    
    # Run the test
    if [ "$TEST_MODE" == "dummy" ]; then
        # Dummy test mode - just check if the option works
        timeout 10s $BUILD_DIR/bin/llm gpu-info --verbose 2>&1 | tee $output_file || true
        echo -e "${GREEN}✓ GPU info test completed${NC}"
    else
        # Real model test
        /usr/bin/time -v $BUILD_DIR/bin/llm generate \
            -m "$MODEL_PATH" \
            -p "$PROMPT" \
            -n $N_PREDICT \
            -ngl 40 \
            --gpu-strategy $strategy \
            --log-level info \
            2>&1 | tee $output_file
        
        # Extract performance metrics
        local elapsed=$(grep "Elapsed (wall clock) time" $output_file | awk '{print $8}')
        local tokens_per_sec=$(grep "eval time" $output_file | grep -oP '\d+\.\d+ tokens/s' | head -1)
        
        echo -e "\n${GREEN}Results:${NC}"
        echo "  Elapsed time: $elapsed"
        echo "  Performance: $tokens_per_sec"
    fi
}

# Test 1: Get GPU information
echo -e "\n${BLUE}=== GPU Information ===${NC}"
$BUILD_DIR/bin/llm gpu-info --verbose

# Count GPUs
GPU_COUNT=$($BUILD_DIR/bin/llm gpu-info --json 2>/dev/null | grep -c '"device_id"' || echo "0")
echo -e "\n${GREEN}Detected $GPU_COUNT GPU(s)${NC}"

if [ "$GPU_COUNT" -lt 2 ]; then
    echo -e "${YELLOW}Warning: Less than 2 GPUs detected. Intelligent distribution benefits are limited.${NC}"
    echo "The test will continue but performance improvements may not be visible."
fi

# Test 2: Equal distribution (baseline)
echo -e "\n${BLUE}=== Performance Tests ===${NC}"
run_test "equal" "Equal Distribution (Baseline)"
EQUAL_LOG=$(ls -t test_equal_*.log | head -1)

# Test 3: Optimized distribution
run_test "optimized" "Intelligent Distribution"
OPTIMIZED_LOG=$(ls -t test_optimized_*.log | head -1)

# Test 4: Manual distribution (if tensor split provided)
if [ ! -z "$TENSOR_SPLIT" ]; then
    run_test "manual" "Manual Distribution (tensor_split=$TENSOR_SPLIT)"
fi

# Compare results
echo -e "\n${BLUE}=== Performance Comparison ===${NC}"

if [ -f "$EQUAL_LOG" ] && [ -f "$OPTIMIZED_LOG" ]; then
    # Extract metrics for comparison
    EQUAL_TIME=$(grep "Elapsed (wall clock) time" $EQUAL_LOG | awk '{print $8}' | sed 's/[^0-9.]//g')
    OPTIMIZED_TIME=$(grep "Elapsed (wall clock) time" $OPTIMIZED_LOG | awk '{print $8}' | sed 's/[^0-9.]//g')
    
    if [ ! -z "$EQUAL_TIME" ] && [ ! -z "$OPTIMIZED_TIME" ]; then
        # Calculate speedup
        SPEEDUP=$(echo "scale=2; $EQUAL_TIME / $OPTIMIZED_TIME" | bc -l)
        IMPROVEMENT=$(echo "scale=1; ($SPEEDUP - 1) * 100" | bc -l)
        
        echo -e "${GREEN}Equal distribution time: ${EQUAL_TIME}s${NC}"
        echo -e "${GREEN}Optimized distribution time: ${OPTIMIZED_TIME}s${NC}"
        echo -e "${GREEN}Speedup: ${SPEEDUP}x (${IMPROVEMENT}% improvement)${NC}"
        
        # Check if we achieved target
        if (( $(echo "$SPEEDUP > 1.2" | bc -l) )); then
            echo -e "\n${GREEN}✓ SUCCESS: Achieved >20% performance improvement!${NC}"
        else
            echo -e "\n${YELLOW}⚠ Performance improvement less than 20%${NC}"
            echo "This may be due to:"
            echo "  - Homogeneous GPU setup (all GPUs are similar)"
            echo "  - Model too small to benefit from distribution"
            echo "  - Other system bottlenecks"
        fi
    fi
fi

# Show distribution details
echo -e "\n${BLUE}=== Distribution Analysis ===${NC}"
if [ -f "$OPTIMIZED_LOG" ]; then
    echo "Intelligent distribution decisions:"
    grep -A 20 "Layer Distribution Result" $OPTIMIZED_LOG | head -30 || true
fi

# Cleanup old logs (keep last 5)
ls -t test_*.log 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null || true

echo -e "\n${BLUE}======================================${NC}"
echo -e "${BLUE}Test Complete${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo "Log files saved:"
echo "  - Equal distribution: $EQUAL_LOG"
echo "  - Optimized distribution: $OPTIMIZED_LOG"
echo ""
echo "To test with your own model:"
echo "  MODEL_PATH=/path/to/model.gguf ./test_layer_distribution.sh"
echo ""
echo "To test with custom tensor split:"
echo "  TENSOR_SPLIT='3,1' ./test_layer_distribution.sh"