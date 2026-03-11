#!/bin/bash

# Build script for perception model containers
# This script builds all perception model Docker images using the local CUDA base image

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BASE_IMAGE="nvidia/cuda:11.8.0-devel-ubuntu22.04"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Perception Model Build Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if base image exists
echo -e "${YELLOW}Checking base image: $BASE_IMAGE${NC}"
if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
    echo -e "${RED}Error: Base image $BASE_IMAGE not found!${NC}"
    echo "Please ensure you have the base image available."
    exit 1
fi
echo -e "${GREEN}✓ Base image found${NC}"
echo ""

# Function to build a model
build_model() {
    local model_name=$1
    local model_dir=$2
    local tag=$3
    
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Building $model_name${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    local dockerfile_path="$PROJECT_ROOT/docker/$model_dir/Dockerfile"
    
    if [ ! -f "$dockerfile_path" ]; then
        echo -e "${RED}Error: Dockerfile not found at $dockerfile_path${NC}"
        return 1
    fi
    
    cd "$PROJECT_ROOT/docker/$model_dir"
    
    # Special handling for SAM2: copy local source code
    if [ "$model_name" = "SAM2" ]; then
        local sam2_source="/data1/yuhongjie2/sam2"
        if [ -d "$sam2_source" ]; then
            echo "Copying local SAM2 source code from $sam2_source"
            cp -r "$sam2_source" ./sam2
            echo -e "${GREEN}✓ Local SAM2 source code copied${NC}"
        else
            echo -e "${RED}Error: Local SAM2 source not found at $sam2_source${NC}"
            return 1
        fi
    fi
    
    # Special handling for RemoteSAM: copy local source code
    if [ "$model_name" = "RemoteSAM" ]; then
        local remotesam_source="/data1/yuhongjie2/RemoteSAM"
        if [ -d "$remotesam_source" ]; then
            echo "Copying local RemoteSAM source code from $remotesam_source"
            cp -r "$remotesam_source" ./RemoteSAM
            echo -e "${GREEN}✓ Local RemoteSAM source code copied${NC}"
        else
            echo -e "${RED}Error: Local RemoteSAM source not found at $remotesam_source${NC}"
            return 1
        fi
    fi
    
    echo "Building image: $tag"
    if docker build -t "$tag" .; then
        echo -e "${GREEN}✓ $model_name built successfully${NC}"
        
        # Clean up copied SAM2 source code
        if [ "$model_name" = "SAM2" ] && [ -d "./sam2" ]; then
            echo "Cleaning up copied SAM2 source code"
            rm -rf ./sam2
            echo -e "${GREEN}✓ Cleanup completed${NC}"
        fi
        
        # Clean up copied RemoteSAM source code
        if [ "$model_name" = "RemoteSAM" ] && [ -d "./RemoteSAM" ]; then
            echo "Cleaning up copied RemoteSAM source code"
            rm -rf ./RemoteSAM
            echo -e "${GREEN}✓ Cleanup completed${NC}"
        fi
        
        echo ""
        return 0
    else
        echo -e "${RED}✗ $model_name build failed${NC}"
        
        # Clean up copied SAM2 source code even on failure
        if [ "$model_name" = "SAM2" ] && [ -d "./sam2" ]; then
            echo "Cleaning up copied SAM2 source code"
            rm -rf ./sam2
        fi
        
        # Clean up copied RemoteSAM source code even on failure
        if [ "$model_name" = "RemoteSAM" ] && [ -d "./RemoteSAM" ]; then
            echo "Cleaning up copied RemoteSAM source code"
            rm -rf ./RemoteSAM
        fi
        
        echo ""
        return 1
    fi
}

# Build SAM2
build_model "SAM2" "sam2" "terrabox/sam2:latest" || exit 1

# Build RemoteCLIP
build_model "RemoteCLIP" "remoteclip" "terrabox/remoteclip:latest" || exit 1

# Build RemoteSAM
build_model "RemoteSAM" "remotesam" "terrabox/remotesam:latest" || exit 1

# Build Strip-RCNN
build_model "Strip-RCNN" "strip_rcnn" "terrabox/strip-rcnn:latest" || exit 1

# Pull vLLM official image
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Pulling vLLM official image${NC}"
echo -e "${YELLOW}========================================${NC}"
echo "Pulling image: vllm/vllm-openai:latest"
if docker pull vllm/vllm-openai:latest; then
    echo -e "${GREEN}✓ vLLM image pulled successfully${NC}"
    echo ""
else
    echo -e "${RED}✗ vLLM image pull failed${NC}"
    echo ""
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All models built successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Built images:"
echo "  - terrabox/sam2:latest"
echo "  - terrabox/remoteclip:latest"
echo "  - terrabox/remotesam:latest"
echo "  - terrabox/strip-rcnn:latest"
echo "  - vllm/vllm-openai:latest (official)"
echo ""
echo "You can now use these images with your Terrabox system."