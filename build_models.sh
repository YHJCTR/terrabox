#!/bin/bash

# Build script for perception model containers
# This script builds all perception model Docker images.
#
# BUILD_MODE controls how source code is acquired inside the Docker image:
#   local (default) — copy from the host machine (fast, no internet needed)
#   git             — clone from GitHub at build time (for open-source users)
#
# Usage examples:
#   ./build_models.sh                           # local mode (default, uses paths below)
#   BUILD_MODE=git ./build_models.sh            # git mode (no local source needed)
#   BUILD_MODE=git SAM2_GIT_URL=https://bgithub.xyz/facebookresearch/sam2.git \
#     ./build_models.sh                         # git mode with custom mirror URL

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Build mode: local (default, backward-compatible) or git (open-source friendly)
BUILD_MODE="${BUILD_MODE:-local}"

# Git URLs (only used when BUILD_MODE=git; override via environment variable for mirrors)
SAM2_GIT_URL="${SAM2_GIT_URL:-https://github.com/facebookresearch/sam2.git}"
REMOTESAM_GIT_URL="${REMOTESAM_GIT_URL:-https://github.com/xiaobdul/RemoteSAM.git}"
STRIP_RCNN_GIT_URL="${STRIP_RCNN_GIT_URL:-https://github.com/wjy5446/Strip-RCNN.git}"
INSTRUCTSAM_GIT_URL="${INSTRUCTSAM_GIT_URL:-https://github.com/VoyagerXvoyagerx/InstructSAM.git}"

# Local source paths (only used when BUILD_MODE=local)
SAM2_SOURCE="${SAM2_SOURCE:-/data1/yuhongjie2/sam2}"
REMOTESAM_SOURCE="${REMOTESAM_SOURCE:-/data1/yuhongjie2/RemoteSAM}"
STRIP_RCNN_SOURCE="${STRIP_RCNN_SOURCE:-/data1/yuhongjie2/Strip-RCNN}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Perception Model Build Script${NC}"
echo -e "${GREEN}Build mode: ${BUILD_MODE}${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check base image (only in local mode where nvidia image may be from local registry)
BASE_IMAGE="nvidia/cuda:11.8.0-devel-ubuntu22.04"
if [ "$BUILD_MODE" = "local" ]; then
    echo -e "${YELLOW}Checking base image: $BASE_IMAGE${NC}"
    if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
        echo -e "${RED}Error: Base image $BASE_IMAGE not found!${NC}"
        echo "Please ensure you have the base image available."
        exit 1
    fi
    echo -e "${GREEN}✓ Base image found${NC}"
    echo ""
fi

# Function to build a model
# Arguments:
#   $1 model_name   - display name (e.g. "SAM2")
#   $2 model_dir    - subdirectory under docker/ (e.g. "sam2")
#   $3 tag          - docker image tag (e.g. "terrabox/sam2:latest")
#   $4 local_source - host path to source (only used in local mode, optional)
#   $5 git_url_arg  - build-arg string for git URL (e.g. "SAM2_GIT_URL=https://...")
build_model() {
    local model_name=$1
    local model_dir=$2
    local tag=$3
    local local_source="${4:-}"
    local git_url_arg="${5:-}"

    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Building $model_name (mode=$BUILD_MODE)${NC}"
    echo -e "${YELLOW}========================================${NC}"

    local dockerfile_path="$PROJECT_ROOT/docker/$model_dir/Dockerfile"
    if [ ! -f "$dockerfile_path" ]; then
        echo -e "${RED}Error: Dockerfile not found at $dockerfile_path${NC}"
        return 1
    fi

    cd "$PROJECT_ROOT/docker/$model_dir"

    local copied_dir=""

    if [ "$BUILD_MODE" = "local" ] && [ -n "$local_source" ]; then
        # Copy local source into build context
        local dir_name
        dir_name="$(basename "$local_source")"
        if [ -d "$local_source" ]; then
            echo "Copying local source code from $local_source"
            cp -r "$local_source" "./$dir_name"
            copied_dir="$dir_name"
            echo -e "${GREEN}✓ Source code copied${NC}"
        else
            echo -e "${RED}Error: Local source not found at $local_source${NC}"
            return 1
        fi
    fi

    # Build the image
    local build_args="--build-arg BUILD_MODE=$BUILD_MODE"
    if [ "$BUILD_MODE" = "git" ] && [ -n "$git_url_arg" ]; then
        build_args="$build_args --build-arg $git_url_arg"
    fi

    echo "Building image: $tag"
    if docker build $build_args -t "$tag" .; then
        echo -e "${GREEN}✓ $model_name built successfully${NC}"
    else
        echo -e "${RED}✗ $model_name build failed${NC}"
        [ -n "$copied_dir" ] && rm -rf "./$copied_dir"
        return 1
    fi

    # Clean up copied source
    if [ -n "$copied_dir" ] && [ -d "./$copied_dir" ]; then
        echo "Cleaning up copied source code"
        rm -rf "./$copied_dir"
        echo -e "${GREEN}✓ Cleanup completed${NC}"
    fi

    echo ""
    return 0
}

# Build SAM2
build_model "SAM2" "sam2" "terrabox/sam2:latest" \
    "$SAM2_SOURCE" "SAM2_GIT_URL=$SAM2_GIT_URL" || exit 1

# Build RemoteCLIP (no third-party source; start.py is already in docker/remoteclip/)
build_model "RemoteCLIP" "remoteclip" "terrabox/remoteclip:latest" || exit 1

# Build RemoteSAM
build_model "RemoteSAM" "remotesam" "terrabox/remotesam:latest" \
    "$REMOTESAM_SOURCE" "REMOTESAM_GIT_URL=$REMOTESAM_GIT_URL" || exit 1

# Build Strip-RCNN
build_model "Strip-RCNN" "strip_rcnn" "terrabox/strip-rcnn:latest" \
    "$STRIP_RCNN_SOURCE" "STRIP_RCNN_GIT_URL=$STRIP_RCNN_GIT_URL" || exit 1

# Build InstructSAM (git-only; no local source needed — always cloned from GitHub)
build_model "InstructSAM" "instructsam" "terrabox/instructsam:latest" \
    "" "INSTRUCTSAM_GIT_URL=$INSTRUCTSAM_GIT_URL" || exit 1

# Build vLLM (custom image wrapping vllm/vllm-openai:latest; no third-party source)
build_model "vLLM" "vllm" "terrabox/vllm:latest" || exit 1

# Build Agent LLM (custom image wrapping vllm/vllm-openai:latest; no third-party source)
build_model "Agent-LLM" "agent_llm" "terrabox/agent-llm:latest" || exit 1

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All models built successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Built images:"
echo "  - terrabox/sam2:latest"
echo "  - terrabox/remoteclip:latest"
echo "  - terrabox/remotesam:latest"
echo "  - terrabox/strip-rcnn:latest"
echo "  - terrabox/instructsam:latest"
echo "  - terrabox/vllm:latest"
echo "  - terrabox/agent-llm:latest"
echo ""
echo "You can now use these images with your Terrabox system."
