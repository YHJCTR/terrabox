#!/bin/bash
# Build the terrabox/sam2 Docker image.
#
# SOURCE MODE — two options:
#   local (default)  Copy from a local directory. Fast, no network needed.
#   git              Clone from GitHub. Useful for a clean/reproducible build.
#
# Usage:
#   bash build.sh                          # local copy, tag=terrabox/sam2:latest
#   bash build.sh terrabox/sam2:v2         # custom tag
#   SOURCE_MODE=git bash build.sh          # clone from GitHub
#   bash build.sh --git                    # same as above
#
# Overridable env vars:
#   SAM2_SRC        Local source path  (default: /data1/yuhongjie2/sam2)
#   SAM2_GIT_URL    Git clone URL      (default: GitHub facebookresearch/sam2)
#   SOURCE_MODE     "local" or "git"   (default: local)

set -e
cd "$(dirname "$0")"   # always run from this script's directory

# --- Configuration -------------------------------------------------------
TAG="terrabox/sam2:latest"
SOURCE_MODE="${SOURCE_MODE:-local}"

for arg in "$@"; do
    case "$arg" in
        --git) SOURCE_MODE=git ;;
        *)     TAG="$arg" ;;
    esac
done

# Local mode: path to existing source
LOCAL_SRC="${SAM2_SRC:-/data1/yuhongjie2/sam2}"

# Git mode: repository URL (swap for a mirror if GitHub is unreachable)
GIT_URL="${SAM2_GIT_URL:-https://github.com/facebookresearch/sam2.git}"
# GIT_URL="https://bgithub.xyz/facebookresearch/sam2.git"   # China mirror

CONTEXT_DIR="sam2"   # must match Dockerfile: COPY sam2 /sam2_src
# -------------------------------------------------------------------------

cleanup() {
    [ -d "./$CONTEXT_DIR" ] && rm -rf "./$CONTEXT_DIR"
}
trap cleanup EXIT   # clean up even if the build fails

if [ "$SOURCE_MODE" = "git" ]; then
    echo "Cloning $GIT_URL ..."
    git clone "$GIT_URL" "./$CONTEXT_DIR"
else
    if [ ! -d "$LOCAL_SRC" ]; then
        echo "Error: SAM2 source not found at $LOCAL_SRC" >&2
        exit 1
    fi
    echo "Copying SAM2 source from $LOCAL_SRC ..."
    cp -r "$LOCAL_SRC" "./$CONTEXT_DIR"
fi

echo "Building $TAG ..."
docker build -t "$TAG" .

echo "Done: $TAG"
