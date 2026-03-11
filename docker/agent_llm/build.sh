#!/bin/bash
# Build the terrabox/agent-llm Docker image.
#
# Usage:
#   bash build.sh                    # tag=terrabox/agent-llm:latest
#   bash build.sh terrabox/agent-llm:v2   # custom tag

set -e
cd "$(dirname "$0")"

TAG="${1:-terrabox/agent-llm:latest}"

echo "Building $TAG ..."
docker build -t "$TAG" .

echo "Done: $TAG"
