#!/bin/bash
# Build the terrabox/codegen-sandbox Docker image.
#
# Usage:
#   bash build.sh                              # tag=terrabox/codegen-sandbox:latest
#   bash build.sh terrabox/codegen-sandbox:v2  # custom tag

set -e
cd "$(dirname "$0")"

TAG="${1:-terrabox/codegen-sandbox:latest}"

echo "Building $TAG ..."
docker build -t "$TAG" .

echo "Done: $TAG"
