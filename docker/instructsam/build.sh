#!/usr/bin/env bash
# Build the InstructSAM Docker image.
# Usage: cd docker/instructsam && bash build.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
docker build -t terrabox/instructsam:latest .
echo "Build complete: terrabox/instructsam:latest"
