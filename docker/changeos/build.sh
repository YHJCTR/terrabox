#!/bin/bash
# Build the terrabox/changeos image (faithful torch==1.10.0 ChangeOS service).
#
# Weights are NOT baked in; they are volume-mounted at runtime from
#   /data1/yuhongjie2/ChangeOS/checkpoints  (changeos_r101.pt / changeos_r18.pt)
#
# Usage:
#   bash build.sh                     # builds terrabox/changeos:latest
#   bash build.sh terrabox/changeos:v2
#
# The build pulls torch 1.10.0+cu113 and the changeos SDK over the network, so
# the host proxy must be reachable from docker build (BuildKit inherits the
# shell's http_proxy/https_proxy by default on this host).

set -e
cd "$(dirname "$0")"

TAG="${1:-terrabox/changeos:latest}"

echo "Building $TAG ..."
# --network=host so the host proxy (127.0.0.1:7897) is reachable from RUN steps;
# without it, 127.0.0.1 would point at the build container itself.
docker build \
    --network=host \
    --build-arg http_proxy="${http_proxy:-}" \
    --build-arg https_proxy="${https_proxy:-}" \
    --build-arg no_proxy="${no_proxy:-localhost,127.0.0.1}" \
    -t "$TAG" .

echo "Done: $TAG"
