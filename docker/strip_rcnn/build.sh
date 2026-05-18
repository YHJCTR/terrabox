#!/bin/bash
# Build the terrabox/strip-rcnn Docker image directly.
# Copies Strip-RCNN source into the build context, builds, then cleans up.
#
# Usage:
#   bash build.sh                        # builds terrabox/strip-rcnn:latest
#   bash build.sh terrabox/strip-rcnn:v2 # custom tag
#
# Override source path:
#   STRIP_RCNN_SRC=/other/path bash build.sh

set -e
cd "$(dirname "$0")"   # always run from this script's directory

STRIP_RCNN_SRC="${STRIP_RCNN_SRC:-/data1/yuhongjie2/Strip-RCNN}"
TAG="${1:-terrabox/strip-rcnn:latest}"

cleanup() {
    [ -d "./Strip-RCNN" ] && rm -rf ./Strip-RCNN
}
trap cleanup EXIT   # clean up even if build fails

if [ ! -d "$STRIP_RCNN_SRC" ]; then
    echo "Error: Strip-RCNN source not found at $STRIP_RCNN_SRC" >&2
    exit 1
fi

echo "Copying Strip-RCNN source from $STRIP_RCNN_SRC ..."
cp -r "$STRIP_RCNN_SRC" ./Strip-RCNN

echo "Building $TAG ..."
docker build -t "$TAG" .

echo "Done: $TAG"
