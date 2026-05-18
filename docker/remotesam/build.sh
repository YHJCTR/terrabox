#!/bin/bash
# Build the terrabox/remotesam Docker image.
#
# SOURCE MODE — two options:
#   local (default)  Copy from a local directory. Fast, no network needed.
#   git              Clone from GitHub. Useful for a clean/reproducible build.
#
# Usage:
#   bash build.sh                              # local copy, tag=terrabox/remotesam:latest
#   bash build.sh terrabox/remotesam:v2        # custom tag
#   SOURCE_MODE=git bash build.sh              # clone from GitHub
#   bash build.sh --git                        # same as above
#
# Overridable env vars:
#   REMOTESAM_SRC       Local source path  (default: /data1/yuhongjie2/RemoteSAM)
#   REMOTESAM_GIT_URL   Git clone URL      (default: GitHub 1e12Leon/RemoteSAM)
#   SOURCE_MODE         "local" or "git"   (default: local)

set -e
cd "$(dirname "$0")"   # always run from this script's directory

# --- Configuration -------------------------------------------------------
TAG="terrabox/remotesam:latest"
SOURCE_MODE="${SOURCE_MODE:-local}"

for arg in "$@"; do
    case "$arg" in
        --git) SOURCE_MODE=git ;;
        *)     TAG="$arg" ;;
    esac
done

# Local mode: path to existing source
LOCAL_SRC="${REMOTESAM_SRC:-/data1/yuhongjie2/RemoteSAM}"

# Git mode: repository URL (swap for a mirror if GitHub is unreachable)
GIT_URL="${REMOTESAM_GIT_URL:-https://github.com/1e12Leon/RemoteSAM.git}"
# GIT_URL="https://bgithub.xyz/1e12Leon/RemoteSAM.git"   # China mirror

CONTEXT_DIR="RemoteSAM"   # must match Dockerfile: COPY RemoteSAM /app/
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
        echo "Error: RemoteSAM source not found at $LOCAL_SRC" >&2
        exit 1
    fi
    echo "Copying RemoteSAM source from $LOCAL_SRC ..."
    
    # Parse .dockerignore and exclude files when copying
    EXCLUDE_ARGS=""
    if [ -f ".dockerignore" ]; then
        while IFS= read -r line; do
            # Skip empty lines and comments
            if [ -z "$line" ] || [[ "$line" =~ ^# ]]; then
                continue
            fi
            # Remove leading/trailing whitespace
            line=$(echo "$line" | xargs)
            if [ -n "$line" ]; then
                EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude='$line'"
            fi
        done < .dockerignore
    fi
    
    # Use rsync with excludes, or cp if rsync not available
    if command -v rsync &> /dev/null; then
        eval rsync -av $EXCLUDE_ARGS "$LOCAL_SRC/" "./$CONTEXT_DIR/"
    else
        # Fallback to cp if rsync not available
        cp -r "$LOCAL_SRC" "./$CONTEXT_DIR"
    fi
fi

echo "Building $TAG ..."
docker build -t "$TAG" .

echo "Done: $TAG"
