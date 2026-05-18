# ── Source acquisition ────────────────────────────────────────────────────────
# BUILD_MODE=git  (default): clone Strip-RCNN from GitHub — for open-source users
# BUILD_MODE=local:           copy pre-existing local source — for internal builds
#
# Strip-RCNN is a mmrotate fork that registers StripRCNN model types;
# installing it (setup name='mmrotate') makes 'import mmrotate' expose StripRCNN to mmdet.
# Do NOT use the upstream open-mmlab/mmrotate here; it has no StripRCNN definitions.
#
# Usage:
#   docker build -t terrabox/strip-rcnn:latest .                                  # git (default)
#   docker build --build-arg BUILD_MODE=local -t terrabox/strip-rcnn:latest .     # local
#   docker build --build-arg STRIP_RCNN_GIT_URL=https://mirror/Strip-RCNN ...     # custom URL
# ─────────────────────────────────────────────────────────────────────────────

# Stage 1: git clone
FROM ubuntu:22.04 AS source_git
ARG STRIP_RCNN_GIT_URL=https://github.com/wjy5446/Strip-RCNN.git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 "$STRIP_RCNN_GIT_URL" /strip_rcnn

# Stage 2: local copy (build context must contain a 'Strip-RCNN' directory)
FROM ubuntu:22.04 AS source_local
COPY Strip-RCNN /strip_rcnn

# Stage 3: select source based on BUILD_MODE
ARG BUILD_MODE=git
FROM source_${BUILD_MODE} AS source

# Continue from final stage (re-use layers from above via BuildKit cache)
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
# FROM docker.m.daocloud.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev git ninja-build \
    libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install "numpy<2.0"
RUN pip3 install \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"

RUN pip3 install mmcv-full \
    -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
RUN pip3 install mmdet==2.28.2

COPY --from=source /strip_rcnn /strip_rcnn
WORKDIR /strip_rcnn
RUN pip3 install -e .

# Reinstall numpy, scipy, scikit-learn to fix version conflicts
# torch 1.13 + numpy 2.x causes "Numpy is not available" error
RUN pip3 uninstall -y numpy scipy scikit-learn || true
RUN pip3 install "numpy<2.0" "scipy<1.11" "scikit-learn<1.3"

RUN pip3 install flask timm

WORKDIR /app
COPY start.py /app/start.py

# Model config and weights are volume-mounted at runtime:
#   -v /your/Strip-RCNN/configs:/configs:ro
#   -v /your/Strip-RCNN/ckpt:/ckpt:ro
# Paths are passed as argparse arguments by the docker manager.
EXPOSE 9005
ENTRYPOINT ["python3", "/app/start.py"]
