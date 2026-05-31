# ── Source acquisition ────────────────────────────────────────────────────────
# BUILD_MODE=git  (default): clone RemoteSAM from GitHub — for open-source users
# BUILD_MODE=local:           copy pre-existing local source — for internal builds
#
# Usage:
#   docker build -t terrabox/remotesam:latest .                              # git (default)
#   docker build --build-arg BUILD_MODE=local -t terrabox/remotesam:latest . # local
#   docker build --build-arg REMOTESAM_GIT_URL=https://mirror/RemoteSAM ...  # custom URL
# ─────────────────────────────────────────────────────────────────────────────

ARG BUILD_MODE=git

# Stage 1: git clone
FROM ubuntu:22.04 AS source_git
ARG REMOTESAM_GIT_URL=https://github.com/xiaobdul/RemoteSAM.git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 "$REMOTESAM_GIT_URL" /app

# Stage 2: local copy (build context must contain a 'RemoteSAM' directory)
FROM ubuntu:22.04 AS source_local
COPY RemoteSAM /app

# Stage 3: select source based on BUILD_MODE
FROM source_${BUILD_MODE} AS source

# Final stage
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev git \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# PyTorch 1.13.1 + CUDA 11.7
RUN pip3 install --no-cache-dir \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# MMCV + MMDet + MMSegmentation (mmcv-full must be installed from the prebuilt wheel)
RUN pip3 install mmcv-full==1.7.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html

# Install packages that do not depend on numpy first
RUN pip3 install --no-deps \
    requests filelock ftfy regex h5py tqdm yacs einops termcolor pycocotools shapely

# Install pinned package versions
RUN pip3 install \
    "opencv-python==4.5.5.62" \
    "matplotlib==3.6.1" \
    "scikit-image==0.20.0" \
    "scipy==1.9.2" \
    "timm==0.6.11" \
    "tokenizers==0.13.1" \
    "transformers==4.30.2" \
    "Pillow" \
    "mmdet==2.28.2" \
    "mmsegmentation==0.17.0" \
    "scikit-learn" \
    "pytorch-ignite" \
    "accelerate==0.21.0" \
    "torchmetrics" \
    "spacy==3.4.4"

# Flask for API server
RUN pip3 install flask importlib-resources

COPY --from=source /app /app
WORKDIR /app

COPY server.py /app/docker_server.py

ENV REMOTESAM_CHECKPOINT="/checkpoints/swin_base_patch4_window12_384_22k.pth"
ENV REMOTESAM_USE_EPOC="true"

EXPOSE 9004
CMD ["python3", "/app/docker_server.py"]
