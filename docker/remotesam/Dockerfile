FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev git \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# torch 1.13 requires numpy < 2.0; pin it before any other install
RUN pip3 install "numpy<2.0"

# RemoteSAM requires torch 1.13; use cu117 wheels
RUN pip3 install --no-cache-dir \
    --progress-bar on \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# mmcv is required by RemoteSAM's lib/mmcv_custom (checkpoint loading)
RUN pip3 install mmcv-full \
    -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html

# Copy requirements.txt to container and install dependencies
# Note: numpy<2.0 is already installed above; skip it to avoid conflicts
COPY RemoteSAM/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-deps -r /tmp/requirements.txt && \
    pip3 install flask opencv-python-headless scipy scikit-learn scikit-image

# Use local RemoteSAM source code (excluding dataset via .dockerignore)
COPY RemoteSAM /app/
WORKDIR /app

# Docker-aware server script (replaces start.py; reads config from env vars)
COPY server.py /app/docker_server.py

# Model checkpoint is volume-mounted at runtime:
#   -v /your/RemoteSAM/pretrained_weights:/checkpoints:ro
# HuggingFace cache (bert-base-uncased + EPOC model) is volume-mounted:
#   -v ~/.cache/huggingface:/root/.cache/huggingface:ro
ENV REMOTESAM_CHECKPOINT="/checkpoints/swin_base_patch4_window12_384_22k.pth"
ENV REMOTESAM_USE_EPOC="true"

EXPOSE 9004
CMD ["python3", "/app/docker_server.py"]