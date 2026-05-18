FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# PyTorch with CUDA 11.8
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install open_clip_torch flask Pillow

WORKDIR /app
COPY start.py /app/start.py

# Checkpoint directory is volume-mounted at runtime:
#   -v /your/RemoteCLIP/checkpoints:/checkpoints:ro
ENV REMOTECLIP_CKPT_DIR="/checkpoints"

EXPOSE 9003
CMD ["python3", "/app/start.py"]
