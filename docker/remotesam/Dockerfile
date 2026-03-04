FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev git \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# RemoteSAM requires torch 1.13; use cu117 wheels
RUN pip3 install \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

RUN pip3 install \
    timm \
    transformers \
    flask \
    opencv-python-headless \
    scipy \
    einops \
    Pillow

# Source code is volume-mounted at runtime (includes pretrained_weights/ and local task imports):
#   -v /your/RemoteSAM:/app:ro
#   -w /app

EXPOSE 9004
CMD ["python3", "/app/start.py"]
