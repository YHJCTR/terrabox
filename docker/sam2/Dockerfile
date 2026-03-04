FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev git \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# PyTorch with CUDA 11.8
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# SAM2 from source
RUN git clone https://github.com/facebookresearch/sam2.git /sam2_src
WORKDIR /sam2_src
RUN pip3 install -e ".[demo]"

# Server script dependencies
RUN pip3 install \
    fastapi \
    uvicorn \
    opencv-python-headless \
    Pillow \
    hydra-core \
    omegaconf \
    rasterio \
    shapely

WORKDIR /app
COPY server.py /app/server.py

# Model checkpoint and config are volume-mounted at runtime:
#   -v /your/sam2/checkpoints:/checkpoints:ro
#   -v /your/sam2/configs:/sam2_configs:ro
ENV SAM2_CHECKPOINT="/checkpoints/sam2.1_hiera_large.pt"
ENV SAM2_CONFIG="/sam2_configs/sam2.1/sam2.1_hiera_l.yaml"

EXPOSE 9002
CMD ["python3", "/app/server.py"]
