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

# MMCV + MMDet + MMSegmentation (需要从源安装 mmcv-full)
RUN pip3 install mmcv-full==1.7.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html

# RemoteSAM 官方 requirements (按顺序安装，避免版本冲突)
COPY RemoteSAM/requirements.txt /tmp/requirements.txt

# 先安装不依赖 numpy 的包
RUN pip3 install --no-deps \
    requests filelock ftfy regex h5py tqdm yacs einops termcolor pycocotools shapely

# 安装指定版本的包
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

COPY RemoteSAM /app/
WORKDIR /app

COPY server.py /app/docker_server.py

ENV REMOTESAM_CHECKPOINT="/checkpoints/swin_base_patch4_window12_384_22k.pth"
ENV REMOTESAM_USE_EPOC="true"

EXPOSE 9004
CMD ["python3", "/app/docker_server.py"]
