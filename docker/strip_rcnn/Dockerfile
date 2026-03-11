FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
# FROM docker.m.daocloud.io/nvidia/cuda:11.8.0-devel-ubuntu22.04
# docker.xuanyuan.me  docker.m.daocloud.io  docker.1ms.run

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev git ninja-build \
    libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# torch 1.13 requires numpy < 2.0; pin it before any other install to prevent numpy 2.x
RUN pip3 install "numpy<2.0"

# mmcv/mmdet are sensitive to PyTorch version; use the verified cu117/torch1.13 combination
RUN pip3 install \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Required CUDA architecture flags for mmcv-full compilation
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"

# Pre-built mmcv-full for cu117/torch1.13
RUN pip3 install mmcv-full \
    -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html

RUN pip3 install mmdet==2.28.2

# Use local Strip-RCNN source code (ckpt excluded via .dockerignore).
# Strip-RCNN is a mmrotate fork that registers StripRCNN model types;
# installing it (setup name='mmrotate') makes 'import mmrotate' expose StripRCNN to mmdet.
# Do NOT use the upstream open-mmlab/mmrotate here; it has no StripRCNN definitions.
#
# Before running docker build, copy the source into the build context:
#   cp -r /data1/yuhongjie2/Strip-RCNN docker/strip_rcnn/Strip-RCNN
# Or use build_models.sh which handles this automatically.
#
# (Alternative: git clone, but requires internet access inside the build environment)
# RUN git clone https://github.com/someuser2026/Strip-RCNN.git /strip_rcnn
# RUN git clone https://bgithub.xyz/someuser2026/Strip-RCNN.git /strip_rcnn  # China mirror
COPY Strip-RCNN /strip_rcnn
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
