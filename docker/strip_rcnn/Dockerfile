# FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
# For China
FROM docker.m.daocloud.io/nvidia/cuda:11.8.0-devel-ubuntu22.04
# docker.xuanyuan.me  docker.m.daocloud.io  docker.1ms.run

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev git ninja-build \
    libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

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

# mmrotate is required for Strip-RCNN registered types
# RUN git clone https://github.com/open-mmlab/mmrotate.git /mmrotate
# For China
RUN git clone https://bgithub.xyz/open-mmlab/mmrotate.git /mmrotate
WORKDIR /mmrotate
RUN pip3 install -r requirements/build.txt && pip3 install -e .

RUN pip3 install flask

WORKDIR /app
COPY start.py /app/start.py

# Model config and weights are volume-mounted at runtime:
#   -v /your/Strip-RCNN/configs:/configs:ro
#   -v /your/Strip-RCNN/ckpt:/ckpt:ro
# Paths are passed as argparse arguments by the docker manager.
EXPOSE 9005
ENTRYPOINT ["python3", "/app/start.py"]
