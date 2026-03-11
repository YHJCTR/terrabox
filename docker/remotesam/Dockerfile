FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev git \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN pip3 install "setuptools<70.0" wheel packaging

RUN pip3 install "numpy==1.23.5"

RUN pip3 install --no-cache-dir \
    --progress-bar on \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

RUN pip3 install mmcv-full \
    -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html

COPY RemoteSAM/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-deps -r /tmp/requirements.txt && \
    pip3 install flask opencv-python-headless scipy scikit-learn scikit-image huggingface_hub safetensors && \
    pip3 install "setuptools<70.0" spacy thinc && \
    pip3 install pyyaml jsonpatch terminaltables addict yapf pytest && \
    pip3 install Pillow sympy onnx onnxruntime ninja

COPY RemoteSAM /app/
WORKDIR /app

COPY server.py /app/docker_server.py

ENV REMOTESAM_CHECKPOINT="/checkpoints/swin_base_patch4_window12_384_22k.pth"
ENV REMOTESAM_USE_EPOC="true"

EXPOSE 9004
CMD ["python3", "/app/docker_server.py"]