# ChangeOS building-damage assessment service (RSE 2021, Z-Zheng/ChangeOS).
#
# FAITHFUL reproduction of the official environment:
#   - torch == 1.10.0 (+cu113)  -> the released .pt are TorchScript exported on 1.10
#   - python 3.9                -> torch 1.10.0 has NO cp310 wheel, so ubuntu22.04's
#                                  default python3.10 cannot be used; install 3.9.
#   - changeos SDK installed --no-deps so we pin torch ourselves (its setup.py
#     would otherwise drag in an arbitrary torch).
#
# Base: nvidia/cuda:11.8.0-devel-ubuntu22.04 (already present on the host).
# The torch+cu113 wheel bundles its own CUDA 11.3 runtime libs, which run fine on
# the host's driver 535 / RTX 3090 (sm_86) via backward compatibility — the base
# image's 11.8 toolkit is only there to satisfy the container, not used by torch.
#
# Weights are volume-mounted at runtime (NOT baked in):
#   -v /data1/yuhongjie2/ChangeOS/checkpoints:/ckpt:ro
# and selected via --checkpoint /ckpt/changeos_r101.pt
#
# Build:
#   cd docker/changeos && docker build -t terrabox/changeos:latest .

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# python3.9 (deadsnakes) — required for the torch 1.10.0 cp39 wheel
RUN apt-get update && apt-get install -y \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
        python3.9 python3.9-dev python3.9-distutils \
        curl git \
        libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# pip for python3.9
RUN curl -sS https://bootstrap.pypa.io/pip/3.9/get-pip.py | python3.9
RUN ln -sf /usr/bin/python3.9 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.9 /usr/local/bin/python3

# Faithful torch stack: 1.10.0 + cu113 (and matching torchvision 0.11.1)
RUN python3.9 -m pip install --no-cache-dir \
        torch==1.10.0+cu113 torchvision==0.11.1+cu113 \
        --extra-index-url https://download.pytorch.org/whl/cu113

# Runtime deps. changeos installed --no-deps so it cannot override torch.
# Flask pinned to 2.2.5 (+Werkzeug<2.3): newer Flask 3.x requires upgrading
# blinker, but the apt-provided blinker 1.4 lives in the shared distutils
# dist-packages and cannot be uninstalled by pip. Flask 2.2 has no hard blinker
# dep, sidestepping the conflict.
RUN python3.9 -m pip install --no-cache-dir \
        "numpy==1.23.5" \
        "scikit-image==0.19.3" \
        "Pillow<10" \
        "flask==2.2.5" "Werkzeug<2.3"
RUN python3.9 -m pip install --no-cache-dir --no-deps changeos==0.2

WORKDIR /app
COPY start.py /app/start.py

EXPOSE 9007
ENTRYPOINT ["python3.9", "/app/start.py"]
