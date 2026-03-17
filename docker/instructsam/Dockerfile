FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# ── 系统依赖（与 sam2 镜像保持一致）──────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev git \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# ── PyTorch（cu118，与 sam2 镜像保持一致）──────────────────────────────────
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# ── 从 GitHub 克隆 InstructSAM（含 third_party/sam2）────────────────────
RUN git clone --depth 1 https://github.com/VoyagerXvoyagerx/InstructSAM.git /app/InstructSAM

# ── 安装 SAM2（InstructSAM 依赖的原始 SAM2，非 SAM2.1）──────────────────
# SAM2_BUILD_CUDA=0：跳过 connected_components.cu 编译，推理不需要它
RUN SAM2_BUILD_CUDA=0 pip3 install -e "/app/InstructSAM/third_party/sam2"

# ── 安装 InstructSAM 包本身 ───────────────────────────────────────────────
RUN pip3 install -e "/app/InstructSAM"

# ── 轻量推理依赖（计数通过 HTTP 调宿主机 vLLM，不在此加载 Qwen）────────
RUN pip3 install \
    flask \
    open_clip_torch \
    scipy \
    scikit-learn \
    opencv-python-headless \
    Pillow \
    requests

# ── 服务脚本 ──────────────────────────────────────────────────────────────
COPY server.py /app/server.py
WORKDIR /app

# ── 运行时可覆盖的环境变量 ────────────────────────────────────────────────
ENV SAM2_CHECKPOINT="/models/sam2_hiera_large.pt"
ENV SAM2_CONFIG="sam2_hiera_l.yaml"
ENV CLIP_MODEL_NAME="ViT-L-14"
ENV CLIP_CHECKPOINT="/models/GeoRSCLIP-ViT-L-14.pt"
# 宿主机 vLLM 服务（通过 --add-host host.docker.internal:host-gateway 访问）
ENV VLLM_API_URL="http://host.docker.internal:9000"
ENV VLLM_MODEL_NAME="/model"

EXPOSE 9006
CMD ["python3", "/app/server.py"]
