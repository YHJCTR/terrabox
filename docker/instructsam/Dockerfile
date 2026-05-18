FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# ── System dependencies (matching the sam2 image) ────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev git \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# ── PyTorch (cu118, matching the sam2 image) ─────────────────────────────────
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# ── Clone InstructSAM from GitHub (includes third_party/sam2) ───────────────
# Override with --build-arg INSTRUCTSAM_GIT_URL=https://mirror/... for a mirror
ARG INSTRUCTSAM_GIT_URL=https://github.com/VoyagerXvoyagerx/InstructSAM.git
RUN git clone --depth 1 "$INSTRUCTSAM_GIT_URL" /app/InstructSAM

# ── Install SAM2 (the original SAM2 required by InstructSAM, not SAM2.1) ─────
# SAM2_BUILD_CUDA=0: skip connected_components.cu compilation; not needed for inference
RUN SAM2_BUILD_CUDA=0 pip3 install -e "/app/InstructSAM/third_party/sam2"

# ── Install the InstructSAM package itself ───────────────────────────────────
RUN pip3 install -e "/app/InstructSAM"

# ── Lightweight inference dependencies (counting done via HTTP to host vLLM, no Qwen) ──
RUN pip3 install \
    flask \
    open_clip_torch \
    scipy \
    scikit-learn \
    opencv-python-headless \
    Pillow \
    requests

# ── Service script ────────────────────────────────────────────────────────────
COPY server.py /app/server.py
WORKDIR /app

# ── Runtime-overridable environment variables ─────────────────────────────────
ENV SAM2_CHECKPOINT="/models/sam2_hiera_large.pt"
ENV SAM2_CONFIG="sam2_hiera_l.yaml"
ENV CLIP_MODEL_NAME="ViT-L-14"
ENV CLIP_CHECKPOINT="/models/GeoRSCLIP-ViT-L-14.pt"
# Host vLLM service (accessed via --add-host host.docker.internal:host-gateway)
ENV VLLM_API_URL="http://host.docker.internal:9000"
ENV VLLM_MODEL_NAME="/model"

EXPOSE 9006
CMD ["python3", "/app/server.py"]
