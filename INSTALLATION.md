# Perception Models & Agent — Installation Guide

This guide covers how to install and configure the AI perception model services and the
Agent LLM. For the main Terrabox API installation (Python environment, database, startup),
see [README.md](README.md).

---

## Table of Contents

1. [Service Overview](#1-service-overview)
2. [Model Weights](#2-model-weights)
3. [Perception Models — Non-Docker Mode](#3-perception-models--non-docker-mode)
4. [Perception Models — Docker Mode](#4-perception-models--docker-mode)
5. [Building Docker Images](#5-building-docker-images)
   - [Build all images at once](#build-all-images-at-once)
   - [Build a single image](#build-a-single-image)
6. [Agent LLM](#6-agent-llm)
   - [Option A: Remote API](#option-a-remote-api-simplest)
   - [Option B: Local LLM subprocess](#option-b-local-llm-subprocess)
   - [Option C: Local LLM Docker container](#option-c-local-llm-docker-container)
7. [Port Reference](#7-port-reference)

---

## 1. Service Overview

Terrabox manages seven AI services. Each is started on demand when the corresponding
tool is first called — you do not need to launch them manually.

| Service | Port | Description |
|---------|------|-------------|
| vLLM (vision) | 9000 | Vision-language model (Qwen-VL or compatible) |
| SAM2 | 9002 | Segment Anything Model 2 — general image segmentation |
| RemoteCLIP | 9003 | Zero-shot classification and retrieval for remote sensing |
| RemoteSAM | 9004 | Text-prompted segmentation for remote sensing images |
| Strip-RCNN | 9005 | Rotated object detection on aerial imagery (DOTA) |
| InstructSAM | 9006 | Instruction-driven instance segmentation and counting |
| Agent LLM | 9100 | LLM powering the Agent reasoning loop |

**Deployment mode** is controlled by `TERRABOX_USE_DOCKER` in `.env`:

| Value | Mode | Description |
|-------|------|-------------|
| `false` (default) | Subprocess | Each service runs as a subprocess inside its own conda environment |
| `true` | Docker | Each service runs inside an isolated Docker container |

---

## 2. Model Weights

All AI services require pre-downloaded model weights. Run `scripts/download_weights.sh` once
before starting any service. The script downloads each model to `./models/<name>/` by
default and prints the exact paths to put in your `.env` / `agent_config.yaml`.

### Prerequisites

```bash
pip install huggingface_hub   # provides huggingface-cli
pip install gdown             # required for Strip-RCNN (Google Drive)
```

### Download all weights

```bash
scripts/download_weights.sh
```

The Agent LLM and VLM model IDs default to `Qwen/Qwen3-8B` and
`Qwen/Qwen2.5-VL-7B-Instruct`. Override them before running the script:

```bash
AGENT_LLM_HF_REPO=your-org/your-llm \
VLM_HF_REPO=your-org/your-vlm \
scripts/download_weights.sh
```

Skip models you already have or do not need:

```bash
scripts/download_weights.sh --skip-agent --skip-vlm
```

Available skip flags: `--skip-agent`, `--skip-vlm`, `--skip-sam2`,
`--skip-remoteclip`, `--skip-remotesam`, `--skip-strip-rcnn`, `--skip-instructsam`

If HuggingFace is slow or blocked, use a mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com scripts/download_weights.sh
```

### Weights reference

| Service | Downloaded file(s) | Default directory | `.env` / config key |
|---------|-------------------|-------------------|---------------------|
| Agent LLM | `Qwen3-8B/` (full dir) | `models/agent_llm/` | `agent_config.yaml` → `local_llm_model_path` |
| VLM | `Qwen2.5-VL-7B-Instruct/` (full dir) | `models/vlm/` | `VLM_MODEL_PATH` |
| SAM2 | `sam2.1_hiera_large.pt` | `models/sam2_checkpoints/` | `SAM2_CHECKPOINT_HOST` (Docker) / `SAM2_WORK_DIR` (subprocess) |
| RemoteCLIP | `RemoteCLIP-ViT-L-14.pt` | `models/remoteclip/` | `REMOTECLIP_CKPT_HOST` |
| RemoteSAM | `swin_base_patch4_window12_384_22k.pth` | `models/remotesam/` | `REMOTESAM_CHECKPOINT_HOST` |
| Strip-RCNN | `stripnet_s.pth` | `models/strip_rcnn/` | `STRIP_RCNN_CKPT_HOST` |
| InstructSAM | `sam2_hiera_large.pt` + `GeoRSCLIP-ViT-L-14.pt` | `models/instructsam/` | `INSTRUCTSAM_MODELS_HOST` |

> **Note on InstructSAM vs SAM2:** InstructSAM uses an older SAM2 release
> (`sam2_hiera_large.pt`, July 2024). The standalone SAM2 service uses the newer
> `sam2.1_hiera_large.pt` (September 2024). These are different files stored in
> separate directories.

> **Note on RemoteSAM BERT:** `bert-base-uncased` is downloaded automatically by
> HuggingFace `transformers` on first run. In Docker mode, pre-cache it by running
> the container once with internet access, or mount your HuggingFace cache via
> `HF_CACHE_HOST` in `.env.docker`.

### Override destination directories

Each directory can be overridden via an environment variable before running the script:

```bash
SAM2_CKPT_DIR=/data1/models/sam2 \
REMOTECLIP_DIR=/data1/models/clip \
scripts/download_weights.sh --skip-agent --skip-vlm
```

---

## 3. Perception Models — Non-Docker Mode

### Prerequisites

- conda installed
- A separate conda environment created and configured for each model
- Model weights downloaded to local disk
- Refer to each model's upstream repository README for environment setup instructions

### Configuration

Copy the example file and edit it:

```bash
cp .env.example .env
```

Replace the placeholder paths with your actual paths:

```dotenv
TERRABOX_USE_DOCKER=false

# ── vLLM (vision-language model, port 9000) ──────────────────────────────────
VLLM_PYTHON_EXEC=/your/conda/envs/YOUR_ENV/bin/python   # conda env with vllm installed
VLM_MODEL_PATH=/path/to/your/qwen_vl_model/             # model weights directory
VLM_GPU_DEVICES=0                                        # GPU index(es)
VLM_TENSOR_PARALLEL_SIZE=1

# ── SAM2 (port 9002) ─────────────────────────────────────────────────────────
SAM2_PYTHON_EXEC=/your/conda/envs/sam2/bin/python
SAM2_SERVER_SCRIPT=/path/to/sam2/sam2_server2.py
SAM2_WORK_DIR=/path/to/sam2
SAM2_GPU_DEVICES=0

# ── RemoteCLIP (port 9003) ───────────────────────────────────────────────────
REMOTECLIP_PYTHON_EXEC=/your/conda/envs/RemoteCLIP/bin/python
REMOTECLIP_SERVER_SCRIPT=/path/to/RemoteCLIP/start2.py
REMOTECLIP_WORK_DIR=/path/to/RemoteCLIP
REMOTECLIP_GPU_DEVICES=0

# ── RemoteSAM (port 9004) ────────────────────────────────────────────────────
REMOTESAM_PYTHON_EXEC=/your/conda/envs/RemoteSAM/bin/python
REMOTESAM_SERVER_SCRIPT=/path/to/RemoteSAM/start.py
REMOTESAM_WORK_DIR=/path/to/RemoteSAM
REMOTESAM_GPU_DEVICES=0

# ── Strip-RCNN (port 9005) ───────────────────────────────────────────────────
STRIP_RCNN_PYTHON_EXEC=/your/conda/envs/strip/bin/python
STRIP_RCNN_SERVER_SCRIPT=/path/to/Strip-RCNN/start.py
STRIP_RCNN_WORK_DIR=/path/to/Strip-RCNN
STRIP_RCNN_GPU_DEVICES=0
STRIP_RCNN_CONFIG_PATH=/path/to/Strip-RCNN/configs/strip_rcnn/orig/strip_rcnn_s_fpn_1x_dota_le90.py
STRIP_RCNN_CHECKPOINT_PATH=/path/to/Strip-RCNN/ckpt/stripnet_s.pth
```

Port numbers can be overridden by uncommenting the corresponding `*_PORT` variable.

---

## 4. Perception Models — Docker Mode

### Prerequisites

- Docker with NVIDIA Container Toolkit installed
- All perception model images built (see [Section 4](#4-building-docker-images))
- Model weights downloaded to the host machine

### Configuration

```bash
cp .env.docker.example .env
```

Replace the placeholder paths with host-side paths to your weights:

```dotenv
TERRABOX_USE_DOCKER=true

# ── vLLM ─────────────────────────────────────────────────────────────────────
VLM_MODEL_PATH=/host/path/to/qwen_vl_model/    # mounted as /model inside container
VLM_GPU_DEVICES=0
VLM_TENSOR_PARALLEL_SIZE=1

# ── SAM2 ─────────────────────────────────────────────────────────────────────
SAM2_GPU_DEVICES=0
SAM2_CHECKPOINT_HOST=/host/path/to/sam2/checkpoints  # contains sam2.1_hiera_large.pt
SAM2_CONFIG_HOST=/host/path/to/sam2/sam2/configs      # sam2 configs directory

# ── RemoteCLIP ───────────────────────────────────────────────────────────────
REMOTECLIP_GPU_DEVICES=0
REMOTECLIP_CKPT_HOST=/host/path/to/RemoteCLIP/checkpoints

# ── RemoteSAM ────────────────────────────────────────────────────────────────
REMOTESAM_GPU_DEVICES=0
REMOTESAM_CHECKPOINT_HOST=/host/path/to/RemoteSAM/pretrained_weights

# ── Strip-RCNN ───────────────────────────────────────────────────────────────
STRIP_RCNN_GPU_DEVICES=0
STRIP_RCNN_CKPT_HOST=/host/path/to/Strip-RCNN/ckpt
STRIP_RCNN_CONFIG_HOST=/host/path/to/Strip-RCNN/configs

# ── InstructSAM ──────────────────────────────────────────────────────────────
INSTRUCTSAM_GPU_DEVICES=0
# Directory must contain: sam2_hiera_large.pt and GeoRSCLIP-ViT-L-14.pt
INSTRUCTSAM_MODELS_HOST=/host/path/to/instructsam/models

# ── Image data root (mounted at the same path in all containers) ──────────────
DATA_MOUNT_HOST=/data1
```

> **Note on `DATA_MOUNT_HOST`**: All containers mount this directory at the same host
> path. Image file paths passed to tool calls must reside under this directory.

---

## 5. Building Docker Images

`scripts/build_models.sh` supports two source-acquisition modes:

| Mode | How source code is obtained | Use case |
|------|----------------------------|----------|
| `BUILD_MODE=local` (default) | Copied from the host machine | Internal builds; fast, no internet needed |
| `BUILD_MODE=git` | Cloned from GitHub at build time | Open-source users; no local source needed |

### Build all images at once

```bash
# Open-source users — clone from GitHub
BUILD_MODE=git scripts/build_models.sh

# Internal users — copy from local source (default)
scripts/build_models.sh

# With mirror URLs (when GitHub is slow or blocked)
BUILD_MODE=git \
  SAM2_GIT_URL=https://bgithub.xyz/facebookresearch/sam2.git \
  REMOTESAM_GIT_URL=https://bgithub.xyz/xiaobdul/RemoteSAM.git \
  STRIP_RCNN_GIT_URL=https://bgithub.xyz/wjy5446/Strip-RCNN.git \
  INSTRUCTSAM_GIT_URL=https://bgithub.xyz/VoyagerXvoyagerx/InstructSAM.git \
  scripts/build_models.sh
```

This produces the following images:

| Image | Port |
|-------|------|
| `terrabox/sam2:latest` | 9002 |
| `terrabox/remoteclip:latest` | 9003 |
| `terrabox/remotesam:latest` | 9004 |
| `terrabox/strip-rcnn:latest` | 9005 |
| `terrabox/instructsam:latest` | 9006 |
| `terrabox/vllm:latest` | 9000 |
| `terrabox/agent-llm:latest` | 9100 |

### Build a single image

Example for SAM2 (substitute the directory and ARG name for other models):

```bash
cd docker/sam2

# git mode (default) — no local source needed
docker build -t terrabox/sam2:latest .

# git mode with a mirror URL
docker build --build-arg SAM2_GIT_URL=https://mirror/sam2.git \
             -t terrabox/sam2:latest .

# local mode — copy source into the build context first
cp -r /path/to/sam2 ./sam2
docker build --build-arg BUILD_MODE=local -t terrabox/sam2:latest .
rm -rf ./sam2
```

`ARG` names for each image:

| Image directory | ARG name | Default URL |
|-----------------|----------|-------------|
| `docker/sam2` | `SAM2_GIT_URL` | github.com/facebookresearch/sam2 |
| `docker/remotesam` | `REMOTESAM_GIT_URL` | github.com/xiaobdul/RemoteSAM |
| `docker/strip_rcnn` | `STRIP_RCNN_GIT_URL` | github.com/wjy5446/Strip-RCNN |
| `docker/instructsam` | `INSTRUCTSAM_GIT_URL` | github.com/VoyagerXvoyagerx/InstructSAM |
| `docker/remoteclip`, `docker/vllm`, `docker/agent_llm` | — | no third-party source |

---

## 6. Agent LLM

The Agent LLM drives the reasoning loop. It is configured separately from the perception
models via `agent_config.yaml` in the project root.

> **First-time setup:** copy the example config and fill in your values:
> ```bash
> cp agent_config.example.yaml agent_config.yaml
> ```
> `agent_config.yaml` is gitignored so your local model paths and API keys stay off
> version control.

### Option A: Remote API (simplest)

No GPU required. Works with any OpenAI-compatible API (OpenAI, DeepSeek, Qwen, etc.).

Edit `agent_config.yaml`:

```yaml
use_local_llm: false

remote_llm_api_base: https://api.openai.com/v1   # or any compatible endpoint
remote_llm_api_key: "sk-your-key-here"
remote_llm_model: gpt-4o
```

---

### Option B: Local LLM subprocess

Runs the LLM as a child process using a local conda environment with vLLM installed.

```bash
# Install vLLM in your conda env (if not already installed)
pip install vllm
```

Edit `agent_config.yaml`:

```yaml
use_local_llm: true
use_docker: false

local_llm_model_path: /path/to/your/llm_model/         # model weights directory
local_llm_python_exec: /your/conda/envs/ENV/bin/python  # Python with vllm installed
local_llm_host: 127.0.0.1
local_llm_port: 9100
local_llm_gpu_devices: "0"         # GPU index; comma-separated for multi-GPU
local_llm_tensor_parallel: 1       # must match number of GPUs
local_llm_max_model_len: 24576     # context window size (keep below 24960 for 8B models)
```

---

### Option C: Local LLM Docker container

Uses the pre-built `terrabox/agent-llm:latest` image. Build it first if needed:

```bash
cd docker/agent_llm && docker build -t terrabox/agent-llm:latest .
```

Edit `agent_config.yaml`:

```yaml
use_local_llm: true
use_docker: true

local_llm_model_path: /host/path/to/your/llm_model/  # mounted as /model inside container
local_llm_docker_image: terrabox/agent-llm:latest
local_llm_host: 127.0.0.1
local_llm_port: 9100
local_llm_gpu_devices: "0"
local_llm_tensor_parallel: 1
local_llm_max_model_len: 24576
```

---

## 7. Port Reference

| Service | Port | Notes |
|---------|------|-------|
| Terrabox API | **8000** | Main FastAPI entry point |
| vLLM (vision) | **9000** | Vision-language model |
| SAM2 | **9002** | |
| RemoteCLIP | **9003** | |
| RemoteSAM | **9004** | |
| Strip-RCNN | **9005** | |
| InstructSAM | **9006** | |
| Agent LLM | **9100** | Separate from port 9000 to avoid conflicts |

To change a port, uncomment and set the corresponding `*_PORT` variable in `.env`:

```dotenv
SAM2_PORT=9012
```
