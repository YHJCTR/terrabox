# Perception Models & Agent — Installation Guide

For main Terrabox API setup (Python env, database, startup) see [README.md](README.md).

---

## 1. Services

Seven AI services start on demand — no manual launch needed.

| Service | Port | Mode |
|---------|------|------|
| vLLM (vision) | 9000 | subprocess or Docker |
| SAM2 | 9002 | subprocess or Docker |
| RemoteCLIP | 9003 | subprocess or Docker |
| RemoteSAM | 9004 | subprocess or Docker |
| Strip-RCNN | 9005 | subprocess or Docker |
| InstructSAM | 9006 | subprocess or Docker |
| Agent LLM | 9100 | subprocess or Docker |

Deployment mode: set `TERRABOX_USE_DOCKER=false` (default) or `true` in `.env`.

---

## 2. Model Weights

```bash
pip install huggingface_hub gdown   # prerequisites
scripts/download_weights.sh         # downloads all models to ./models/
```

Common options:

```bash
scripts/download_weights.sh --skip-agent --skip-vlm          # skip specific models
AGENT_LLM_HF_REPO=your-org/your-llm scripts/download_weights.sh  # custom model
HF_ENDPOINT=https://hf-mirror.com scripts/download_weights.sh    # mirror
```

Skip flags: `--skip-agent` `--skip-vlm` `--skip-sam2` `--skip-remoteclip` `--skip-remotesam` `--skip-strip-rcnn` `--skip-instructsam`

### Weights reference

| Service | File(s) | Default dir | `agent_config.yaml` key |
|---------|---------|-------------|--------------------------|
| Agent LLM | `Qwen3-8B/` | `models/agent_llm/` | `local_llm_model_path` |
| VLM | `NightPro/qwen3vl-8b-4bit-disasterM3/` | `models/vlm/` | `vlm_model_path` |
| SAM2 | `sam2.1_hiera_large.pt` | `models/sam2_checkpoints/` | `sam2_checkpoint_host` (Docker) / `sam2_work_dir` (subprocess) |
| RemoteCLIP | `RemoteCLIP-ViT-L-14.pt` | `models/remoteclip/` | `remoteclip_ckpt_host` |
| RemoteSAM | `swin_base_patch4_window12_384_22k.pth` | `models/remotesam/` | `remotesam_checkpoint_host` |
| Strip-RCNN | `stripnet_s.pth` | `models/strip_rcnn/` | `strip_rcnn_ckpt_host` |
| InstructSAM | `sam2_hiera_large.pt` + `GeoRSCLIP-ViT-L-14.pt` | `models/instructsam/` | `instructsam_models_host` |

> InstructSAM uses `sam2_hiera_large.pt` (July 2024), distinct from SAM2's `sam2.1_hiera_large.pt` (Sep 2024).

---

## 3. Non-Docker Mode (subprocess)

Requires a separate conda environment for each model. See each model's upstream README for env setup.

Add paths to `agent_config.yaml` (gitignored, recommended) or `.env` (fallback). See `agent_config.example.yaml` for all available keys.

**`agent_config.yaml` — key paths:**

```yaml
use_docker: false

vlm_model_path: /path/to/vlm_model/
vlm_python_exec: /path/to/conda/envs/YOUR_ENV/bin/python
vlm_gpu_devices: "0"

sam2_python_exec: /path/to/conda/envs/sam2/bin/python
sam2_server_script: /path/to/sam2/sam2_server2.py
sam2_work_dir: /path/to/sam2

remoteclip_python_exec: /path/to/conda/envs/RemoteCLIP/bin/python
remoteclip_server_script: /path/to/RemoteCLIP/start2.py
remoteclip_work_dir: /path/to/RemoteCLIP

remotesam_python_exec: /path/to/conda/envs/RemoteSAM/bin/python
remotesam_server_script: /path/to/RemoteSAM/start.py
remotesam_work_dir: /path/to/RemoteSAM

strip_rcnn_python_exec: /path/to/conda/envs/strip/bin/python
strip_rcnn_server_script: /path/to/Strip-RCNN/start.py
strip_rcnn_work_dir: /path/to/Strip-RCNN
strip_rcnn_config_path: /path/to/Strip-RCNN/configs/strip_rcnn/orig/strip_rcnn_s_fpn_1x_dota_le90.py
strip_rcnn_checkpoint_path: /path/to/Strip-RCNN/ckpt/stripnet_s.pth
```

Ports default to 9000/9002–9006; override with `vlm_port`, `sam2_port`, etc.

---

## 4. Docker Mode

Requires Docker + NVIDIA Container Toolkit. Build images first (see §5).

Add volume paths to `agent_config.yaml`:

```yaml
use_docker: true

vlm_model_path: /host/path/to/vlm_model/
vlm_gpu_devices: "0"

sam2_checkpoint_host: /host/path/to/sam2/checkpoints
sam2_config_host: /host/path/to/sam2/configs

remoteclip_ckpt_host: /host/path/to/RemoteCLIP/checkpoints

remotesam_checkpoint_host: /host/path/to/RemoteSAM/pretrained_weights

strip_rcnn_ckpt_host: /host/path/to/Strip-RCNN/ckpt
strip_rcnn_config_host: /host/path/to/Strip-RCNN/configs

instructsam_models_host: /host/path/to/instructsam/models  # needs sam2_hiera_large.pt + GeoRSCLIP-ViT-L-14.pt

docker_data_mount_host: /data1  # mounted at the same path in all containers
```

---

## 5. Building Docker Images

```bash
BUILD_MODE=git scripts/build_models.sh    # clone source from GitHub (open-source)
scripts/build_models.sh                   # copy from local source (default)

# Mirror URLs when GitHub is slow
BUILD_MODE=git \
  SAM2_GIT_URL=https://bgithub.xyz/facebookresearch/sam2.git \
  REMOTESAM_GIT_URL=https://bgithub.xyz/xiaobdul/RemoteSAM.git \
  STRIP_RCNN_GIT_URL=https://bgithub.xyz/wjy5446/Strip-RCNN.git \
  INSTRUCTSAM_GIT_URL=https://bgithub.xyz/VoyagerXvoyagerx/InstructSAM.git \
  scripts/build_models.sh
```

Images: `terrabox/vllm:latest` `terrabox/sam2:latest` `terrabox/remoteclip:latest` `terrabox/remotesam:latest` `terrabox/strip-rcnn:latest` `terrabox/instructsam:latest` `terrabox/agent-llm:latest`

Single image build example:

```bash
cd docker/sam2 && docker build -t terrabox/sam2:latest .
# With mirror: docker build --build-arg SAM2_GIT_URL=https://mirror/sam2.git -t terrabox/sam2:latest .
```

---

## 6. Agent LLM

First-time setup:

```bash
cp agent_config.example.yaml agent_config.yaml
```

**Option A — Remote API** (no GPU needed):

```yaml
use_local_llm: false
remote_llm_api_base: https://api.openai.com/v1
remote_llm_api_key: "sk-your-key"
remote_llm_model: gpt-4o
```

**Option B — Local subprocess** (`pip install vllm` in your conda env):

```yaml
use_local_llm: true
use_docker: false
local_llm_model_path: /path/to/llm_model/
local_llm_python_exec: /your/conda/envs/ENV/bin/python
local_llm_gpu_devices: "0"
local_llm_tensor_parallel: 1
local_llm_max_model_len: 24576
```

**Option C — Local Docker container** (`cd docker/agent_llm && docker build -t terrabox/agent-llm:latest .`):

```yaml
use_local_llm: true
use_docker: true
local_llm_model_path: /host/path/to/llm_model/
local_llm_docker_image: terrabox/agent-llm:latest
local_llm_gpu_devices: "0"
local_llm_tensor_parallel: 1
local_llm_max_model_len: 24576
```

---

## 7. Port Reference

| Service | Port |
|---------|------|
| Terrabox API | 8000 |
| vLLM (vision) | 9000 |
| SAM2 | 9002 |
| RemoteCLIP | 9003 |
| RemoteSAM | 9004 |
| Strip-RCNN | 9005 |
| InstructSAM | 9006 |
| Agent LLM | 9100 |

Override any port with `sam2_port: 9012` in `agent_config.yaml` (or `SAM2_PORT=9012` in `.env`).
