# Perception Models — Installation Guide

For main Terrabox API setup (Python env, database, startup) see [README.md](README.md).

---

## 1. Services

AI services start on demand — no manual launch needed.

| Service | Port | Mode |
|---------|------|------|
| vLLM (vision) | 9000 | subprocess or Docker |
| SAM2 | 9002 | subprocess or Docker |
| RemoteCLIP | 9003 | subprocess or Docker |
| RemoteSAM | 9004 | subprocess or Docker |
| Strip-RCNN | 9005 | subprocess or Docker |
| InstructSAM | 9006 | subprocess or Docker |

Docker mode is the default for this branch. Set `TERRABOX_USE_DOCKER=true` in `.env` unless you are intentionally wiring custom subprocess managers.

---

## 2. Model Weights

```bash
pip install huggingface_hub gdown   # prerequisites
scripts/download_weights.sh         # downloads all models to ./models/
```

Common options:

```bash
scripts/download_weights.sh --skip-vlm          # skip specific models
HF_ENDPOINT=https://hf-mirror.com scripts/download_weights.sh    # mirror
```

Skip flags: `--skip-vlm` `--skip-sam2` `--skip-remoteclip` `--skip-remotesam` `--skip-strip-rcnn` `--skip-instructsam`

### Weights reference

| Service | File(s) | Default dir | `agent_config.yaml` key |
|---------|---------|-------------|--------------------------|
| VLM | `Qwen/Qwen3-VL-8B-Instruct` | `models/vlm/` | `vlm_model_path` |
| SAM2 | `sam2.1_hiera_large.pt` | `models/sam2_checkpoints/` | `sam2_checkpoint_host` (Docker) / `sam2_work_dir` (subprocess) |
| RemoteCLIP | `RemoteCLIP-ViT-L-14.pt` | `models/remoteclip/` | `remoteclip_ckpt_host` |
| RemoteSAM | `swin_base_patch4_window12_384_22k.pth` | `models/remotesam/` | `remotesam_checkpoint_host` |
| Strip-RCNN | `stripnet_s.pth` | `models/strip_rcnn/` | `strip_rcnn_ckpt_host` |
| InstructSAM | `sam2_hiera_large.pt` + `GeoRSCLIP-ViT-L-14.pt` | `models/instructsam/` | `instructsam_models_host` |

> InstructSAM uses `sam2_hiera_large.pt` (July 2024), distinct from SAM2's `sam2.1_hiera_large.pt` (Sep 2024).

---


## 3. Docker Mode

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

## 4. Building Docker Images

```bash
scripts/build_models.sh                   # clone source from GitHub (default)
BUILD_MODE=local scripts/build_models.sh  # copy from local source

# Mirror URLs when GitHub is slow
BUILD_MODE=git \
  SAM2_GIT_URL=https://bgithub.xyz/facebookresearch/sam2.git \
  REMOTESAM_GIT_URL=https://bgithub.xyz/xiaobdul/RemoteSAM.git \
  STRIP_RCNN_GIT_URL=https://bgithub.xyz/wjy5446/Strip-RCNN.git \
  INSTRUCTSAM_GIT_URL=https://bgithub.xyz/VoyagerXvoyagerx/InstructSAM.git \
  scripts/build_models.sh
```

Images: `terrabox/vllm:latest` `terrabox/sam2:latest` `terrabox/remoteclip:latest` `terrabox/remotesam:latest` `terrabox/strip-rcnn:latest` `terrabox/instructsam:latest`

Single image build example:

```bash
cd docker/sam2 && docker build -t terrabox/sam2:latest .
# With mirror: docker build --build-arg SAM2_GIT_URL=https://mirror/sam2.git -t terrabox/sam2:latest .
```

---


## 5. Port Reference

| Service | Port |
|---------|------|
| Terrabox API | 8000 |
| vLLM (vision) | 9000 |
| SAM2 | 9002 |
| RemoteCLIP | 9003 |
| RemoteSAM | 9004 |
| Strip-RCNN | 9005 |
| InstructSAM | 9006 |

Override any port with `sam2_port: 9012` in `agent_config.yaml` (or `SAM2_PORT=9012` in `.env`).
