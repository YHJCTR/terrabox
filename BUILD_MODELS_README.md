# Perception Model Build Script

This script builds all perception model Docker images using your local CUDA base image.

## Prerequisites

- Docker installed
- Base image `nvidia/cuda:11.8.0-devel-ubuntu22.04` available locally
- Sufficient disk space for model images

## Usage

Run the build script:

```bash
cd /data1/yuhongjie2/terrabox
./build_models.sh
```

## Built Images

The script will build the following Docker images:

### 1. SAM2 (Segment Anything Model 2)
- **Image**: `terrabox/sam2:latest`
- **Port**: 9002
- **Base**: `nvidia/cuda:11.8.0-devel-ubuntu22.04`
- **Features**: Image segmentation with full box prompt
- **Dependencies**: PyTorch, SAM2 from source, FastAPI, OpenCV

### 2. RemoteCLIP
- **Image**: `terrabox/remoteclip:latest`
- **Port**: 9003
- **Base**: `nvidia/cuda:11.8.0-devel-ubuntu22.04`
- **Features**: Zero-shot classification and retrieval for satellite images
- **Dependencies**: PyTorch, open_clip_torch, Flask

### 3. RemoteSAM
- **Image**: `terrabox/remotesam:latest`
- **Port**: 9004
- **Base**: `nvidia/cuda:11.8.0-devel-ubuntu22.04`
- **Features**: Visual grounding, referring segmentation, detection
- **Dependencies**: PyTorch 1.13.1+cu117, transformers, timm

### 4. Strip-RCNN
- **Image**: `terrabox/strip_rcnn:latest`
- **Port**: 9005
- **Base**: `nvidia/cuda:11.8.0-devel-ubuntu22.04`
- **Features**: Object detection for elongated objects (ships, roads, bridges)
- **Dependencies**: PyTorch 1.13.1+cu117, mmcv-full, mmdet, mmrotate

## Model Checkpoints

Each model requires specific checkpoints to be mounted at runtime:

### SAM2
```bash
-v /your/sam2/checkpoints:/checkpoints:ro
-v /your/sam2/configs:/sam2_configs:ro
```

### RemoteCLIP
```bash
-v /your/RemoteCLIP/checkpoints:/checkpoints:ro
```

### RemoteSAM
```bash
-v /your/RemoteSAM:/app:ro
```

### Strip-RCNN
```bash
-v /your/Strip-RCNN/configs:/configs:ro
-v /your/Strip-RCNN/ckpt:/ckpt:ro
```

## Troubleshooting

### Base Image Not Found
If you get an error about the base image not being found, ensure you have:
```bash
docker pull nvidia/cuda:11.8.0-devel-ubuntu22.04
```

### Build Failures
- Check Docker logs for specific error messages
- Ensure you have sufficient disk space
- Verify network connectivity for package downloads

## Notes

- All models use the same CUDA base image for consistency
- Each model runs in an isolated container for better resource management
- The script includes error handling and will stop if any build fails
- Color-coded output helps track build progress