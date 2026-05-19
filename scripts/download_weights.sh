#!/bin/bash

# Download model weights for Terrabox perception models
#
# Each model's weights are downloaded to a local directory that you then reference
# in agent_config.yaml or matching environment variables.
#
# Usage:
#   ./scripts/download_weights.sh                            # download perception weights
#   ./scripts/download_weights.sh --skip-vlm --skip-sam2     # skip specific models
#   HF_ENDPOINT=https://hf-mirror.com ./scripts/download_weights.sh   # China mirror
#
# HuggingFace model IDs (set to your preferred model):
#   VLM_HF_REPO         HuggingFace repo for the vision LLM
#                       (default: Qwen/Qwen3-VL-8B-Instruct)
#
# Destination directories (override via environment variables):
#   VLM_DIR         default: <project_root>/models/vlm
#   SAM2_CKPT_DIR   default: <project_root>/models/sam2_checkpoints
#   REMOTECLIP_DIR  default: <project_root>/models/remoteclip
#   REMOTESAM_DIR   default: <project_root>/models/remotesam
#   STRIP_RCNN_DIR  default: <project_root>/models/strip_rcnn
#   INSTRUCTSAM_DIR default: <project_root>/models/instructsam

set -e

# Color output (matching build_models.sh)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Model IDs ────────────────────────────────────────────────────────────────
VLM_HF_REPO="${VLM_HF_REPO:-Qwen/Qwen3-VL-8B-Instruct}"

# ── Destination directories ───────────────────────────────────────────────────
VLM_DIR="${VLM_DIR:-$PROJECT_ROOT/models/vlm}"
SAM2_CKPT_DIR="${SAM2_CKPT_DIR:-$PROJECT_ROOT/models/sam2_checkpoints}"
REMOTECLIP_DIR="${REMOTECLIP_DIR:-$PROJECT_ROOT/models/remoteclip}"
REMOTESAM_DIR="${REMOTESAM_DIR:-$PROJECT_ROOT/models/remotesam}"
STRIP_RCNN_DIR="${STRIP_RCNN_DIR:-$PROJECT_ROOT/models/strip_rcnn}"
INSTRUCTSAM_DIR="${INSTRUCTSAM_DIR:-$PROJECT_ROOT/models/instructsam}"

# ── HuggingFace endpoint (override for China mirror) ─────────────────────────
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

# ── Parse --skip-X flags ──────────────────────────────────────────────────────
SKIP_VLM=false
SKIP_SAM2=false
SKIP_REMOTECLIP=false
SKIP_REMOTESAM=false
SKIP_STRIP_RCNN=false
SKIP_INSTRUCTSAM=false

for arg in "$@"; do
    case "$arg" in
        --skip-vlm)         SKIP_VLM=true ;;
        --skip-sam2)        SKIP_SAM2=true ;;
        --skip-remoteclip)  SKIP_REMOTECLIP=true ;;
        --skip-remotesam)   SKIP_REMOTESAM=true ;;
        --skip-strip-rcnn)  SKIP_STRIP_RCNN=true ;;
        --skip-instructsam) SKIP_INSTRUCTSAM=true ;;
        --help|-h)
            sed -n '3,20p' "$0"  # print the usage comment block
            exit 0 ;;
        *)
            echo -e "${RED}Unknown flag: $arg${NC}" >&2
            exit 1 ;;
    esac
done

# ── Prerequisite checks ───────────────────────────────────────────────────────
need_wget=false
need_hf=false
need_gdown=false

$SKIP_SAM2        || need_wget=true
$SKIP_REMOTESAM   || need_wget=true
$SKIP_INSTRUCTSAM || need_wget=true
$SKIP_VLM         || need_hf=true
$SKIP_REMOTECLIP  || need_hf=true
$SKIP_INSTRUCTSAM || need_hf=true
$SKIP_STRIP_RCNN  || need_gdown=true

if $need_wget && ! command -v wget &>/dev/null; then
    echo -e "${RED}Error: wget is required but not installed.${NC}" >&2
    exit 1
fi

if $need_hf && ! command -v huggingface-cli &>/dev/null; then
    echo -e "${RED}Error: huggingface-cli is required but not installed.${NC}"
    echo "  pip install huggingface_hub" >&2
    exit 1
fi

HAS_GDOWN=false
if command -v gdown &>/dev/null; then
    HAS_GDOWN=true
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Terrabox — Perception Model Weights Download${NC}"
echo -e "${GREEN}HF_ENDPOINT:      ${HF_ENDPOINT}${NC}"
echo -e "${GREEN}VLM_HF_REPO:       ${VLM_HF_REPO}${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# ── Helper: section banner ────────────────────────────────────────────────────
banner() {
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}$1${NC}"
    echo -e "${YELLOW}========================================${NC}"
}

# ── VLM ──────────────────────────────────────────────────────────────────────
if ! $SKIP_VLM; then
    banner "VLM — $VLM_HF_REPO"
    mkdir -p "$VLM_DIR"
    huggingface-cli download "$VLM_HF_REPO" \
        --local-dir "$VLM_DIR" \
        --resume-download
    echo -e "${GREEN}✓ VLM downloaded → $VLM_DIR${NC}"
    echo ""
fi

# ── SAM2 — sam2.1_hiera_large.pt (092824 release) ───────────────────────────
# This is for the standalone SAM2 service (port 9002). NOT the same file used
# by InstructSAM (which uses the older 072824 sam2_hiera_large.pt).
if ! $SKIP_SAM2; then
    banner "SAM2 — sam2.1_hiera_large.pt"
    mkdir -p "$SAM2_CKPT_DIR"
    wget -c \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" \
        -P "$SAM2_CKPT_DIR"
    echo -e "${GREEN}✓ SAM2 checkpoint downloaded → $SAM2_CKPT_DIR${NC}"
    echo ""
fi

# ── RemoteCLIP — RemoteCLIP-ViT-L-14.pt ──────────────────────────────────────
if ! $SKIP_REMOTECLIP; then
    banner "RemoteCLIP — chendelong/RemoteCLIP"
    mkdir -p "$REMOTECLIP_DIR"
    huggingface-cli download chendelong/RemoteCLIP \
        RemoteCLIP-ViT-L-14.pt \
        --local-dir "$REMOTECLIP_DIR" \
        --resume-download
    echo -e "${GREEN}✓ RemoteCLIP downloaded → $REMOTECLIP_DIR${NC}"
    echo ""
fi

# ── RemoteSAM — Swin-Base backbone (ImageNet-22k, 384px) ─────────────────────
# The RemoteSAM repo expects this exact filename: swin_base_patch4_window12_384_22k.pth
if ! $SKIP_REMOTESAM; then
    banner "RemoteSAM — Swin-Base backbone"
    mkdir -p "$REMOTESAM_DIR"
    wget -c \
        "https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window12_384_22k-d59b0d1d.pth" \
        -O "$REMOTESAM_DIR/swin_base_patch4_window12_384_22k.pth"
    echo -e "${GREEN}✓ RemoteSAM backbone downloaded → $REMOTESAM_DIR${NC}"
    echo ""
fi

# ── Strip-RCNN — stripnet_s.pth (Google Drive) ───────────────────────────────
if ! $SKIP_STRIP_RCNN; then
    banner "Strip-RCNN — stripnet_s.pth"
    mkdir -p "$STRIP_RCNN_DIR"
    if $HAS_GDOWN; then
        # Google Drive file ID for Strip-RCNN-S (without EMA)
        # Source: https://github.com/YXB-NKU/Strip-R-CNN
        gdown 1_c2aXANKHl0cIBb370LNIkCyDmQpA3_o \
            -O "$STRIP_RCNN_DIR/stripnet_s.pth"
        echo -e "${GREEN}✓ Strip-RCNN checkpoint downloaded → $STRIP_RCNN_DIR${NC}"
    else
        echo -e "${YELLOW}⚠ gdown not found — skipping Strip-RCNN.${NC}"
        echo "  Install it with:  pip install gdown"
        echo "  Then download manually:"
        echo "    gdown 1_c2aXANKHl0cIBb370LNIkCyDmQpA3_o -O \"$STRIP_RCNN_DIR/stripnet_s.pth\""
        echo "  Or visit: https://drive.google.com/file/d/1_c2aXANKHl0cIBb370LNIkCyDmQpA3_o"
    fi
    echo ""
fi

# ── InstructSAM — sam2_hiera_large.pt + GeoRSCLIP-ViT-L-14.pt ───────────────
# InstructSAM is pinned to the *original* SAM2 (072824 release), not SAM2.1.
# GeoRSCLIP is loaded via open_clip from a local checkpoint file.
if ! $SKIP_INSTRUCTSAM; then
    banner "InstructSAM — SAM2 (072824) + GeoRSCLIP"
    mkdir -p "$INSTRUCTSAM_DIR"

    echo "Downloading sam2_hiera_large.pt (072824, original SAM2)..."
    wget -c \
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt" \
        -P "$INSTRUCTSAM_DIR"

    echo "Downloading GeoRSCLIP-ViT-L-14.pt..."
    huggingface-cli download Zilun/GeoRSCLIP \
        GeoRSCLIP-ViT-L-14.pt \
        --local-dir "$INSTRUCTSAM_DIR" \
        --resume-download

    echo -e "${GREEN}✓ InstructSAM models downloaded → $INSTRUCTSAM_DIR${NC}"
    echo ""
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Download complete — path reference${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${CYAN}Set these paths in agent_config.yaml or matching environment variables:${NC}"
echo ""

! $SKIP_VLM         && echo "  agent_config.yaml  vlm_model_path:              $VLM_DIR"
! $SKIP_SAM2        && echo "  agent_config.yaml  sam2_checkpoint_host:        $SAM2_CKPT_DIR"
! $SKIP_REMOTECLIP  && echo "  agent_config.yaml  remoteclip_ckpt_host:        $REMOTECLIP_DIR"
! $SKIP_REMOTESAM   && echo "  agent_config.yaml  remotesam_checkpoint_host:   $REMOTESAM_DIR"
! $SKIP_STRIP_RCNN  && echo "  agent_config.yaml  strip_rcnn_ckpt_host:        $STRIP_RCNN_DIR"
! $SKIP_INSTRUCTSAM && echo "  agent_config.yaml  instructsam_models_host:     $INSTRUCTSAM_DIR"
echo ""
echo -e "${CYAN}The Docker service managers also accept uppercase environment variables; see README.md.${NC}"
echo ""
