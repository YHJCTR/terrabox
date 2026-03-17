#!/usr/bin/env python3
"""
InstructSAM 模型下载脚本
========================
将 InstructSAM 所需的全部模型下载到指定目录。

所需模型（来自 InstructSAM GitHub: https://github.com/VoyagerXvoyagerx/InstructSAM）:
  1. SAM2 Hiera Large         ~2.5 GB   facebook/sam2-hiera-large
  2. GeoRSCLIP ViT-L-14       ~0.9 GB   XShadow/GeoRSCLIP
  3. Qwen2.5-VL-7B-Instruct   ~15  GB   Qwen/Qwen2.5-VL-7B-Instruct  [可选]

注：默认架构中计数步骤调用宿主机已有的 vLLM 服务（port 9000），
    无需下载 Qwen。如需独立运行可加 --model qwen 单独下载。

默认保存路径: /data1/yuhongjie2/terra_model/instructsam/

目录结构（下载后）:
  instructsam/
  ├── sam2_hiera_large.pt
  ├── GeoRSCLIP-ViT-L-14.pt
  └── Qwen2.5-VL-7B-Instruct/
      ├── config.json
      ├── model-00001-of-...safetensors
      └── ...

用法:
  python scripts/download_instructsam_models.py
  python scripts/download_instructsam_models.py --dest /your/path
  python scripts/download_instructsam_models.py --skip-qwen   # 跳过大模型
  python scripts/download_instructsam_models.py --model sam2  # 只下载某项
"""

import argparse
import os
import sys
import hashlib
from pathlib import Path

# ── 颜色输出 ──────────────────────────────────────────────────────────────────
def _green(s):  return f"\033[32m{s}\033[0m"
def _yellow(s): return f"\033[33m{s}\033[0m"
def _red(s):    return f"\033[31m{s}\033[0m"


def _require(pkg: str):
    """检查 Python 包是否可用，给出友好提示。"""
    import importlib
    if importlib.util.find_spec(pkg) is None:
        print(_red(f"[ERROR] Required package '{pkg}' not found."))
        print(f"        Install with: pip install {pkg}")
        sys.exit(1)


# ── 单文件 wget 下载（带进度条）────────────────────────────────────────────────
def _download_file(url: str, dest: Path, desc: str = ""):
    """使用 requests 流式下载单个文件，显示进度。"""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(_yellow(f"  [SKIP] {dest.name} already exists."))
        return

    print(f"  Downloading {desc or dest.name}  ←  {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done  = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    print(f"\r    {pct:5.1f}%  ({done >> 20} / {total >> 20} MB)", end="", flush=True)
    print(f"\r    {_green('Done')}  →  {dest}")


# ── 1. SAM2 Hiera Large ────────────────────────────────────────────────────────
def download_sam2(dest_dir: Path):
    print("\n[1/3] SAM2 Hiera Large (~2.5 GB)")
    # 官方下载地址来自 facebook/sam2 仓库的 checkpoints/download_ckpts.sh
    url  = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    dest = dest_dir / "sam2_hiera_large.pt"
    _download_file(url, dest, "sam2_hiera_large.pt")


# ── 2. GeoRSCLIP ViT-L-14 ─────────────────────────────────────────────────────
def download_georsclip(dest_dir: Path):
    print("\n[2/3] GeoRSCLIP ViT-L-14 (~0.9 GB)")
    _require("huggingface_hub")
    from huggingface_hub import hf_hub_download

    dest = dest_dir / "GeoRSCLIP-ViT-L-14.pt"
    if dest.exists():
        print(_yellow(f"  [SKIP] {dest.name} already exists."))
        return

    print("  Downloading from HuggingFace: XShadow/GeoRSCLIP ...")
    # GeoRSCLIP 在 HuggingFace 上的文件名
    downloaded = hf_hub_download(
        repo_id="XShadow/GeoRSCLIP",
        filename="GeoRSCLIP-ViT-L-14.pt",
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
    )
    # hf_hub_download 会放到子目录，重命名到标准位置
    downloaded_path = Path(downloaded)
    if downloaded_path != dest:
        downloaded_path.rename(dest)
    print(f"  {_green('Done')}  →  {dest}")


# ── 3. Qwen2.5-VL-7B-Instruct ─────────────────────────────────────────────────
def download_qwen(dest_dir: Path):
    print("\n[3/3] Qwen2.5-VL-7B-Instruct (~15 GB)")
    _require("huggingface_hub")
    from huggingface_hub import snapshot_download

    qwen_dir = dest_dir / "Qwen2.5-VL-7B-Instruct"
    if qwen_dir.exists() and any(qwen_dir.iterdir()):
        print(_yellow(f"  [SKIP] {qwen_dir} already exists and is non-empty."))
        return

    print("  Downloading from HuggingFace: Qwen/Qwen2.5-VL-7B-Instruct ...")
    print("  (This may take 20-40 minutes depending on your network speed)")
    snapshot_download(
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        local_dir=str(qwen_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    print(f"  {_green('Done')}  →  {qwen_dir}")


# ── 验证 ──────────────────────────────────────────────────────────────────────
def verify(dest_dir: Path, skip_qwen: bool):
    print("\n── Verification ──────────────────────────────────────────────────────")
    checks = [
        (dest_dir / "sam2_hiera_large.pt",      "SAM2 Hiera Large"),
        (dest_dir / "GeoRSCLIP-ViT-L-14.pt",   "GeoRSCLIP ViT-L-14"),
    ]
    if not skip_qwen:
        checks.append((dest_dir / "Qwen2.5-VL-7B-Instruct" / "config.json",
                        "Qwen2.5-VL-7B-Instruct (config.json)"))

    all_ok = True
    for path, name in checks:
        if path.exists():
            size_mb = path.stat().st_size >> 20
            print(f"  {_green('OK')}  {name}  ({size_mb} MB)  →  {path}")
        else:
            print(f"  {_red('MISSING')}  {name}  →  {path}")
            all_ok = False

    if all_ok:
        print(_green("\nAll models downloaded successfully!"))
    else:
        print(_red("\nSome models are missing. Please re-run the script."))
        sys.exit(1)


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download InstructSAM required models."
    )
    parser.add_argument(
        "--dest",
        default="/data1/yuhongjie2/terra_model/instructsam",
        help="Destination directory for model files.",
    )
    parser.add_argument(
        "--skip-qwen",
        action="store_true",
        help="Skip downloading Qwen2.5-VL-7B-Instruct (~15 GB).",
    )
    parser.add_argument(
        "--model",
        choices=["sam2", "georsclip", "qwen", "all"],
        default="all",
        help="Download only a specific model (default: all).",
    )
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Destination: {dest_dir}")

    model = args.model

    if model in ("sam2", "all"):
        download_sam2(dest_dir)

    if model in ("georsclip", "all"):
        download_georsclip(dest_dir)

    if model in ("qwen", "all") and not args.skip_qwen:
        download_qwen(dest_dir)
    elif args.skip_qwen:
        print(_yellow("\n[3/3] Qwen2.5-VL-7B-Instruct — SKIPPED (--skip-qwen)"))

    verify(dest_dir, skip_qwen=(args.skip_qwen or model not in ("qwen", "all")))


if __name__ == "__main__":
    main()
