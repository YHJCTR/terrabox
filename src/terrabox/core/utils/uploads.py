"""Shared file-upload utilities."""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import List

from fastapi import UploadFile

UPLOAD_DIR = Path(os.getenv("TERRABOX_UPLOAD_DIR", "/data1/terrabox_uploads"))


async def save_upload_files(files: List[UploadFile]) -> List[str]:
    """Save a list of UploadFile objects to UPLOAD_DIR; return their absolute paths."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    for f in files:
        suffix = Path(f.filename or "upload").suffix or ".bin"
        dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
        dest.write_bytes(await f.read())
        saved.append(str(dest))
    return saved
