from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import models
from ..db.session import get_db
from ..routers.deps import current_user_from_jwt
from .document_processor import DocumentProcessor
from .embedding import EmbeddingService
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/gui/knowledge-bases", tags=["knowledge-bases"])

_UPLOAD_DIR = Path("./terrabox_uploads/kb_docs")


class KBCreateRequest(BaseModel):
    name: str
    description: str = ""


class KBResponse(BaseModel):
    id: str
    name: str
    description: str
    document_count: int = 0
    created_at: str = ""


class DocResponse(BaseModel):
    id: str
    filename: str
    status: str
    chunk_count: int = 0
    created_at: str = ""


def _get_user_kbs(db: Session, user_id: str) -> List[models.KnowledgeBase]:
    return db.query(models.KnowledgeBase).filter_by(owner_id=user_id).all()


@router.post("", response_model=KBResponse)
def create_kb(
    req: KBCreateRequest,
    current_user: models.User = Depends(current_user_from_jwt),
    db: Session = Depends(get_db),
):
    kb = models.KnowledgeBase(
        name=req.name,
        description=req.description,
        owner_id=current_user.id,
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return KBResponse(
        id=kb.id, name=kb.name, description=kb.description,
        document_count=0, created_at=str(kb.created_at),
    )


@router.get("", response_model=List[KBResponse])
def list_kbs(
    current_user: models.User = Depends(current_user_from_jwt),
    db: Session = Depends(get_db),
):
    kbs = _get_user_kbs(db, current_user.id)
    return [
        KBResponse(
            id=kb.id, name=kb.name, description=kb.description,
            document_count=len(kb.documents), created_at=str(kb.created_at),
        )
        for kb in kbs
    ]


@router.delete("/{kb_id}")
def delete_kb(
    kb_id: str,
    current_user: models.User = Depends(current_user_from_jwt),
    db: Session = Depends(get_db),
):
    kb = db.query(models.KnowledgeBase).filter_by(id=kb_id, owner_id=current_user.id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    try:
        vs = VectorStore()
        vs.drop_collection(kb_id)
    except Exception as e:
        logger.warning("Failed to drop vector collection: %s", e)
    db.delete(kb)
    db.commit()
    return {"status": "ok"}


@router.post("/{kb_id}/docs", response_model=DocResponse)
def upload_doc(
    kb_id: str,
    files: List[UploadFile] = File(...),
    current_user: models.User = Depends(current_user_from_jwt),
    db: Session = Depends(get_db),
):
    kb = db.query(models.KnowledgeBase).filter_by(id=kb_id, owner_id=current_user.id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    processor = DocumentProcessor()
    embedding_svc = EmbeddingService()
    vs = VectorStore()

    last_doc = None
    for f in files:
        if not DocumentProcessor.is_supported(f.filename):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {f.filename}")

        dest = _UPLOAD_DIR / f"{uuid.uuid4().hex}_{f.filename}"
        dest.write_bytes(f.file.read())
        file_path = str(dest)

        doc = models.Document(
            kb_id=kb_id,
            filename=f.filename,
            file_path=file_path,
            status="processing",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        try:
            chunks = processor.process(file_path, metadata={"kb_id": kb_id, "doc_id": doc.id})
            if not chunks:
                doc.status = "empty"
                db.commit()
                continue

            texts = [c.content for c in chunks]
            vectors = embedding_svc.embed(texts)

            ids = [str(uuid.uuid4()) for _ in chunks]
            doc_ids = [doc.id] * len(chunks)
            metadatas = [json.dumps(c.metadata, ensure_ascii=False) for c in chunks]

            vs.insert(
                kb_id=kb_id,
                ids=ids,
                vectors=vectors,
                contents=texts,
                doc_ids=doc_ids,
                metadatas=metadatas,
            )

            for chunk, cid in zip(chunks, ids):
                db_chunk = models.DocumentChunk(
                    doc_id=doc.id,
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    embedding_id=cid,
                    metadata_json=json.dumps(chunk.metadata, ensure_ascii=False),
                )
                db.add(db_chunk)

            doc.status = "ready"
            doc.chunk_count = len(chunks)
            db.commit()
            db.refresh(doc)
            last_doc = doc

        except Exception as e:
            logger.error("Failed to process document %s: %s", f.filename, e)
            doc.status = "error"
            db.commit()
            raise HTTPException(status_code=500, detail=f"Document processing failed: {e}")

    if last_doc is None:
        raise HTTPException(status_code=400, detail="No valid documents uploaded")
    return DocResponse(
        id=last_doc.id, filename=last_doc.filename,
        status=last_doc.status, chunk_count=last_doc.chunk_count,
        created_at=str(last_doc.created_at),
    )


@router.get("/{kb_id}/docs", response_model=List[DocResponse])
def list_docs(
    kb_id: str,
    current_user: models.User = Depends(current_user_from_jwt),
    db: Session = Depends(get_db),
):
    kb = db.query(models.KnowledgeBase).filter_by(id=kb_id, owner_id=current_user.id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return [
        DocResponse(
            id=d.id, filename=d.filename, status=d.status,
            chunk_count=d.chunk_count, created_at=str(d.created_at),
        )
        for d in kb.documents
    ]


@router.delete("/{kb_id}/docs/{doc_id}")
def delete_doc(
    kb_id: str,
    doc_id: str,
    current_user: models.User = Depends(current_user_from_jwt),
    db: Session = Depends(get_db),
):
    kb = db.query(models.KnowledgeBase).filter_by(id=kb_id, owner_id=current_user.id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    doc = db.query(models.Document).filter_by(id=doc_id, kb_id=kb_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        vs = VectorStore()
        vs.delete_by_doc(kb_id, doc_id)
    except Exception as e:
        logger.warning("Failed to delete vectors: %s", e)
    db.delete(doc)
    db.commit()
    return {"status": "ok"}
