from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    chunk_index: int
    content: str
    metadata: dict = field(default_factory=dict)


def _recursive_splitter(chunk_size: int, chunk_overlap: int):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )


class DocumentProcessor:
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".txt", ".md", ".html", ".htm", ".csv"}
    # Extensions that carry structural markers (headers, sections) worth preserving
    _STRUCTURED_EXTENSIONS = {".md", ".html", ".htm"}

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        if ext in {".txt", ".md", ".csv"}:
            return Path(file_path).read_text(encoding="utf-8", errors="ignore")
        try:
            from unstructured.partition.auto import partition
            elements = partition(filename=file_path)
            return "\n\n".join(str(el) for el in elements)
        except ImportError:
            logger.warning("unstructured not installed, falling back to plain text for %s", file_path)
            return Path(file_path).read_text(encoding="utf-8", errors="ignore")

    def chunk(self, text: str, metadata: dict | None = None) -> List[ChunkResult]:
        """Default chunker: RecursiveCharacterTextSplitter (good for unstructured prose)."""
        texts = _recursive_splitter(self.chunk_size, self.chunk_overlap).split_text(text)
        return [ChunkResult(i, t, metadata or {}) for i, t in enumerate(texts)]

    def _structure_chunk(self, text: str, metadata: dict | None = None) -> List[ChunkResult]:
        """Structure-aware chunker for Markdown / HTML.

        Splits on section headers (# / ##) and double newlines first to keep
        semantically cohesive blocks together, then applies the size splitter
        on any section that still exceeds chunk_size.
        """
        # Split on markdown headers or paragraph breaks
        sections = re.split(r"(?=\n#{1,3}\s)|\n{2,}", text)
        splitter = _recursive_splitter(self.chunk_size, self.chunk_overlap)
        chunks: List[str] = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                chunks.extend(splitter.split_text(section))
        return [ChunkResult(i, c, metadata or {}) for i, c in enumerate(chunks) if c.strip()]

    def process(self, file_path: str, metadata: dict | None = None) -> List[ChunkResult]:
        text = self.parse(file_path)
        if not text.strip():
            logger.warning("Empty content from %s", file_path)
            return []
        ext = Path(file_path).suffix.lower()
        meta = {"filename": Path(file_path).name, **(metadata or {})}
        if ext in self._STRUCTURED_EXTENSIONS:
            return self._structure_chunk(text, meta)
        return self.chunk(text, meta)

    @classmethod
    def is_supported(cls, filename: str) -> bool:
        return Path(filename).suffix.lower() in cls.SUPPORTED_EXTENSIONS
