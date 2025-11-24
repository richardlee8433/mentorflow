"""
RAG (Retrieval-Augmented Generation) helper utilities.

This module is responsible for:

1. Loading and persisting a very small local "vector store"
   under backend/uploaded_docs/vector_store.json
2. Ingesting TXT / PDF files into that store
3. Serving higher-level helpers that the FastAPI app can call:
   - build_kb_from_txt_file(...)
   - build_kb_from_pdf_file(...)
   - retrieve_relevant_chunks(...)

設計目標：
- 盡量簡單、無外部 DB，相容 Render / Netlify demo
- 若沒有任何文件上傳，retrieve_relevant_chunks() 會回傳 []
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from fastapi import UploadFile, HTTPException


# ==========================
# 路徑 & 基本設定
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploaded_docs"
VECTOR_STORE_PATH = UPLOAD_DIR / "vector_store.json"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ==========================
# 資料結構
# ==========================


@dataclass
class StoredChunk:
    """Single chunk stored in the local vector store."""

    id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]


# in-memory vector store (loaded at import time)
_VECTOR_STORE: List[StoredChunk] = []


# ==========================
# 基礎工具
# ==========================


def _load_vector_store() -> None:
    """Load vector store from disk into memory."""
    global _VECTOR_STORE

    if not VECTOR_STORE_PATH.exists():
        _VECTOR_STORE = []
        return

    try:
        raw = json.loads(VECTOR_STORE_PATH.read_text(encoding="utf-8"))
        _VECTOR_STORE = [
            StoredChunk(
                id=item["id"],
                text=item["text"],
                embedding=item["embedding"],
                metadata=item.get("metadata", {}),
            )
            for item in raw
        ]
    except Exception:
        # 若檔案壞掉，直接重建（demo 用，容錯較寬鬆）
        _VECTOR_STORE = []


def _save_vector_store() -> None:
    """Persist current in-memory vector store to disk."""
    VECTOR_STORE_PATH.write_text(
        json.dumps([asdict(c) for c in _VECTOR_STORE], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# module import 時載入一次
_load_vector_store()


def embed_text(text: str) -> List[float]:
    """
    Call OpenAI embeddings API and return a vector.

    抽成獨立函式，方便未來更換模型或改成 self-hosted embedding。
    """
    text = text.strip()
    if not text:
        return []

    resp = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    # new OpenAI client: resp.data[0].embedding
    return resp.data[0].embedding  # type: ignore[no-any-return]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _simple_text_splitter(
    text: str, max_chars: int = 1200, overlap: int = 200
) -> List[str]:
    """
    Very small, character-based splitter.

    對 demo 來說已經足夠，不額外引入 tiktoken。
    """
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = end - overlap  # 重疊一點，避免斷句太硬

    return chunks


# ==========================
# 搜尋 API (給 app.py 用)
# ==========================


def search_chunks(
    query_embedding: List[float],
    top_k: int = 3,
    score_threshold: Optional[float] = None,
) -> List[Tuple[StoredChunk, float]]:
    """
    在本地 vector store 裡做簡單 cosine similarity 搜尋。

    回傳 [(StoredChunk, score), ...]，已依 score 由高到低排序。
    """
    if not query_embedding or not _VECTOR_STORE:
        return []

    scored: List[Tuple[StoredChunk, float]] = []
    for item in _VECTOR_STORE:
        score = _cosine_similarity(query_embedding, item.embedding)
        if score_threshold is None or score >= score_threshold:
            scored.append((item, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def retrieve_relevant_chunks(
    question: str,
    top_k: int = 3,
    score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve Top-K relevant chunks for a given user question.

    This function:
    1) Embeds the question
    2) Searches the vector store
    3) Returns a list of dicts containing text, score, and metadata,
       ready to be used by the RAG prompt builder.

    Args:
        question:
            User's natural language question.
        top_k:
            Maximum number of chunks to retrieve.
        score_threshold:
            Optional minimum cosine similarity.

    Returns:
        A list of dicts, each containing:
        - "text": str
        - "score": float
        - "metadata": Dict[str, Any]
    """
    question = (question or "").strip()
    if not question:
        return []

    query_embedding = embed_text(question)
    results = search_chunks(
        query_embedding=query_embedding,
        top_k=top_k,
        score_threshold=score_threshold,
    )

    formatted: List[Dict[str, Any]] = []
    for item, score in results:
        formatted.append(
            {
                "text": item.text,
                "score": float(score),
                "metadata": dict(item.metadata),
            }
        )

    return formatted


# ==========================
# 上傳 / 建立知識庫
# ==========================


def _ingest_text(
    text: str,
    *,
    source_file: Path,
    doc_type: str,
) -> Dict[str, Any]:
    """
    Core ingestion logic shared by TXT & PDF.

    會：
    1. 切 chunk
    2. 為每個 chunk 建 embedding
    3. 寫入 in-memory store 並存檔到 JSON
    """
    global _VECTOR_STORE

    chunks = _simple_text_splitter(text)
    if not chunks:
        return {
            "status": "no_content",
            "chunks_added": 0,
            "doc_id": source_file.name,
        }

    new_items: List[StoredChunk] = []

    for idx, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        metadata = {
            "doc_id": source_file.name,
            "chunk_index": idx,
            "doc_type": doc_type,
            "relative_path": str(source_file.name),
        }
        new_items.append(
            StoredChunk(
                id=f"{source_file.name}__{idx}",
                text=chunk,
                embedding=embedding,
                metadata=metadata,
            )
        )

    _VECTOR_STORE.extend(new_items)
    _save_vector_store()

    return {
        "status": "ok",
        "chunks_added": len(new_items),
        "doc_id": source_file.name,
    }


def build_kb_from_txt_file(file_path: Path) -> Dict[str, Any]:
    """
    Ingest a plain text file into the vector store.

    Called by /admin/upload endpoint when content_type starts with text/plain.
    """
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    return _ingest_text(text, source_file=file_path, doc_type="txt")


def build_kb_from_pdf_file(file_path: Path) -> Dict[str, Any]:
    """
    Ingest a PDF file into the vector store.

    For simplicity we use PyPDF2 (if available). If import fails,
    this will raise an ImportError and FastAPI 會回傳 500。
    """
    try:
        import PyPDF2  # type: ignore[import]

        reader = PyPDF2.PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages)
    except Exception:
        # 萬一 PDF 解析失敗，就當成沒有內容
        text = ""

    return _ingest_text(text, source_file=file_path, doc_type="pdf")

async def handle_admin_upload(file: UploadFile) -> Dict[str, Any]:
    """
    FastAPI 專用的上傳處理入口。

    app.py 會呼叫這個函式：
        result = await handle_admin_upload(file)

    這裡負責：
    1) 檢查副檔名
    2) 把上傳檔案存到 UPLOAD_DIR
    3) 依檔案類型呼叫 build_kb_from_txt_file 或 build_kb_from_pdf_file
    4) 回傳簡單的 JSON 給前端
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a name.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".txt", ".pdf"}:
        raise HTTPException(
            status_code=400,
            detail="Only .txt and .pdf files are supported.",
        )

    # 確保上傳目錄存在
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # 儲存原始檔案
    target_path = UPLOAD_DIR / Path(file.filename).name
    contents = await file.read()
    target_path.write_bytes(contents)

    # 依副檔名進行向量化與寫入向量庫
    if suffix == ".txt":
        ingest_result = build_kb_from_txt_file(target_path)
    else:
        ingest_result = build_kb_from_pdf_file(target_path)

    return {
        "status": "ok",
        "filename": target_path.name,
        "chunks_added": ingest_result.get("chunks_added"),
        "doc_id": ingest_result.get("doc_id"),
    }
