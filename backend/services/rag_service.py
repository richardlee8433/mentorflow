"""
RAG helpers for MentorFlow v0.7 (MVP).

Responsibilities:
- Chunk raw text into 300–500 character segments (with optional overlap)
- Generate embeddings for chunks using OpenAI text-embedding-3-small
- Store chunk embeddings + metadata in the global vector store
- Provide a simple search interface for retrieving Top-K relevant chunks
  for a given question

This module does NOT deal with FastAPI routes directly.
It is intended to be used by app.py (or admin routes) and the
RAG answer generation logic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI

from services.vector_store import (
    add_chunk_embeddings,
    search_chunks,
)


# =========================
# Settings & OpenAI client
# =========================

EMBEDDING_MODEL = "text-embedding-3-small"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =========================
# Chunking helpers
# =========================

def chunk_text(
    text: str,
    max_chunk_chars: int = 400,
    overlap_chars: int = 50,
) -> List[Tuple[str, int, int]]:
    """
    Split a long string into overlapping character-based chunks.

    Args:
        text:
            Raw text content.
        max_chunk_chars:
            Target maximum length for each chunk (approx. 300–500 chars
            recommended for v0.7 MVP).
        overlap_chars:
            Number of characters to overlap between consecutive chunks
            to reduce boundary issues.

    Returns:
        A list of tuples: (chunk_text, start_index, end_index),
        where start_index and end_index are 0-based character indices
        into the original text.
    """
    text = text or ""
    text = text.strip()
    if not text:
        return []

    max_len = max(50, max_chunk_chars)  # avoid too small by mistake
    overlap = max(0, min(overlap_chars, max_len - 1))

    chunks: List[Tuple[str, int, int]] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_len, text_len)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append((chunk, start, end))

        if end >= text_len:
            break

        # 下一個 chunk 從 end - overlap 開始，創造重疊區
        start = max(0, end - overlap)

    return chunks


# =========================
# Embedding helpers
# =========================

def embed_text(text: str) -> List[float]:
    """
    Generate an embedding vector for a single piece of text.

    Uses OpenAI text-embedding-3-small for low-cost semantic embeddings.

    Args:
        text:
            Input text to embed.

    Returns:
        A list of floats representing the embedding vector.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Cannot embed empty text.")

    res = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return res.data[0].embedding


def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    """
    Generate embeddings for a sequence of texts.

    Args:
        texts:
            A sequence of non-empty strings.

    Returns:
        A list of embedding vectors, one per input text.
    """
    cleaned_texts = [(t or "").strip() for t in texts]
    if not cleaned_texts:
        return []

    # 過濾掉完全空白的內容，以免 API 出錯
    non_empty_texts: List[str] = [t for t in cleaned_texts if t]
    if not non_empty_texts:
        raise ValueError("All texts are empty; nothing to embed.")

    res = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=non_empty_texts,
    )
    embeddings = [item.embedding for item in res.data]

    # 如果中間有空 string，目前不支援保持原位置（MVP）
    # 之後若需要，可增加 mapping。
    if len(embeddings) != len(non_empty_texts):
        raise RuntimeError("Embedding response size mismatch.")

    return embeddings


# =========================
# KB building helpers
# =========================

def build_kb_from_text(
    text: str,
    file_name: str = "inline_text",
    page: Optional[int] = None,
    max_chunk_chars: int = 400,
    overlap_chars: int = 50,
) -> int:
    """
    Build knowledge base entries from a raw text string.

    This function:
    1) Chunks the text
    2) Embeds each chunk
    3) Adds the chunks to the global vector store with metadata

    Args:
        text:
            Raw text content.
        file_name:
            Logical file name for metadata and tracing (e.g., "lesson4.txt").
        page:
            Optional page number. For TXT-based ingestion, this is usually None.
            For PDF-based ingestion, this can be the actual page index (1-based).
        max_chunk_chars:
            Max characters per chunk.
        overlap_chars:
            Overlap between chunks.

    Returns:
        Number of chunks added to the vector store.
    """
    chunks = chunk_text(
        text=text,
        max_chunk_chars=max_chunk_chars,
        overlap_chars=overlap_chars,
    )
    if not chunks:
        return 0

    chunk_texts = [c[0] for c in chunks]
    embeddings = embed_texts(chunk_texts)

    metadatas: List[Dict[str, Any]] = []
    for idx, (_, start_char, end_char) in enumerate(chunks):
        chunk_id = f"{file_name}_chunk_{idx+1:04d}"
        metadatas.append(
            {
                "chunk_id": chunk_id,
                "file_name": file_name,
                "page": page,
                "start_char": start_char,
                "end_char": end_char,
            }
        )

    add_chunk_embeddings(
        embeddings=embeddings,
        texts=chunk_texts,
        metadatas=metadatas,
    )

    return len(chunks)


def build_kb_from_txt_file(
    file_path: str,
    max_chunk_chars: int = 400,
    overlap_chars: int = 50,
) -> int:
    """
    Build KB entries from a .txt file.

    This is the primary V1 ingestion path for MentorFlow v0.7.

    Args:
        file_path:
            Path to the .txt file.
        max_chunk_chars:
            Max characters per chunk.
        overlap_chars:
            Overlap between chunks.

    Returns:
        Number of chunks added.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"TXT file not found: {file_path}")

    text = path.read_text(encoding="utf-8")
    file_name = path.name

    return build_kb_from_text(
        text=text,
        file_name=file_name,
        page=None,
        max_chunk_chars=max_chunk_chars,
        overlap_chars=overlap_chars,
    )


def build_kb_from_pdf_file(
    file_path: str,
    max_chunk_chars: int = 400,
    overlap_chars: int = 50,
) -> int:
    """
    Build KB entries from a PDF file.

    This is an optional V1.1 feature. It uses pypdf for simple text extraction.
    If pypdf is not installed, this function will raise an ImportError.

    Args:
        file_path:
            Path to the PDF file.
        max_chunk_chars:
            Max characters per chunk.
        overlap_chars:
            Overlap between chunks.

    Returns:
        Total number of chunks added across all pages.
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for PDF ingestion. Please install it first."
        ) from exc

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    reader = PdfReader(str(path))
    total_chunks = 0
    file_name = path.name

    for page_index, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        raw_text = raw_text.strip()
        if not raw_text:
            continue

        added = build_kb_from_text(
            text=raw_text,
            file_name=file_name,
            page=page_index,
            max_chunk_chars=max_chunk_chars,
            overlap_chars=overlap_chars,
        )
        total_chunks += added

    return total_chunks


# =========================
# Retrieval for RAG answers
# =========================

def retrieve_relevant_chunks(
    question: str,
    top_k: int = 3,
    score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve Top-K relevant chunks for a given user question.
    """

    question = (question or "").strip()
    if not question:
        return []

    # 1) embed question
    query_embedding = embed_text(question)

    # 2) search vector DB
    results = search_chunks(
        query_embedding=query_embedding,
        top_k=top_k,
        score_threshold=score_threshold,  # if implemented in the store
    )

    # 2.5) fallback threshold filtering (in case vector store didn't apply it)
    if score_threshold is not None:
        results = [
            (item, score)
            for item, score in results
            if score is not None and score >= score_threshold
        ]

    # 3) format output
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

