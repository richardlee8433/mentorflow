"""
In-memory vector store for MentorFlow v0.7 RAG MVP.

- Stores embeddings + raw text + metadata for each chunk.
- Supports simple cosine-similarity Top-K search.
- Designed as a lightweight MVP before introducing a real vector DB (e.g. ChromaDB).

This module is intentionally minimal and framework-agnostic.
FastAPI routes or RAG services should import and use the global
`VECTOR_STORE` instance and the helper functions defined here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class VectorItem:
    """
    A single stored vector item in the in-memory vector store.

    Attributes:
        embedding: The embedding vector for this chunk (e.g., 1536-dim).
        text: The raw text content of the chunk.
        metadata: Arbitrary metadata associated with this chunk.
                  For MentorFlow v0.7, recommended keys include:
                  - "chunk_id": str
                  - "file_name": str
                  - "page": Optional[int]
                  - "start_char": Optional[int]
                  - "end_char": Optional[int]
    """
    embedding: List[float]
    text: str
    metadata: Dict[str, Any]


class InMemoryVectorStore:
    """
    Simple in-memory vector store using cosine similarity.

    This is the MVP implementation for MentorFlow v0.7 RAG.
    It keeps all embeddings in Python memory and does linear scan
    for retrieval. This is acceptable for small-scale usage and
    can be replaced later with a dedicated vector database.

    The store is append-only for now. If you need to rebuild the
    knowledge base (e.g., after re-uploading a file), call `clear()`
    and then re-ingest all chunks.
    """

    def __init__(self) -> None:
        self._items: List[VectorItem] = []

    def clear(self) -> None:
        """
        Remove all stored vector items.

        Typical use case:
        - Rebuild the KB after replacing the underlying documents.
        """
        self._items.clear()

    def add_item(self, item: VectorItem) -> None:
        """
        Add a single vector item to the store.

        Args:
            item: The VectorItem to add.
        """
        self._items.append(item)

    def add_items(self, items: Sequence[VectorItem]) -> None:
        """
        Add multiple vector items to the store.

        Args:
            items: A sequence of VectorItem objects.
        """
        if not items:
            return
        self._items.extend(items)

    def _cosine_similarity(
        self,
        a: Sequence[float],
        b: Sequence[float],
    ) -> float:
        """
        Compute cosine similarity between two vectors.

        Returns:
            A float in [-1, 1]. If either vector has zero norm,
            the similarity is defined as 0.0.
        """
        if len(a) != len(b):
            # For safety; in MVP we assume consistent embedding length.
            raise ValueError("Embedding dimensions do not match.")

        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0

        for ai, bi in zip(a, b):
            dot += ai * bi
            norm_a += ai * ai
            norm_b += bi * bi

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 3,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[VectorItem, float]]:
        """
        Retrieve the Top-K most similar items for a given query embedding.

        Args:
            query_embedding:
                The embedding vector of the query (same dimension as stored vectors).
            top_k:
                Maximum number of results to return. Default is 3.
            score_threshold:
                Optional minimum cosine similarity required to include a result.
                If None, no threshold is applied.

        Returns:
            A list of (VectorItem, score) tuples, sorted by descending score.
        """
        if not self._items:
            return []

        # Compute similarity for each item (linear scan).
        scored: List[Tuple[VectorItem, float]] = []
        for item in self._items:
            score = self._cosine_similarity(query_embedding, item.embedding)
            if score_threshold is not None and score < score_threshold:
                continue
            scored.append((item, score))

        # Sort by similarity descending and return top_k.
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(top_k, 0)]

    def size(self) -> int:
        """
        Returns:
            The number of vector items currently stored.
        """
        return len(self._items)


# Global singleton used by the rest of the application.
# For v0.7 we keep a single KB in memory. In the future, this can be
# extended to support multiple named stores (per-tenant, per-user, etc.).
VECTOR_STORE = InMemoryVectorStore()


def add_chunk_embedding(
    embedding: Sequence[float],
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Convenience helper to add a single chunk to the global VECTOR_STORE.

    Args:
        embedding:
            Embedding vector for this chunk.
        text:
            Raw text content of the chunk.
        metadata:
            Optional metadata dictionary (chunk_id, file_name, page, etc.).
    """
    item = VectorItem(
        embedding=list(embedding),
        text=text,
        metadata=metadata or {},
    )
    VECTOR_STORE.add_item(item)


def add_chunk_embeddings(
    embeddings: Sequence[Sequence[float]],
    texts: Sequence[str],
    metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
) -> None:
    """
    Convenience helper to bulk-add multiple chunks to the global VECTOR_STORE.

    Args:
        embeddings:
            A sequence of embedding vectors.
        texts:
            A sequence of raw text chunks (same length as embeddings).
        metadatas:
            Optional sequence of metadata dicts; if provided, length must
            match `embeddings` and `texts`. If None, empty metadata dicts
            will be used.
    """
    if len(embeddings) != len(texts):
        raise ValueError("embeddings and texts must have the same length.")

    if metadatas is not None and len(metadatas) != len(embeddings):
        raise ValueError("metadatas length must match embeddings and texts.")

    items: List[VectorItem] = []
    for idx, (emb, text) in enumerate(zip(embeddings, texts)):
        metadata = {}
        if metadatas is not None and metadatas[idx] is not None:
            metadata = dict(metadatas[idx])  # shallow copy

        items.append(
            VectorItem(
                embedding=list(emb),
                text=text,
                metadata=metadata,
            )
        )

    VECTOR_STORE.add_items(items)


def search_chunks(
    query_embedding: Sequence[float],
    top_k: int = 3,
    score_threshold: Optional[float] = None,
) -> List[Tuple[VectorItem, float]]:
    """
    Convenience helper to search the global VECTOR_STORE.

    Args:
        query_embedding:
            Embedding vector of the query.
        top_k:
            Maximum number of results to return. Default is 3.
        score_threshold:
            Optional minimum cosine similarity.

    Returns:
        A list of (VectorItem, score) tuples sorted by descending score.
    """
    return VECTOR_STORE.search(
        query_embedding=query_embedding,
        top_k=top_k,
        score_threshold=score_threshold,
    )
