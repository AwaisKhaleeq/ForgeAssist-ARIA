"""
Retriever — vector search + contradiction detection.

Wraps ChromaDB queries and adds:
  1. Cosine similarity → confidence score
  2. Contradiction detection (same topic, different answers, multiple sources)
  3. Staleness surfacing (newest chunk wins, conflict annotated)
"""

from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF_IMPORT", "1")

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from chatbot.ingest import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL_NAME

# ─────────────────────── tuneable constants ────────────────────────────
TOP_K = 5                     # chunks to retrieve
CONFIDENCE_THRESHOLD = 0.55   # cosine similarity; below → escalate
CONTRADICTION_TOPIC_OVERLAP = 1   # shared topics needed to flag conflict


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    doc_type: str          # faq | policy | ticket
    doc_id: str
    title: str
    source_file: str
    last_reviewed: str
    date_sort_key: int
    is_outdated_flag: bool
    topics: list[str]
    similarity: float      # 0–1; higher = more relevant


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    top_score: float
    confident: bool
    contradictions: list[dict]   # list of {topic, chunks: [a, b], winner: chunk}
    has_stale: bool


# ───────────────────────── singleton model ──────────────────────────────
_model: Optional[SentenceTransformer] = None
_collection: Optional[chromadb.Collection] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


# ──────────────────────── contradiction logic ────────────────────────────

def _detect_contradictions(chunks: list[RetrievedChunk]) -> list[dict]:
    """
    Find pairs of chunks that:
      - Share at least one topic tag
      - Come from different source files (faq vs policy vs ticket)
      OR explicitly flag stale content for the same topic.

    Returns a list of conflict dicts, each with:
      {topic, chunk_a, chunk_b, winner (newer by date_sort_key)}
    """
    conflicts: list[dict] = []
    seen_pairs: set[frozenset] = set()

    for i, a in enumerate(chunks):
        for j, b in enumerate(chunks):
            if i >= j:
                continue
            pair_key = frozenset([a.chunk_id, b.chunk_id])
            if pair_key in seen_pairs:
                continue

            shared_topics = set(a.topics) & set(b.topics)
            if not shared_topics:
                continue

            # Contradiction conditions:
            # 1. Different source files addressing same topic
            # 2. One or both flagged as potentially outdated
            is_cross_source = a.source_file != b.source_file
            either_stale = a.is_outdated_flag or b.is_outdated_flag

            if is_cross_source or either_stale:
                winner = a if a.date_sort_key >= b.date_sort_key else b
                loser = b if winner is a else a
                conflicts.append({
                    "topics": list(shared_topics),
                    "chunk_a": a,
                    "chunk_b": b,
                    "winner": winner,
                    "loser": loser,
                    "reason": (
                        "stale content flagged in source"
                        if either_stale
                        else "conflicting sources for same topic"
                    ),
                })
                seen_pairs.add(pair_key)

    return conflicts


# ───────────────────────────── main API ─────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K) -> RetrievalResult:
    """
    Embed the query, search ChromaDB, detect contradictions, assess confidence.

    Args:
        query: The user's raw question text.
        top_k: Number of candidate chunks to retrieve.

    Returns:
        A RetrievalResult with all metadata needed by the prompt builder.
    """
    model = _get_model()
    collection = _get_collection()

    # Embed query (normalised for cosine similarity)
    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    # ChromaDB with cosine space returns distances in [0, 2]; convert to similarity
    distances = results["distances"][0]
    similarities = [max(0.0, 1.0 - (d / 2.0)) for d in distances]

    chunks: list[RetrievedChunk] = []
    for doc, meta, sim in zip(docs, metas, similarities):
        topics_raw = meta.get("topics", "general")
        topics = [t.strip() for t in topics_raw.split(",") if t.strip()]
        chunks.append(
            RetrievedChunk(
                chunk_id=meta.get("doc_id", "unknown"),
                text=doc,
                doc_type=meta.get("doc_type", "unknown"),
                doc_id=meta.get("doc_id", "unknown"),
                title=meta.get("title", ""),
                source_file=meta.get("source_file", ""),
                last_reviewed=meta.get("last_reviewed", "unknown"),
                date_sort_key=int(meta.get("date_sort_key", 0)),
                is_outdated_flag=meta.get("is_outdated_flag", "False") == "True",
                topics=topics,
                similarity=round(sim, 4),
            )
        )

    top_score = similarities[0] if similarities else 0.0
    confident = top_score >= CONFIDENCE_THRESHOLD
    contradictions = _detect_contradictions(chunks)
    has_stale = any(c.is_outdated_flag for c in chunks)

    return RetrievalResult(
        chunks=chunks,
        top_score=round(top_score, 4),
        confident=confident,
        contradictions=contradictions,
        has_stale=has_stale,
    )
