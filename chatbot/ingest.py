"""
LearnForge AI Support Chatbot — knowledge ingestion pipeline.

Parses faqs.md, policies.md, and tickets.md, chunks each document entry
individually, attaches rich metadata, embeds with all-MiniLM-L6-v2, and
stores everything in a persistent ChromaDB collection.

Run once (or whenever the knowledge base changes):
    python -m chatbot.ingest
"""

from __future__ import annotations

# Suppress TensorFlow / Keras verbose output before any imports that trigger TF
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF_IMPORT", "1")

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

# ─────────────────────────────── paths ────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR                        # .md files live at project root
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "learnforge_kb"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ─────────────────────────── stale-data keywords ───────────────────────
_STALE_PATTERNS = [
    r"older?\s+version",
    r"outdated",
    r"previous\s+(version|article|policy|guidance|wording|documentation)",
    r"no longer\s+(applies?|current|accurate|offered|valid)",
    r"has been (retired|removed|replaced)",
    r"should not be treated as",
    r"that wording\s+.{0,20}replaced",
    r"that article\s+.{0,30}(outdated|older|incorrect)",
    r"IMPORTANT:.*older version",
    r"old.{0,20}(stated|said|recommended|required|instructed)",
]
_STALE_RE = re.compile("|".join(_STALE_PATTERNS), re.IGNORECASE)

# ─────────────────────────── date extraction ───────────────────────────
_DATE_RE = re.compile(
    r"(?:Last reviewed|Reviewed|Updated|Effective(?: date)?|Last updated)"
    r"\s*:?\s*([A-Za-z]+\s+\d{4}|\d{4}-\d{2}-\d{2}|\w+ \d{4})",
    re.IGNORECASE,
)

# Map month names → sortable int for recency comparison
_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _parse_date_sort_key(date_str: str) -> int:
    """Return YYYYMM int for sorting; 0 if unparseable."""
    if not date_str or date_str == "unknown":
        return 0
    m = re.search(r"(\w+)\s+(\d{4})", date_str)
    if m:
        month = _MONTH_MAP.get(m.group(1).lower(), 0)
        year = int(m.group(2))
        return year * 100 + month
    m2 = re.search(r"(\d{4})-(\d{2})", date_str)
    if m2:
        return int(m2.group(1)) * 100 + int(m2.group(2))
    return 0


# ──────────────────────────── data model ───────────────────────────────
@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    doc_type: str          # faq | policy | ticket
    doc_id: str            # FAQ-02, POLICY-05, TICKET-11
    title: str
    last_reviewed: str     # human-readable date or "unknown"
    date_sort_key: int     # YYYYMM int for recency comparison
    is_outdated_flag: bool
    chunk_index: int = 0
    topics: list[str] = field(default_factory=list)   # coarse topic tags


# ─────────────────────────── topic tagging ────────────────────────────
_TOPIC_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\brefund\b|\bmoney.back\b|\bprice.adjust", re.I), "refund"),
    (re.compile(r"\bsubscription\b|\brenewal\b|\bbilling\b|\bplan\b", re.I), "subscription"),
    (re.compile(r"\bpayment\b|\bcharge\b|\bcard\b|\btransaction\b|\bchargeback\b", re.I), "payment"),
    (re.compile(r"\bcertificate\b|\bcompleti", re.I), "certificate"),
    (re.compile(r"\bprogress\b|\bsync", re.I), "progress"),
    (re.compile(r"\boffline\b|\bdownload\b", re.I), "offline"),
    (re.compile(r"\bpassword\b|\bsign.?in\b|\blogin\b|\bsecurity\b", re.I), "account_security"),
    (re.compile(r"\bemail\b|\baccount.*change\b|\bchange.*email\b", re.I), "account_management"),
    (re.compile(r"\bcaption\b|\baccessib\b|\bscreen.?reader\b|\btranscript\b", re.I), "accessibility"),
    (re.compile(r"\bmobile\b|\bapp\b|\biphone\b|\bandroid\b", re.I), "mobile"),
    (re.compile(r"\bvideo\b|\bplayback\b|\bstream\b|\bload", re.I), "video_playback"),
    (re.compile(r"\bfamily\b|\borganization\b|\bteam\b|\bshare\b", re.I), "account_sharing"),
    (re.compile(r"\bbrowser\b|\bdevice\b|\bchrome\b|\bsafari\b|\bfirefox\b", re.I), "technical"),
]


def _tag_topics(text: str) -> list[str]:
    return [tag for pattern, tag in _TOPIC_RULES if pattern.search(text)]


# ───────────────────────── parser helpers ─────────────────────────────

def _extract_date(text: str) -> str:
    m = _DATE_RE.search(text)
    return m.group(1).strip() if m else "unknown"


def _is_stale(text: str) -> bool:
    return bool(_STALE_RE.search(text))


# ──────────────────── document parsers ───────────────────────────────

def parse_faqs(path: Path) -> list[Chunk]:
    """Parse FAQ-NN entries from faqs.md."""
    text = path.read_text(encoding="utf-8")
    # Split on FAQ-NN headers
    blocks = re.split(r"\n(?=# FAQ-\d+)", text)
    chunks: list[Chunk] = []
    for block in blocks:
        m = re.match(r"# (FAQ-\d+)\s+[—–-]\s+(.+)", block)
        if not m:
            continue
        doc_id = m.group(1)
        title = m.group(2).strip()
        date = _extract_date(block)
        stale = _is_stale(block)
        topics = _tag_topics(block)
        chunk = Chunk(
            chunk_id=f"{doc_id.lower()}-chunk-0",
            text=block.strip(),
            source_file="faqs.md",
            doc_type="faq",
            doc_id=doc_id,
            title=title,
            last_reviewed=date,
            date_sort_key=_parse_date_sort_key(date),
            is_outdated_flag=stale,
            chunk_index=0,
            topics=topics,
        )
        chunks.append(chunk)
    return chunks


def parse_policies(path: Path) -> list[Chunk]:
    """Parse POLICY-NN entries from policies.md."""
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=# POLICY-\d+)", text)
    chunks: list[Chunk] = []
    for block in blocks:
        m = re.match(r"# (POLICY-\d+)\s+[—–-]\s+(.+)", block)
        if not m:
            continue
        doc_id = m.group(1)
        title = m.group(2).strip()
        date = _extract_date(block)
        stale = _is_stale(block)
        topics = _tag_topics(block)
        chunk = Chunk(
            chunk_id=f"{doc_id.lower()}-chunk-0",
            text=block.strip(),
            source_file="policies.md",
            doc_type="policy",
            doc_id=doc_id,
            title=title,
            last_reviewed=date,
            date_sort_key=_parse_date_sort_key(date),
            is_outdated_flag=stale,
            chunk_index=0,
            topics=topics,
        )
        chunks.append(chunk)
    return chunks


def parse_tickets(path: Path) -> list[Chunk]:
    """Parse TICKET-NN entries from tickets.md."""
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=# TICKET-\d+)", text)
    chunks: list[Chunk] = []
    for block in blocks:
        m = re.match(r"# (TICKET-\d+)\s+[—–-]\s+(.+)", block)
        if not m:
            continue
        doc_id = m.group(1)
        title = m.group(2).strip()
        date = _extract_date(block)
        stale = _is_stale(block)
        topics = _tag_topics(block)
        # Extract status line for extra metadata
        status_m = re.search(r"STATUS:\s*(.+)", block, re.IGNORECASE)
        status = status_m.group(1).strip() if status_m else "unknown"
        chunk = Chunk(
            chunk_id=f"{doc_id.lower()}-chunk-0",
            text=block.strip(),
            source_file="tickets.md",
            doc_type="ticket",
            doc_id=doc_id,
            title=title,
            last_reviewed=date,
            date_sort_key=_parse_date_sort_key(date),
            is_outdated_flag=stale,
            chunk_index=0,
            topics=topics,
        )
        chunks.append(chunk)
    return chunks


# ─────────────────────────── ingestion ────────────────────────────────

def ingest(reset: bool = False) -> chromadb.Collection:
    """
    Parse all knowledge-base files, embed them, and upsert into ChromaDB.

    Args:
        reset: If True, delete and recreate the collection from scratch.

    Returns:
        The populated ChromaDB collection.
    """
    print("[KB] Loading knowledge base files...")
    faqs_path = KB_DIR / "faqs.md"
    policies_path = KB_DIR / "policies.md"
    tickets_path = KB_DIR / "tickets.md"

    chunks: list[Chunk] = []
    chunks.extend(parse_faqs(faqs_path))
    chunks.extend(parse_policies(policies_path))
    chunks.extend(parse_tickets(tickets_path))

    print(f"   Parsed {len(chunks)} chunks total "
          f"({sum(1 for c in chunks if c.doc_type=='faq')} FAQs, "
          f"{sum(1 for c in chunks if c.doc_type=='policy')} policies, "
          f"{sum(1 for c in chunks if c.doc_type=='ticket')} tickets)")

    # ── embed ──────────────────────────────────────────────────────────
    print(f"\n[EMBED] Loading embedding model: {EMBED_MODEL_NAME} ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c.text for c in chunks]
    print("   Embedding all chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    # ── chroma ─────────────────────────────────────────────────────────
    print(f"\n[CHROMA] Connecting to ChromaDB at: {CHROMA_DIR}")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("   Deleted existing collection.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ── upsert ─────────────────────────────────────────────────────────
    ids = [c.chunk_id for c in chunks]
    metadatas = [
        {
            "source_file": c.source_file,
            "doc_type": c.doc_type,
            "doc_id": c.doc_id,
            "title": c.title,
            "last_reviewed": c.last_reviewed,
            "date_sort_key": c.date_sort_key,
            "is_outdated_flag": str(c.is_outdated_flag),
            "chunk_index": c.chunk_index,
            "topics": ",".join(c.topics) if c.topics else "general",
        }
        for c in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=[e.tolist() for e in embeddings],
        documents=texts,
        metadatas=metadatas,
    )

    print(f"\n[DONE] Ingestion complete. Collection '{COLLECTION_NAME}' "
          f"now has {collection.count()} documents.")
    return collection


if __name__ == "__main__":
    import sys
    reset_flag = "--reset" in sys.argv
    ingest(reset=reset_flag)
