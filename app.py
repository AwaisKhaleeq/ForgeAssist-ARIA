"""
LearnForge AI Support Chatbot — Streamlit frontend.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

# Suppress TensorFlow / Keras verbose logs before any ML imports
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF_IMPORT", "1")

import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load .env before importing chatbot modules that read env vars
load_dotenv(Path(__file__).parent / ".env")

from chatbot.retriever import retrieve, CONFIDENCE_THRESHOLD
from chatbot.confidence import assess
from chatbot.prompt_builder import build_prompt, build_escalation_prompt
from chatbot.llm_client import chat_completion, GROQ_MODEL
from chatbot.ingest import ingest, CHROMA_DIR, COLLECTION_NAME

# ─────────────────────────── page config ────────────────────────────────
st.set_page_config(
    page_title="LearnForge Support — Aria",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── custom CSS ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* ── main container ── */
.block-container {
    max-width: 820px;
    padding-top: 2rem;
}

/* ── hero header ── */
.hero-header {
    text-align: center;
    padding: 2rem 1rem 1rem;
}
.hero-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero-header p {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: 0;
}

/* ── chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
}

/* ── user bubble ── */
[data-testid="stChatMessage"][data-testid*="user"],
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(99, 102, 241, 0.18);
}

/* ── assistant bubble ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(255,255,255,0.05);
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.85);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}

/* ── confidence badge ── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin: 2px 3px;
}
.badge-high   { background: rgba(52,211,153,0.2); color:#34d399; border:1px solid #34d399; }
.badge-medium { background: rgba(251,191,36,0.2);  color:#fbbf24; border:1px solid #fbbf24; }
.badge-low    { background: rgba(239,68,68,0.2);   color:#ef4444; border:1px solid #ef4444; }
.badge-stale  { background: rgba(251,146,60,0.2);  color:#fb923c; border:1px solid #fb923c; }
.badge-conflict { background: rgba(167,139,250,0.2); color:#a78bfa; border:1px solid #a78bfa; }
.badge-faq    { background: rgba(96,165,250,0.2);  color:#60a5fa; border:1px solid #60a5fa; }
.badge-policy { background: rgba(52,211,153,0.2);  color:#34d399; border:1px solid #34d399; }
.badge-ticket { background: rgba(248,113,113,0.2); color:#f87171; border:1px solid #f87171; }

/* ── escalation banner ── */
.escalation-box {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(251,146,60,0.10));
    border: 1px solid rgba(239,68,68,0.40);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 0.8rem;
}
.escalation-box h4 { color: #f87171; margin: 0 0 0.3rem; }
.escalation-box p  { color: #fca5a5; margin: 0; font-size: 0.9rem; }

/* ── conflict warning ── */
.conflict-box {
    background: rgba(167,139,250,0.08);
    border: 1px solid rgba(167,139,250,0.35);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin-top: 0.6rem;
    font-size: 0.85rem;
    color: #c4b5fd;
}

/* ── stale warning ── */
.stale-box {
    background: rgba(251,146,60,0.08);
    border: 1px solid rgba(251,146,60,0.35);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin-top: 0.4rem;
    font-size: 0.85rem;
    color: #fdba74;
}

/* ── source expander ── */
details summary {
    cursor: pointer;
    color: #94a3b8;
    font-size: 0.82rem;
    margin-top: 0.6rem;
}

/* ── input ── */
[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(167,139,250,0.4) !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── metric cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 0.8rem 1rem;
}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
}

/* ── divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── spinner ── */
.stSpinner > div { border-top-color: #a78bfa !important; }

/* ── suggestion chips ── */
.suggestion-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.5rem 0 1rem; }
.chip {
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.8rem;
    color: #a5b4fc;
    cursor: pointer;
    transition: background 0.2s;
}
.chip:hover { background: rgba(99,102,241,0.30); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── helpers ─────────────────────────────────────

def ensure_kb_ready() -> bool:
    """Return True if ChromaDB collection exists with data, else False."""
    import chromadb
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        col = client.get_collection(COLLECTION_NAME)
        return col.count() > 0
    except Exception:
        return False


def badge(label: str, kind: str) -> str:
    return f'<span class="badge badge-{kind}">{label}</span>'


def doc_type_badge(doc_type: str) -> str:
    label_map = {"faq": "FAQ", "policy": "Policy", "ticket": "Ticket"}
    return badge(label_map.get(doc_type, doc_type.upper()), doc_type)


def confidence_badge(level: str, score: float) -> str:
    return badge(f"Confidence: {level.upper()} ({score:.0%})", level)


# ─────────────────────────── session state ───────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []   # {"role", "content"}
if "debug_info" not in st.session_state:
    st.session_state.debug_info = []  # metadata per assistant turn
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "escalations" not in st.session_state:
    st.session_state.escalations = 0
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = ensure_kb_ready()
if "suggested_query" not in st.session_state:
    st.session_state.suggested_query = None


# ───────────────────────────── sidebar ───────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 LearnForge Support")
    st.markdown("---")

    # KB status
    if st.session_state.kb_ready:
        st.success("✅ Knowledge base ready")
    else:
        st.error("⚠️ Knowledge base not ingested")
        if st.button("🔄 Ingest Knowledge Base"):
            with st.spinner("Ingesting…"):
                try:
                    ingest(reset=True)
                    st.session_state.kb_ready = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.markdown("---")

    # Session stats
    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.total_queries)
    with col2:
        st.metric("Escalations", st.session_state.escalations)

    st.markdown("---")

    # Model info
    active_model = os.getenv("GROQ_MODEL", GROQ_MODEL)
    st.markdown("### ⚙️ Config")
    st.markdown(f"""
    - **LLM:** `{active_model}`
    - **Embeddings:** `all-MiniLM-L6-v2`
    - **Vector DB:** ChromaDB (local)
    - **Confidence threshold:** `{CONFIDENCE_THRESHOLD:.0%}`
    - **History window:** 5 turns
    """)

    st.markdown("---")

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.debug_info = []
        st.session_state.total_queries = 0
        st.session_state.escalations = 0
        st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Try asking:")
    suggestions = [
        "Can I get a refund for my course?",
        "I can't find my certificate.",
        "Can I download courses offline?",
        "My video won't load.",
        "I was charged twice.",
        "How do I cancel my subscription?",
    ]
    for s in suggestions:
        if st.button(s, key=f"sug_{s[:20]}"):
            st.session_state.suggested_query = s
            st.rerun()


# ────────────────────────── hero header ───────────────────────────────────
st.markdown("""
<div class="hero-header">
  <h1>🎓 LearnForge Support</h1>
  <p>Hi! I'm <strong>Aria</strong>, your AI support assistant. Ask me anything about your courses, billing, or account.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────── KB guard ──────────────────────────────────────
if not st.session_state.kb_ready:
    st.warning(
        "⚠️ The knowledge base hasn't been ingested yet. "
        "Click **'Ingest Knowledge Base'** in the sidebar to set up the vector store."
    )
    st.stop()

# ─────────────────────────── metadata renderer ────────────────────────────
def _render_metadata(dbg: dict):
    """Render source badges, confidence, and warnings under an assistant message."""
    badge_html = []

    # Confidence badge
    badge_html.append(confidence_badge(dbg["confidence_level"], dbg["top_score"]))

    # Escalation badge
    if dbg.get("escalated"):
        badge_html.append(badge("🚨 ESCALATED", "low"))

    # Stale badge
    if dbg.get("has_stale"):
        badge_html.append(badge("⚠️ Stale Source", "stale"))

    # Conflict badge
    if dbg.get("contradictions"):
        badge_html.append(badge(f"⚡ {len(dbg['contradictions'])} Conflict(s)", "conflict"))

    st.markdown(" ".join(badge_html), unsafe_allow_html=True)

    # Source docs
    if dbg.get("sources"):
        with st.expander(f"📚 Sources ({len(dbg['sources'])} chunks retrieved)"):
            for src in dbg["sources"]:
                src_html = (
                    f"{doc_type_badge(src['doc_type'])} "
                    f"<strong>{src['doc_id']}</strong> — {src['title']} "
                    f"<code>sim={src['similarity']:.0%}</code>"
                    + (" &nbsp;⚠️ stale" if src.get("is_outdated_flag") else "")
                    + (f" &nbsp;📅 {src['last_reviewed']}" if src["last_reviewed"] != "unknown" else "")
                )
                st.markdown(src_html, unsafe_allow_html=True)

    # Contradiction details
    if dbg.get("contradictions"):
        for c in dbg["contradictions"]:
            topics_str = ", ".join(c["topics"])
            st.markdown(
                f'<div class="conflict-box">⚡ <strong>Conflict detected</strong> on topic(s): '
                f'<em>{topics_str}</em><br>'
                f'Authoritative source: <strong>{c["winner_id"]}</strong> '
                f'({c["winner_date"]}) &gt; {c["loser_id"]} ({c["loser_date"]})<br>'
                f'Reason: {c["reason"]}</div>',
                unsafe_allow_html=True,
            )

    # Stale warning
    if dbg.get("has_stale") and not dbg.get("escalated"):
        st.markdown(
            '<div class="stale-box">⚠️ One or more retrieved sources contain '
            'explicitly flagged outdated content. The answer above uses the most '
            'recently reviewed information available.</div>',
            unsafe_allow_html=True,
        )

    # Escalation banner
    if dbg.get("escalated"):
        st.markdown(
            '<div class="escalation-box">'
            '<h4>🚨 Escalated to Human Support</h4>'
            '<p>This question couldn\'t be confidently answered from the knowledge base. '
            'Please reach out to a human agent:<br>'
            '📧 <strong>support@learnforge.com</strong> &nbsp;|&nbsp; '
            '💬 Use the <strong>Live Chat</strong> button on the LearnForge website.</p>'
            '</div>',
            unsafe_allow_html=True,
        )


# ───────────────────── render chat history ────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

        # Show metadata badges for assistant turns
        if msg["role"] == "assistant":
            turn_idx = sum(1 for m in st.session_state.messages[:i+1] if m["role"] == "assistant") - 1
            if turn_idx < len(st.session_state.debug_info):
                dbg = st.session_state.debug_info[turn_idx]
                _render_metadata(dbg)


# ─────────────────────────── chat input ─────────────────────────────────
# Handle sidebar suggestion chips
default_val = ""
if st.session_state.suggested_query:
    default_val = st.session_state.suggested_query
    st.session_state.suggested_query = None

user_input = st.chat_input(
    "Ask about courses, billing, certificates, refunds…",
    key="chat_input",
)

# Use suggestion if no typed input
if not user_input and default_val:
    user_input = default_val

if user_input:
    # ── display user message ──────────────────────────────────────────
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    st.session_state.total_queries += 1

    # ── retrieve ──────────────────────────────────────────────────────
    with st.spinner("🔍 Searching knowledge base…"):
        try:
            retrieval = retrieve(user_input)
        except Exception as e:
            st.error(f"Retrieval error: {e}")
            st.stop()

    # ── confidence ────────────────────────────────────────────────────
    conf = assess(retrieval)

    # ── build prompt ──────────────────────────────────────────────────
    # Pass only role/content pairs as history
    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.messages[:-1]]  # exclude current user msg

    if conf.should_escalate:
        messages = build_escalation_prompt(user_input, conf.reason, history)
        st.session_state.escalations += 1
    else:
        messages = build_prompt(user_input, retrieval, history)

    # ── stream response ───────────────────────────────────────────────
    with st.chat_message("assistant", avatar="🤖"):
        response_placeholder = st.empty()
        full_response = ""
        try:
            for token in chat_completion(messages, stream=True):
                full_response += token
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        except Exception as e:
            full_response = (
                f"I'm sorry, I encountered an error connecting to the AI service: {e}. "
                "Please check your GROQ_API_KEY in the .env file and try again."
            )
            response_placeholder.error(full_response)

        # ── build debug metadata ──────────────────────────────────────
        dbg = {
            "confidence_level": conf.level,
            "top_score": conf.score,
            "escalated": conf.should_escalate,
            "has_stale": retrieval.has_stale,
            "contradictions": [
                {
                    "topics": c["topics"],
                    "winner_id": c["winner"].doc_id,
                    "winner_date": c["winner"].last_reviewed,
                    "loser_id": c["loser"].doc_id,
                    "loser_date": c["loser"].last_reviewed,
                    "reason": c["reason"],
                }
                for c in retrieval.contradictions
            ],
            "sources": [
                {
                    "doc_id": ch.doc_id,
                    "title": ch.title,
                    "doc_type": ch.doc_type,
                    "source_file": ch.source_file,
                    "similarity": ch.similarity,
                    "last_reviewed": ch.last_reviewed,
                    "is_outdated_flag": ch.is_outdated_flag,
                }
                for ch in retrieval.chunks
            ],
        }
        st.session_state.debug_info.append(dbg)

        # Render metadata immediately after this response
        _render_metadata(dbg)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": full_response})
