"""
Prompt builder — assembles the full Groq prompt from:
  - System persona + hard rules
  - Retrieved context chunks (with metadata labels)
  - Detected contradictions / staleness warnings
  - Conversation history (last N turns)
  - Current user query
"""

from __future__ import annotations

from chatbot.retriever import RetrievalResult, RetrievedChunk

# Maximum conversation turns to include in the prompt
MAX_HISTORY_TURNS = 5

SYSTEM_PROMPT = """You are Aria, a helpful and honest customer support agent for LearnForge — an online education platform.

Your role:
- Answer learner questions accurately using ONLY the provided knowledge base context.
- Be concise, warm, and professional.
- If the context contains contradictions or flagged outdated information, explicitly acknowledge this and rely on the most recently reviewed source.
- If you are not sure or the context does not cover the question, say so clearly — do NOT make up information.
- Never invent policies, prices, deadlines, or procedures not found in the provided context.
- When escalating, always explain why and provide the next step for the user.

Formatting rules:
- Use plain prose. Avoid excessive bullet points unless listing steps.
- Keep responses under 200 words unless the question genuinely requires more detail.
- Always end with a clear action or next step for the user.
"""


def _format_chunk(chunk: RetrievedChunk, idx: int) -> str:
    """Format a single retrieved chunk with its metadata header."""
    badges = []
    if chunk.is_outdated_flag:
        badges.append("⚠️ CONTAINS OUTDATED CONTENT")
    if chunk.last_reviewed != "unknown":
        badges.append(f"Last reviewed: {chunk.last_reviewed}")

    badge_line = " | ".join(badges) if badges else ""
    header = (
        f"[SOURCE {idx + 1}: {chunk.doc_id} — {chunk.title} "
        f"({chunk.doc_type.upper()}, {chunk.source_file})"
        + (f" | {badge_line}" if badge_line else "")
        + "]"
    )
    return f"{header}\n{chunk.text}"


def _format_contradictions(contradictions: list[dict]) -> str:
    """Produce a warning block about detected contradictions."""
    if not contradictions:
        return ""
    lines = ["⚠️ CONTRADICTION NOTICE — The following sources address overlapping topics with potentially conflicting information:"]
    for c in contradictions:
        topics_str = ", ".join(c["topics"])
        winner = c["winner"]
        loser = c["loser"]
        lines.append(
            f"  • Topic(s): {topics_str}\n"
            f"    Authoritative source (newer): {winner.doc_id} ({winner.last_reviewed})\n"
            f"    Potentially outdated source: {loser.doc_id} ({loser.last_reviewed})\n"
            f"    Reason: {c['reason']}"
        )
    lines.append("When answering, prefer the authoritative (newer) source and explicitly note if older guidance is being superseded.")
    return "\n".join(lines)


def build_prompt(
    query: str,
    retrieval: RetrievalResult,
    history: list[dict],   # [{"role": "user"|"assistant", "content": str}, ...]
) -> list[dict]:
    """
    Build the messages list for the Groq chat completion API.

    Returns:
        List of {"role": ..., "content": ...} dicts ready for Groq.
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # ── context block ──────────────────────────────────────────────────
    context_parts: list[str] = ["=== KNOWLEDGE BASE CONTEXT ===\n"]

    if retrieval.chunks:
        for idx, chunk in enumerate(retrieval.chunks):
            context_parts.append(_format_chunk(chunk, idx))
            context_parts.append("")  # blank line between chunks
    else:
        context_parts.append("No relevant documents found in the knowledge base.")

    # Contradiction warnings
    contradiction_block = _format_contradictions(retrieval.contradictions)
    if contradiction_block:
        context_parts.append("\n" + contradiction_block)

    context_parts.append("\n=== END OF CONTEXT ===")
    context_message = "\n".join(context_parts)

    # Inject context as a system-level message (keeps it clearly separate)
    messages.append({"role": "system", "content": context_message})

    # ── conversation history ───────────────────────────────────────────
    recent_history = history[-(MAX_HISTORY_TURNS * 2):]  # user+assistant pairs
    for turn in recent_history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # ── current user query ─────────────────────────────────────────────
    messages.append({"role": "user", "content": query})

    return messages


def build_escalation_prompt(query: str, reason: str, history: list[dict]) -> list[dict]:
    """
    Build a shorter prompt for the escalation response.
    The LLM still produces a polite message acknowledging the escalation.
    """
    system = (
        SYSTEM_PROMPT
        + f"\n\nCRITICAL: You do NOT have enough information to answer this question. "
        f"Reason: {reason}\n"
        "Respond with a brief, empathetic message acknowledging you cannot fully answer, "
        "and tell the user you are escalating to a human support agent. "
        "Suggest they contact support@learnforge.com or use the live chat button. "
        "Do NOT attempt to answer the question."
    )
    messages: list[dict] = [{"role": "system", "content": system}]
    recent_history = history[-(MAX_HISTORY_TURNS * 2):]
    for turn in recent_history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": query})
    return messages
