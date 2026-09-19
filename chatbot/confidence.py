"""
Confidence scoring and escalation decision.

Separates the escalation logic from the retriever so it can be
tuned or replaced independently.
"""

from __future__ import annotations

from dataclasses import dataclass

from chatbot.retriever import RetrievalResult, CONFIDENCE_THRESHOLD


@dataclass
class ConfidenceReport:
    score: float           # top similarity score 0–1
    level: str             # "high" | "medium" | "low"
    should_escalate: bool
    reason: str            # human-readable explanation for UI badge


def assess(result: RetrievalResult) -> ConfidenceReport:
    """
    Determine whether the bot should answer or escalate.

    Escalation is triggered when:
      - Top similarity < CONFIDENCE_THRESHOLD (no relevant chunks found)

    Confidence levels:
      high   >= 0.65
      medium >= CONFIDENCE_THRESHOLD
      low    <  CONFIDENCE_THRESHOLD → escalate
    """
    score = result.top_score
    escalate = not result.confident

    if score >= 0.65:
        level = "high"
        reason = f"Strong match found (similarity {score:.0%})"
    elif score >= CONFIDENCE_THRESHOLD:
        level = "medium"
        reason = f"Partial match found (similarity {score:.0%}) — answer may be incomplete"
    else:
        level = "low"
        reason = (
            f"No sufficiently relevant information found in the knowledge base "
            f"(similarity {score:.0%} < threshold {CONFIDENCE_THRESHOLD:.0%}). "
            "Escalating to a human agent."
        )

    return ConfidenceReport(
        score=score,
        level=level,
        should_escalate=escalate,
        reason=reason,
    )
