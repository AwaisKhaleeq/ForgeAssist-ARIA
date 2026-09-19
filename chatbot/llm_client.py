"""
Groq LLM client wrapper.

Reads GROQ_API_KEY from the environment (.env file loaded by the caller).
"""

from __future__ import annotations

import os
from typing import Generator

from groq import Groq

from pathlib import Path
from dotenv import load_dotenv

# Model to use — Groq models available on free tier
# Defaults to qwen/qwen3.8-27b or custom set via GROQ_MODEL env var
DEFAULT_MODEL = "qwen/qwen3.8-27b"
GROQ_MODEL = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
MAX_TOKENS = 512
TEMPERATURE = 0.2   # Low temperature → more factual, less creative


def _get_client() -> Groq:
    # Always reload .env so changes made while the app is running take effect
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return Groq(api_key=api_key)



def chat_completion(messages: list[dict], stream: bool = True) -> Generator[str, None, None]:
    """
    Send messages to Groq and yield response text token by token.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        stream: If True, yields tokens as they arrive (for live UI).

    Yields:
        Text fragments from the LLM.
    """
    client = _get_client()
    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)

    if stream:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stream=False,
        )
        yield response.choices[0].message.content or ""
