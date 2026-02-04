from __future__ import annotations
from typing import List, Tuple
import time
from openai import OpenAI
from openai import (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    AuthenticationError,
)
from core.config import settings
from core.logging import setup_logger
from rag.schemas import Chunk

logger = setup_logger("llm")
def build_context(hits: List[Tuple[Chunk, float]], max_chars: int) -> str:
    """
    Compact context with explicit per-chunk headers (for grounding + citations).
    """
    parts: List[str] = []
    used = 0

    for chunk, score in hits:
        header = f"[source={chunk.source} chunk_id={chunk.chunk_id}"
        if chunk.page is not None:
            header += f" page={chunk.page}"
        header += f" score={score:.3f}]"

        block = f"{header}\n{chunk.text}\n"
        if used + len(block) > max_chars:
            break

        parts.append(block)
        used += len(block)

    return "\n".join(parts).strip()

def retrieval_fallback_answer(hits: List[Tuple[Chunk, float]]) -> str:
    """
    If LLM is unavailable/quota-limited, return deterministic answer from excerpts.
    Still grounded and meets "no hallucination" requirement.
    """
    if not hits:
        return "I don't know based on the uploaded documents."

    lines = ["LLM is unavailable. Showing the most relevant excerpts:"]
    for c, score in hits[:5]:
        snippet = c.text[:280].replace("\n", " ").strip()
        lines.append(f"- ({c.source}#chunk{c.chunk_id}, score={score:.3f}) {snippet}...")

    return "\n".join(lines)

def grounded_answer(question: str, hits: List[Tuple[Chunk, float]]) -> Tuple[str, List[str], float]:
    """
    Returns (answer_text, citations_list, latency_seconds)

    citations_list example:
      ["doc.pdf#chunk12", "notes.txt#chunk3"]
    """
    t0 = time.perf_counter()
    if not question.strip():
        return ("Please enter a question.", [], 0.0)

    citations = [f"{c.source}#chunk{c.chunk_id}" for c, _ in hits]
    if not hits:
        dt = time.perf_counter() - t0
        return ("I don't know based on the uploaded documents.", [], dt)

    # If key is missing, do retrieval-only fallback (still grounded)
    if not settings.OPENAI_API_KEY:
        dt = time.perf_counter() - t0
        return (retrieval_fallback_answer(hits), citations, dt)

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    system = (
        "You are a helpful AI knowledge assistant.\n"
        "Rules:\n"
        "1) Answer ONLY using the provided CONTEXT.\n"
        "2) If the answer is not in the context, say exactly: "
        "\"I don't know based on the uploaded documents.\" \n"
        "3) Keep the answer concise.\n"
        "4) After the answer, add a 'Sources:' section listing the sources used "
        "(use source filename and chunk_id).\n"
        "Do not invent facts."
    )

    context = build_context(hits, settings.MAX_CONTEXT_CHARS)
    user = f"""QUESTION:
{question}

CONTEXT:
{context}
"""

    try:
        resp = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )

        text = resp.choices[0].message.content.strip()
        dt = time.perf_counter() - t0
        logger.info(f"LLM answered in {dt:.2f}s")
        return (text, citations, dt)

    except (RateLimitError, AuthenticationError, APITimeoutError, APIConnectionError) as e:
        dt = time.perf_counter() - t0
        logger.warning(f"LLM unavailable, falling back to excerpts. error={e}")
        return (retrieval_fallback_answer(hits), citations, dt)

    except Exception as e:
        dt = time.perf_counter() - t0
        logger.exception(f"Unexpected LLM error, falling back. error={e}")
        return (retrieval_fallback_answer(hits), citations, dt)