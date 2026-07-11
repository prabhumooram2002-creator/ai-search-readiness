"""Layer 1 step 5 — Claim Extraction via local LLM (CLAUDE.md).

``llm_complete(json=True, temperature=0)`` extracts discrete, atomic factual
claims from each chunk and tags each with its exact source sentence. Fully
local (Ollama by default), zero keys.

- Sentences come from pysbd with char spans (step-1 discipline: offsets are
  preserved so every claim traces to an exact location in the page).
- Output: ``claim{id, text, source_sentence_id, source_chunk_id}`` plus the
  sentence's char offsets and text for downstream evidence extraction (step 6).
- Verification rule (enforced here, per CLAUDE.md): every claim maps to a real
  sentence — claims with invalid/missing sentence references are dropped, with
  the reason recorded on the StepTrace at drop time.
- Trace: emits one StepTrace named ``claim_extraction`` when a Trace is passed,
  including on error.
"""
from __future__ import annotations

import hashlib
import json as _json
from typing import Optional

from . import providers
from .core.logging import get_logger

logger = get_logger(__name__)

MAX_SENTENCES_PER_CALL = 40   # keep prompts small for a 7B local model
MAX_CHARS_PER_SENTENCE = 500

_segmenter = None

CLAIM_PROMPT = """You extract factual claims from website content.

Below are numbered sentences from one content chunk. Extract every DISCRETE,
ATOMIC factual claim they state. Rules:
- One claim = one verifiable fact (split compound statements).
- Copy facts faithfully; do not add outside knowledge; do not infer.
- Tag each claim with the number of the ONE sentence it comes from.
- Skip questions, navigation text, and opinions.
- If there are no factual claims, return an empty list.

Sentences:
{sentences}

Return ONLY valid JSON, exactly this shape:
{{"claims": [{{"claim": "<atomic factual claim>", "sentence": <sentence number>}}]}}"""


def _get_segmenter():
    global _segmenter
    if _segmenter is None:
        import pysbd  # lazy
        _segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)
    return _segmenter


def segment_sentences(text: str, chunk_id: str = "") -> list[dict]:
    """pysbd sentence records with char offsets preserved (step-1 discipline).

    Returns [{"sentence_id", "number", "text", "char_start", "char_end"}]
    (number is 1-based — it's what the LLM sees and references).
    """
    spans = _get_segmenter().segment(text)
    out = []
    for i, sp in enumerate(spans, 1):
        stripped = sp.sent.strip()
        if not stripped:
            continue
        out.append({
            "sentence_id": f"{chunk_id}:s{i}" if chunk_id else f"s{i}",
            "number": i,
            "text": stripped,
            "char_start": int(sp.start),
            "char_end": int(sp.end),
        })
    return out


def _claim_id(chunk_id: str, sentence_number: int, text: str) -> str:
    h = hashlib.sha256(f"{chunk_id}|{sentence_number}|{text}".encode()).hexdigest()[:16]
    return f"claim_{h}"


def _parse_claims_json(raw: str) -> list[dict]:
    """Parse the LLM's JSON, tolerating fences/prose around it."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = _json.loads(text)
    except _json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if 0 <= start < end:
            try:
                data = _json.loads(text[start:end])
            except _json.JSONDecodeError:
                logger.warning(f"Claim JSON unparseable: {text[:200]}")
                return []
        else:
            logger.warning(f"Claim JSON unparseable: {text[:200]}")
            return []
    claims = data.get("claims", []) if isinstance(data, dict) else []
    return claims if isinstance(claims, list) else []


def extract_claims(chunks: list[dict], trace=None) -> list[dict]:
    """Extract atomic factual claims per chunk via the local LLM.

    chunks: [{"content": str, "chunk_id": str}, ...]
    Returns: [{"id", "text", "source_sentence_id", "source_chunk_id",
               "sentence_text", "char_start", "char_end"}]
    """
    step_cm = trace.start_step(
        "claim_extraction", n_chunks=len(chunks),
        model=f"{providers.LLM_PROVIDER}:{providers.LLM_MODEL}",
    ) if trace is not None else None
    st = step_cm.__enter__() if step_cm is not None else None

    try:
        all_claims: list[dict] = []
        for c in chunks:
            chunk_id = str(c.get("chunk_id", ""))
            sentences = segment_sentences(c["content"], chunk_id)[:MAX_SENTENCES_PER_CALL]
            if not sentences:
                continue
            by_number = {s["number"]: s for s in sentences}

            numbered = "\n".join(
                f"{s['number']}. {s['text'][:MAX_CHARS_PER_SENTENCE]}" for s in sentences
            )
            raw = providers.llm_complete(
                CLAIM_PROMPT.format(sentences=numbered),
                json=True, temperature=0,
                system="You are a precise claim extraction system. Output ONLY valid JSON.",
            )

            for item in _parse_claims_json(raw):
                text = str(item.get("claim", "")).strip()
                num = item.get("sentence")
                if not text:
                    continue
                # CLAUDE.md verification: claim must map to a REAL sentence.
                if not isinstance(num, int) or num not in by_number:
                    if st is not None:
                        st.drop({"claim": text, "sentence": num, "chunk_id": chunk_id},
                                "sentence reference not a real step-1 sentence")
                    continue
                s = by_number[num]
                all_claims.append({
                    "id": _claim_id(chunk_id, num, text),
                    "text": text,
                    "source_sentence_id": s["sentence_id"],
                    "source_chunk_id": chunk_id,
                    "sentence_text": s["text"],
                    "char_start": s["char_start"],
                    "char_end": s["char_end"],
                })

        logger.info(f"Claims: {len(all_claims)} atomic claims from {len(chunks)} chunks")
        if st is not None:
            st.outputs["claims"] = all_claims
            st.scores["n_claims"] = float(len(all_claims))  # heuristic
        return all_claims
    except BaseException as exc:
        if step_cm is not None:
            step_cm.__exit__(type(exc), exc, exc.__traceback__)
            step_cm = None
        raise
    finally:
        if step_cm is not None:
            step_cm.__exit__(None, None, None)
