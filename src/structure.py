"""Phase 3b — Structural scorer (GEO-SFE-style, deterministic, no ML).

Research basis: content STRUCTURE predicts AI citation independently of content
(one claim per block, tables over inline numeric series, schema-tagged pages
cited more). This node scores each chunk on structure alone — pure rules, no
model — and feeds concrete flags with exact locations into Layer-3
recommendations.

Per-chunk ``structure_score`` in [0,1] (labeled heuristic) + a flag list.
Emits a StepTrace named ``structural_scoring`` when a Trace is passed.
"""
from __future__ import annotations

import re
from typing import Optional

from .core.logging import get_logger

logger = get_logger(__name__)

MAX_CLAIMS_PER_BLOCK = 2       # >2 sentence-claims in a paragraph -> flag
MAX_PARAGRAPH_WORDS = 120
MIN_NUMERIC_SERIES = 3         # >=3 numbers in prose -> should be a table
INTERROGATIVES = ("what", "how", "why", "when", "where", "who", "which",
                  "is", "are", "can", "does", "do", "should")

# weights sum to 1.0 (schema is page-level, folded in via has_schema)
_W = {"claims": 0.30, "length": 0.25, "tabular": 0.20,
      "heading": 0.15, "schema": 0.10}

_SENT = re.compile(r"[.!?]+(?:\s|$)")
_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?")


def _claims_per_block(text: str) -> int:
    """Rough claim count = declarative sentences in the chunk's largest block."""
    blocks = [b for b in re.split(r"\n{2,}", text) if b.strip()] or [text]
    return max(len([s for s in _SENT.split(b) if len(s.split()) >= 3])
               for b in blocks)


def _max_paragraph_words(text: str) -> int:
    blocks = [b for b in re.split(r"\n{2,}", text) if b.strip()] or [text]
    return max(len(b.split()) for b in blocks)


def _numeric_series(text: str) -> int:
    """Count numbers that sit in prose (not already inside a markdown table)."""
    prose = "\n".join(l for l in text.splitlines() if "|" not in l)
    return len(_NUM.findall(prose))


def _heading_question_aligned(heading_path: list[str], text: str) -> Optional[bool]:
    """Does the section heading read like a question the content answers?
    None when there's no heading to judge."""
    if not heading_path:
        return None
    h = heading_path[-1].lower().strip()
    is_question = h.endswith("?") or h.split()[:1] and h.split()[0] in INTERROGATIVES
    return bool(is_question)


def score_chunk(chunk: dict, has_schema: bool = False) -> dict:
    """chunk: {"id", "text", "heading_path"?}. Returns structure_score + flags."""
    text = chunk.get("text", "")
    heading_path = chunk.get("heading_path", []) or []
    flags: list[dict] = []
    sub: dict[str, float] = {}

    cpb = _claims_per_block(text)
    sub["claims"] = 1.0 if cpb <= 1 else (0.5 if cpb <= MAX_CLAIMS_PER_BLOCK else 0.0)
    if cpb > MAX_CLAIMS_PER_BLOCK:
        flags.append({"flag": "too_many_claims_per_block", "value": cpb,
                      "chunk_id": chunk.get("id"),
                      "fix": f"Split this {cpb}-claim block into single-claim "
                             "paragraphs (one verifiable fact each)."})

    words = _max_paragraph_words(text)
    sub["length"] = 1.0 if words <= MAX_PARAGRAPH_WORDS else \
        max(0.0, 1.0 - (words - MAX_PARAGRAPH_WORDS) / MAX_PARAGRAPH_WORDS)
    if words > MAX_PARAGRAPH_WORDS:
        flags.append({"flag": "paragraph_too_long", "value": words,
                      "chunk_id": chunk.get("id"),
                      "fix": f"Break the {words}-word paragraph into shorter "
                             "blocks so each answers one thing."})

    nums = _numeric_series(text)
    sub["tabular"] = 1.0 if nums < MIN_NUMERIC_SERIES else \
        max(0.0, 1.0 - (nums - MIN_NUMERIC_SERIES + 1) / 6.0)
    if nums >= MIN_NUMERIC_SERIES:
        flags.append({"flag": "numeric_series_should_be_table", "value": nums,
                      "chunk_id": chunk.get("id"),
                      "fix": f"Move the {nums} inline numbers into a table — "
                             "AI engines cite tabular data more readily."})

    aligned = _heading_question_aligned(heading_path, text)
    if aligned is None:
        sub["heading"] = 0.5
        flags.append({"flag": "no_heading", "chunk_id": chunk.get("id"),
                      "fix": "Add a descriptive (ideally question-form) heading "
                             "for this section."})
    else:
        sub["heading"] = 1.0 if aligned else 0.6

    sub["schema"] = 1.0 if has_schema else 0.0
    if not has_schema:
        flags.append({"flag": "no_schema_markup", "chunk_id": chunk.get("id"),
                      "fix": "Add relevant JSON-LD schema to this page — "
                             "schema-tagged pages are cited materially more."})

    score = round(sum(_W[k] * sub[k] for k in _W), 4)
    return {"chunk_id": chunk.get("id"), "structure_score": score,  # heuristic
            "subscores": sub, "flags": flags}


def score_chunks(chunks: list[dict], pages_with_schema: Optional[set] = None,
                 trace=None) -> list[dict]:
    """Score every chunk. ``pages_with_schema``: urls that carry valid JSON-LD."""
    pages_with_schema = pages_with_schema or set()
    step_cm = trace.start_step("structural_scoring",
                               n_chunks=len(chunks)) if trace else None
    st = step_cm.__enter__() if step_cm else None
    try:
        results = [score_chunk(c, has_schema=c.get("page_url") in pages_with_schema)
                   for c in chunks]
        mean = round(sum(r["structure_score"] for r in results) / len(results), 4) \
            if results else 0.0
        n_flags = sum(len(r["flags"]) for r in results)
        logger.info(f"Structural scoring (heuristic): {len(results)} chunks, "
                    f"mean={mean}, {n_flags} flags")
        if st is not None:
            st.outputs["mean_structure_score"] = mean
            st.outputs["n_flags"] = n_flags
            st.scores["mean_structure_score"] = mean   # heuristic
        return results
    except BaseException as exc:
        if step_cm is not None:
            step_cm.__exit__(type(exc), exc, exc.__traceback__)
            step_cm = None
        raise
    finally:
        if step_cm is not None:
            step_cm.__exit__(None, None, None)
