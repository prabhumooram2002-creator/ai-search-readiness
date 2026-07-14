"""Section 16 — Content Intelligence [DERIVED, mostly local]"""
from __future__ import annotations

import re

import textstat


def _claims_per_100_words(text: str, n_claims: int) -> float:
    words = len(text.split())
    return round(n_claims * 100 / words, 2) if words else 0.0


def page_content_intel(
    page_url: str, page_text: str, n_claims: int, n_entities: int,
    structure_score: float | None, unsupported_rate: float | None,
    sub_intents_covered: int, sub_intents_total: int, freshness: float | None,
) -> dict:
    """readability (textstat), factual/entity density, answer completeness
    (this page's covered sub-intents / topic total), citation potential
    (structure x evidence-strength-proxy x freshness — heuristic)."""
    try:
        flesch = round(textstat.flesch_reading_ease(page_text), 2) if page_text.strip() else None
    except Exception:
        flesch = None

    completeness = (round(sub_intents_covered / sub_intents_total, 4)
                   if sub_intents_total else None)
    evidence_strength = round(1.0 - (unsupported_rate or 0.0), 4)
    citation_potential = round(
        (structure_score or 0.0) * evidence_strength * (freshness if freshness is not None else 0.5),
        4)

    return {
        "url": page_url,
        "readability_flesch": flesch,
        "factual_density_per_100w": _claims_per_100_words(page_text, n_claims),
        "entity_density_per_100w": _claims_per_100_words(page_text, n_entities),
        "answer_completeness": completeness,
        "citation_potential": {"value": citation_potential, "label": "heuristic",
                              "formula": "structure_score * (1 - unsupported_rate) * freshness"},
        "hallucination_risk_surface": round(unsupported_rate, 4) if unsupported_rate is not None else None,
    }


def near_duplicate_chunks(chunks: list[dict], embeddings: list[list[float]],
                          threshold: float = 0.95) -> list[dict]:
    """Cheap O(n^2) near-dup check over chunk embeddings — fine at this
    pipeline's chunk-count scale (hundreds, not millions)."""
    from .core_utils import cosine
    pairs = []
    n = len(chunks)
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine(embeddings[i], embeddings[j])
            if sim >= threshold:
                pairs.append({"a": chunks[i]["chunk_id"], "b": chunks[j]["chunk_id"],
                             "similarity": round(sim, 4)})
    return pairs
