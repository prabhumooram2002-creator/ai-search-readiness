"""Section 18 — Citation Readiness [REPLACES per-engine probabilities]

Printing "ChatGPT probability 0.73 / Gemini 0.68" without engine access is
fabricated precision. Three separate columns instead, three separate
labels, NEVER merged into one number.
"""
from __future__ import annotations

CITATION_READINESS_FORMULA = (
    "heuristic composite: structure_score * evidence_strength * "
    "crawlability * freshness * internal_authority")


def citation_readiness_row(
    url: str, structure_score: float | None, evidence_strength: float | None,
    invisible_ratio: float | None, freshness: float | None, authority: float | None,
    simulated_retrieval_success: bool | None,
    observed_citations: dict[str, int] | None = None,
) -> dict:
    crawlability = 1.0 - (invisible_ratio if invisible_ratio is not None else 0.5)
    parts = [structure_score, evidence_strength, crawlability, freshness, authority]
    have = [p for p in parts if p is not None]
    readiness = round(
        (structure_score or 0) * (evidence_strength or 0) * crawlability
        * (freshness or 0) * (authority or 0), 4) if len(have) == len(parts) else None

    return {
        "url": url,
        "citation_readiness_score": {"value": readiness, "label": "heuristic",
                                    "formula": CITATION_READINESS_FORMULA},
        "simulated_retrieval_success": {"value": simulated_retrieval_success, "label": "simulated"},
        "observed_citations": {"value": observed_citations or {}, "label": "observed",
                              "note": "blank until a calibration panel is run"
                                     if not observed_citations else None},
    }


def citation_readiness_table(pages_data: list[dict]) -> list[dict]:
    """pages_data: [{"url","structure_score","evidence_strength",
    "invisible_ratio","freshness","authority","simulated_retrieval_success",
    "observed_citations"?}]."""
    return [citation_readiness_row(**p) for p in pages_data]
