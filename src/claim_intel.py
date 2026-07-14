"""Sections 11/12 — Claim Intelligence + Evidence Graph [NEW aggregation]
(INTELLIGENCE_REPORT_BRIEF.md).

The AI trust report: for every claim, cross-page evidence SITE-WIDE (embed
match + NLI, reusing the exact models Layer 1 step 6 already loads) — not
just the local +/-2-sentence window src/evidence.py checks at index time.
"""
from __future__ import annotations

import re

from .intelligence.core_utils import cosine

_SUPERLATIVE_RE = re.compile(
    r"\b(best|#1|no\.?\s?1|number one|leading|top|largest|biggest|most trusted|"
    r"most popular|india'?s\s+no)\b", re.IGNORECASE)

TOP_K_CROSS_PAGE = 5


def is_superlative_claim(text: str) -> bool:
    return bool(_SUPERLATIVE_RE.search(text))


def cross_page_evidence(
    claims: list[dict], chunks_by_id: dict[str, dict],
    claim_embeddings: dict[str, list[float]], chunk_embeddings: dict[str, list[float]],
    score_fn, top_k: int = TOP_K_CROSS_PAGE,
) -> list[dict]:
    """claims: [{"id","text","source_chunk_id"}]. score_fn(pairs) -> [(label,
    conf), ...], same NLI signature as src.evidence._score_pairs — reused,
    not reimplemented.

    For each claim: rank ALL chunks (site-wide) by embedding similarity,
    take the top-k excluding the claim's own source chunk, run NLI against
    each, and split into supporting / contradicting / neutral.
    """
    out = []
    for claim in claims:
        cid = claim["id"]
        cvec = claim_embeddings.get(cid)
        if cvec is None:
            out.append({"claim_id": cid, "text": claim["text"],
                       "is_superlative": is_superlative_claim(claim["text"]),
                       "supporting": [], "contradicting": [], "evidence_count": 0,
                       "confidence": None, "note": "no embedding available"})
            continue

        ranked = sorted(
            ((chid, cosine(cvec, cvec2)) for chid, cvec2 in chunk_embeddings.items()
             if chid != claim.get("source_chunk_id")),
            key=lambda kv: -kv[1])[:top_k]

        pairs = [(chunks_by_id[chid]["content"], claim["text"])
                for chid, _ in ranked if chid in chunks_by_id]
        labels = score_fn(pairs) if pairs else []

        supporting, contradicting = [], []
        for (chid, sim), (label, conf) in zip(ranked, labels):
            chunk = chunks_by_id.get(chid, {})
            entry = {"chunk_id": chid, "url": chunk.get("url", ""),
                    "similarity": round(sim, 4), "nli_label": label,
                    "nli_conf": round(conf, 4)}
            if label == "entailment":
                supporting.append(entry)
            elif label == "contradiction":
                contradicting.append(entry)

        evidence_count = len(supporting) + len(contradicting)
        mean_conf = (sum(e["nli_conf"] for e in supporting) / len(supporting)
                    if supporting else 0.0)
        confidence = round(evidence_count and mean_conf * min(1.0, evidence_count / top_k), 4)

        out.append({
            "claim_id": cid, "text": claim["text"],
            "is_superlative": is_superlative_claim(claim["text"]),
            "supporting": supporting, "contradicting": contradicting,
            "evidence_count": evidence_count,
            "confidence": {"value": confidence, "label": "heuristic"},
        })
    return out


def rank_claims(claim_records: list[dict]) -> dict:
    """Strongest-evidenced claims; naked claims (superlative with zero
    support); contradicted claims."""
    strongest = sorted(
        [c for c in claim_records if c["evidence_count"] > 0],
        key=lambda c: -(c["confidence"]["value"] if c.get("confidence") else 0))[:20]
    naked = [c for c in claim_records if c["is_superlative"] and c["evidence_count"] == 0]
    contradicted = [c for c in claim_records if c["contradicting"]]
    return {"strongest_evidenced": strongest, "naked_superlative_claims": naked,
           "contradicted_claims": contradicted}
