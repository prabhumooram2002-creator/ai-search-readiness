"""Section 19 — Recommendation Engine [EXISTS Phase 10a — extend]

Groups Phase 10a's per-recommendation findings into causal chains sharing
the same root entity/chunk (missing entity -> weak evidence on chunks
mentioning it -> trust bottleneck on the page), each ending with the
predicted impact already computed by src/impact.py's what-if overlay —
no new inference, just grouping + linking to section evidence.
"""
from __future__ import annotations

_CHAIN_ORDER = ["missing_entity", "retrieval_dead_end", "weak_evidence",
               "unsupported_answer_sentence", "trust_bottleneck"]


def _chain_key(rec: dict) -> str:
    """Group by whatever identifies the shared root: entity name (from
    finding_id 'entity:<slug>'), else chunk_id, else the finding_id itself."""
    fid = rec.get("finding_id", "")
    if fid.startswith("entity:"):
        return fid
    if rec.get("chunk_id"):
        return f"chunk:{rec['chunk_id']}"
    return fid or rec.get("rule", "unknown")


def build_causal_chains(recommendations: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for rec in recommendations:
        groups.setdefault(_chain_key(rec), []).append(rec)

    chains = []
    for key, recs in groups.items():
        recs_sorted = sorted(
            recs, key=lambda r: _CHAIN_ORDER.index(r["rule"])
            if r.get("rule") in _CHAIN_ORDER else len(_CHAIN_ORDER))
        best_impact = None
        for r in recs_sorted:
            imp = r.get("predicted_impact")
            if imp and imp.get("kind") == "coverage":
                if best_impact is None or (imp.get("delta") or 0) > (best_impact.get("delta") or 0):
                    best_impact = imp
        chains.append({
            "chain_root": key,
            "steps": [{"rule": r.get("rule"), "action": r.get("action"),
                      "finding_id": r.get("finding_id")} for r in recs_sorted],
            "n_steps": len(recs_sorted),
            "predicted_impact": best_impact,
        })

    chains.sort(key=lambda c: -(c["predicted_impact"].get("delta", 0)
                               if c["predicted_impact"] else 0))
    return chains
