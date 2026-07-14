"""Section 5 — Semantic Chunks [EXISTS]

Per-chunk deep-dive: page URL + heading path + char range, text, entities,
claims, evidence, structure flags/score, freshness, parent topic, and which
queries it won/lost in retrieval (from Layer 2 traces).
"""
from __future__ import annotations


def chunk_dossiers(chunks: list[dict], entities: list[dict], claims_by_chunk: dict,
                   evidence_by_chunk: dict, struct_by_chunk: dict,
                   topic_by_chunk: dict, traces: list) -> list[dict]:
    """chunks: [{"chunk_id","content","url","heading_path"?,"char_start"?,
    "char_end"?}]. entities: section-2-style records with .mentions[chunk_id]."""
    entities_by_chunk: dict[str, list[str]] = {}
    for e in entities:
        for m in e.get("mentions", []):
            entities_by_chunk.setdefault(m["chunk_id"], []).append(e["name"])

    won: dict[str, list[str]] = {}
    lost: dict[str, list[str]] = {}
    for t in traces:
        winners = {r["chunk_id"] for r in t.reranked}
        for r in t.retrieved:
            (won if r["chunk_id"] in winners else lost).setdefault(
                r["chunk_id"], []).append(t.query)

    out = []
    for c in chunks:
        cid = c["chunk_id"]
        struct = struct_by_chunk.get(cid, {})
        out.append({
            "chunk_id": cid, "url": c.get("url", ""),
            "heading_path": c.get("heading_path", []),
            "char_start": c.get("char_start"), "char_end": c.get("char_end"),
            "text": c.get("content", ""),
            "entities": entities_by_chunk.get(cid, []),
            "claims": claims_by_chunk.get(cid, []),
            "evidence": evidence_by_chunk.get(cid, []),
            "structure_score": struct.get("structure_score"),
            "structure_flags": [f["flag"] for f in struct.get("flags", [])],
            "topic": topic_by_chunk.get(cid),
            "queries_won": won.get(cid, []),
            "queries_lost": lost.get(cid, []),
        })
    return out
