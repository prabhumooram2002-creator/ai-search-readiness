"""Section 13 — Query Simulation [EXISTS — print fully]

Full per-query trace dump (not just dead ends/citations like Phase 10's
summary report) plus the P2 parked-irrelevant-queries list.
"""
from __future__ import annotations


def full_query_dossiers(traces: list, skipped_queries: list[dict] | None = None) -> dict:
    dossiers = []
    for t in traces:
        dossiers.append({
            "query": t.query,
            "intent": t.intent,
            "constraints": t.constraints,
            "fanout": [{"sub_query": q, "weight": None} for q in t.expanded_queries],
            "retrieved_top30": t.retrieved[:30],
            "reranked_top5": t.reranked[:5],
            "graph_paths": t.graph_paths,
            "evidence": t.evidence,
            "contradictions": t.contradictions,
            "confidence": t.confidence,
            "confidence_breakdown": t.confidence_breakdown,
            "answer": t.answer,
            "citations": t.citations,
            "unsupported_sentences": t.unsupported_sentences,
            "dead_ends": t.retrieval_dead_ends,
        })
    return {"queries": dossiers, "n_queries": len(dossiers),
           "skipped_irrelevant": skipped_queries or []}
