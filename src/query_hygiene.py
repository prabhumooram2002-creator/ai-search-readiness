"""PHASE 12 P2 — Query hygiene (INTELLIGENCE_REPORT_BRIEF.md).

Two cheap, local steps before a query set feeds Section 13 (Query
Simulation): dedupe near-identical queries, then park off-brand/irrelevant
queries (ones with no real topical connection to the site) in a "skipped
(irrelevant)" list rather than silently simulating nonsense answers for them.

Relevance is scored against topic centroids computed directly from the KG's
own Chunk embeddings + BelongsToTopic edges — no new embedding index, no
new model, just an aggregate over data the pipeline already stores.
"""
from __future__ import annotations

import math
from typing import Optional

from .core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_RELEVANCE_FLOOR = 0.3


def dedupe_queries(queries: list[str]) -> list[str]:
    """Case/whitespace-insensitive dedupe, first-seen order preserved."""
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        key = " ".join(q.strip().lower().split())
        if key and key not in seen:
            seen.add(key)
            out.append(q.strip())
    return out


def topic_centroids(kg) -> dict[str, list[float]]:
    """Mean embedding per topic, aggregated from its member chunks —
    Chunk.embedding + BelongsToTopic already exist in the KG, no new store."""
    res = kg._exec(
        "MATCH (c:Chunk)-[:BelongsToTopic]->(t:Topic) RETURN t.id, c.embedding")
    sums: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    while res.has_next():
        tid, emb = res.get_next()
        if not emb:
            continue
        if tid not in sums:
            sums[tid] = [0.0] * len(emb)
            counts[tid] = 0
        for i, v in enumerate(emb):
            sums[tid][i] += v
        counts[tid] += 1
    return {tid: [v / counts[tid] for v in vec]
            for tid, vec in sums.items() if counts.get(tid)}


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def score_query_relevance(
    queries: list[str], centroids: dict[str, list[float]], embed_fn,
    threshold: float = DEFAULT_RELEVANCE_FLOOR,
) -> dict:
    """Score every query's max cosine similarity against any topic centroid.
    Below-threshold queries are parked, not dropped — Section 13 still lists
    them, labeled 'skipped (irrelevant)', so nothing silently disappears.

    Returns {"relevant": [str, ...], "skipped": [{"query", "max_similarity"}]}.
    """
    if not queries:
        return {"relevant": [], "skipped": []}
    if not centroids:
        # No topics to score against (e.g. too few chunks) — don't fabricate
        # a relevance judgement with nothing to judge it against.
        return {"relevant": list(queries), "skipped": []}

    vecs = embed_fn([q[:512] for q in queries])
    relevant: list[str] = []
    skipped: list[dict] = []
    for q, v in zip(queries, vecs):
        best = max((_cosine(v, c) for c in centroids.values()), default=0.0)
        if best >= threshold:
            relevant.append(q)
        else:
            skipped.append({"query": q, "max_similarity": round(best, 4)})
    logger.info(f"[query_hygiene] {len(relevant)} relevant, {len(skipped)} "
               f"parked as off-brand/irrelevant (floor={threshold})")
    return {"relevant": relevant, "skipped": skipped}


def clean_query_set(
    queries: list[str], kg=None, embed_fn=None,
    threshold: float = DEFAULT_RELEVANCE_FLOOR,
) -> dict:
    """Full P2 pipeline: dedupe, then (if a KG + embed function are given)
    park off-brand queries. Returns {"relevant", "skipped", "n_duplicates_removed"}."""
    deduped = dedupe_queries(queries)
    n_dupes = len(queries) - len(deduped)
    if kg is None or embed_fn is None:
        return {"relevant": deduped, "skipped": [], "n_duplicates_removed": n_dupes}
    centroids = topic_centroids(kg)
    scored = score_query_relevance(deduped, centroids, embed_fn, threshold)
    scored["n_duplicates_removed"] = n_dupes
    return scored
