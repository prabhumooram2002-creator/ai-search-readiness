"""Section 4 — Semantic Topics [EXISTS + derived]"""
from __future__ import annotations

from .core_utils import cosine


def topic_stats(kg, page_signals: dict, fanout_by_query: list[dict] | None = None,
                dead_end_floor: float = 0.35) -> dict:
    """Primary/secondary/supporting by share of chunks; depth (chunks +
    distinct entities); topical gaps = fan-out sub-intents whose max cosine
    to any topic centroid is below floor (heuristic, floor logged)."""
    from ..query_hygiene import topic_centroids

    res = kg._exec(
        "MATCH (t:Topic)<-[:BelongsToTopic]-(c:Chunk)<-[:HasChunk]-(p:Page) "
        "RETURN t.id, t.label, c.id, p.url")
    by_topic: dict[str, dict] = {}
    total_chunks = 0
    while res.has_next():
        tid, label, cid, url = res.get_next()
        t = by_topic.setdefault(tid, {"id": tid, "label": label,
                                      "chunk_ids": set(), "pages": set()})
        t["chunk_ids"].add(cid)
        t["pages"].add(url)
        total_chunks += 1

    res2 = kg._exec(
        "MATCH (t:Topic)<-[:BelongsToTopic]-(c:Chunk)-[:MentionsEntity]->(e:Entity) "
        "RETURN t.id, e.id")
    entities_by_topic: dict[str, set] = {}
    while res2.has_next():
        tid, eid = res2.get_next()
        entities_by_topic.setdefault(tid, set()).add(eid)

    topics = []
    for tid, t in by_topic.items():
        share = round(len(t["chunk_ids"]) / total_chunks, 4) if total_chunks else 0.0
        auth_vals = [page_signals.get(u, {}).get("source_authority", 0.0) for u in t["pages"]]
        topics.append({
            "id": tid, "label": t["label"],
            "n_chunks": len(t["chunk_ids"]), "n_pages": len(t["pages"]),
            "share_of_chunks": share,
            "role": "primary" if share >= 0.25 else ("secondary" if share >= 0.1 else "supporting"),
            "depth": len(t["chunk_ids"]) + len(entities_by_topic.get(tid, set())),
            "topic_authority": round(sum(auth_vals) / len(auth_vals), 4) if auth_vals else None,
        })
    topics.sort(key=lambda t: -t["share_of_chunks"])

    gaps = []
    if fanout_by_query:
        centroids = topic_centroids(kg)
        for entry in fanout_by_query:
            for sub in entry.get("fanout", []):
                qv = sub.get("embedding")
                if qv is None or not centroids:
                    continue
                best = max((cosine(qv, c) for c in centroids.values()), default=0.0)
                if best < dead_end_floor:
                    gaps.append({"sub_query": sub["sub_query"], "weight": sub.get("weight"),
                                "max_topic_similarity": round(best, 4)})

    return {"topics": topics, "n_topics": len(topics), "topical_gaps": gaps,
           "topical_gap_floor": dead_end_floor}
