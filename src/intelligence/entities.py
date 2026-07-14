"""Section 2 — Complete Entity Graph [DERIVED: new stats over existing entities]

Every stat here is computed from data the pipeline already stores (Entity,
MentionsEntity, RelatesTo edges in the KG) — no new extraction.
"""
from __future__ import annotations

from .graph_stats import page_pagerank


def entity_stats(kg, l0_pages: list[dict]) -> dict:
    """importance = frequency x page_pagerank(sum over pages it appears on)
    x degree(relation count). Every entity links to its mention chunk_ids
    (never printed as a bare id in the report — resolved to url+snippet by
    the renderer)."""
    pr = page_pagerank(l0_pages)

    res = kg._exec(
        "MATCH (e:Entity)<-[m:MentionsEntity]-(c:Chunk)<-[:HasChunk]-(p:Page) "
        "RETURN e.id, e.name, e.type, m.score, c.id, p.url")
    by_entity: dict[str, dict] = {}
    while res.has_next():
        eid, name, etype, score, chunk_id, url = res.get_next()
        e = by_entity.setdefault(eid, {
            "id": eid, "name": name, "type": etype,
            "mentions": [], "pages": set(), "scores": [],
        })
        e["mentions"].append({"chunk_id": chunk_id, "url": url})
        e["pages"].add(url)
        if score is not None:
            e["scores"].append(score)

    degree: dict[str, int] = {}
    res2 = kg._exec("MATCH (a:Entity)-[:RelatesTo]->(b:Entity) RETURN a.id, b.id")
    while res2.has_next():
        a, b = res2.get_next()
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1

    entities = []
    for eid, e in by_entity.items():
        page_weight = sum(pr.get(u, 0.0) for u in e["pages"])
        deg = degree.get(eid, 0)
        importance = round(len(e["mentions"]) * (page_weight or 0.001) * (deg + 1), 6)
        entities.append({
            "id": eid, "name": e["name"], "type": e["type"],
            "frequency": len(e["mentions"]),
            "n_pages": len(e["pages"]),
            "confidence": round(sum(e["scores"]) / len(e["scores"]), 2) if e["scores"] else None,
            "degree": deg,
            "importance": importance,
            "mentions": e["mentions"][:10],  # capped sample; full count is 'frequency'
        })

    entities.sort(key=lambda e: -e["importance"])
    type_counts: dict[str, int] = {}
    for e in entities:
        type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1

    orphans = [e for e in entities if e["frequency"] == 1 and e["degree"] == 0]
    disconnected = [e for e in entities if e["degree"] == 0]

    return {
        "n_entities": len(entities),
        "type_counts": type_counts,
        "top_by_importance": entities[:50],
        "orphan_entities": orphans[:50],
        "disconnected_entities": len(disconnected),
        "all": entities,
    }
