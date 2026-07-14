"""Section 3 — Relationship Graph [EXISTS after P1]

All subject-relation-object triplets with source chunk locations and
support_count, plus derived quality findings and a dependency-free inline
SVG graph of the top entities by importance (no external CDN).
"""
from __future__ import annotations

import math

from .graph_stats import entity_relation_graph


def relationship_triplets(kg) -> list[dict]:
    res = kg._exec(
        "MATCH (a:Entity)-[r:RelatesTo]->(b:Entity) "
        "RETURN a.id, a.name, r.relation, b.id, b.name, r.support_count, r.score")
    out = []
    while res.has_next():
        aid, aname, rel, bid, bname, support, score = res.get_next()
        out.append({
            "subject_id": aid, "subject": aname, "relation": rel,
            "object_id": bid, "object": bname,
            "support_count": support, "score": round(score, 2) if score is not None else None,
        })
    return out


def relationship_findings(triplets: list[dict], weak_support_floor: int = 1,
                          weak_score_floor: float = 0.6) -> dict:
    """Weak edges (support_count=1 and low confidence). Missing/unsupported-
    claim-edge checks need fan-out sub-intents / claim-edge data this
    pipeline doesn't yet join at the KG level — printed as excluded rather
    than guessed (SupportsClaim is Chunk->Claim, not Entity->Entity, so
    there's no direct 'claim-edge' to check without a new join the brief
    doesn't specify the shape of)."""
    weak = [t for t in triplets
           if (t.get("support_count") or 0) <= weak_support_floor
           and (t.get("score") or 1.0) < weak_score_floor]
    return {
        "weak_edges": weak,
        "n_weak": len(weak),
        "missing_edges": {"available": False,
                          "reason": "needs fan-out sub-intent -> entity-pair mapping, "
                                   "not computed by this pipeline"},
    }


def render_top_entities_svg(kg, top_n: int = 50, width: int = 760, height: int = 560) -> str:
    """Inline SVG force-ish layout (circular, deterministic — no physics sim,
    no external JS) of the top-N entities by relation degree."""
    nodes, edges = entity_relation_graph(kg)
    degree: dict[str, int] = {n: 0 for n in nodes}
    for a, b in edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    top = sorted(nodes, key=lambda n: -degree.get(n, 0))[:top_n]
    top_set = set(top)
    cx, cy, r = width / 2, height / 2, min(width, height) / 2 - 40
    pos = {}
    for i, n in enumerate(top):
        angle = 2 * math.pi * i / max(1, len(top))
        pos[n] = (cx + r * math.cos(angle), cy + r * math.sin(angle))

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
            f'font-family="sans-serif" font-size="10">']
    for a, b in edges:
        if a in top_set and b in top_set:
            x1, y1 = pos[a]
            x2, y2 = pos[b]
            parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                        f'stroke="#ccc" stroke-width="1"/>')
    names = {}
    if top:
        res = kg._exec("MATCH (e:Entity) WHERE e.id IN $ids RETURN e.id, e.name",
                       {"ids": top})
        while res.has_next():
            eid, name = res.get_next()
            names[eid] = name
    for n in top:
        x, y = pos[n]
        rad = 3 + min(8, degree.get(n, 0))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad}" fill="#a8480f" '
                    f'opacity="0.8"><title>{names.get(n, n)} (degree {degree.get(n, 0)})'
                    f'</title></circle>')
    parts.append("</svg>")
    return "\n".join(parts)
