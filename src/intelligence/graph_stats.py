"""Shared graph statistics for sections 2 (Entity Graph) and 10 (KG Quality).
Reuses the same networkx-over-internal-links PageRank pattern already
proven in src/layer0.py's AI Invisibility Score — no new dependency.
"""
from __future__ import annotations

from .core_utils import cosine


def page_pagerank(l0_pages: list[dict]) -> dict[str, float]:
    """Internal PageRank over the InternalLink graph (same pattern as
    layer0.invisibility_score, factored out so sections 2/10/7 can share it
    without recomputing or importing Layer 0's heavier module)."""
    import networkx as nx
    g = nx.DiGraph()
    for p in l0_pages:
        g.add_node(p["url"])
        for link in p.get("internal_links", []):
            dest = link["dest_url"] if isinstance(link, dict) else link
            if any(dest == q["url"] for q in l0_pages):
                g.add_edge(p["url"], dest)
    return nx.pagerank(g) if g.number_of_nodes() else {}


def orphan_pages(l0_pages: list[dict]) -> list[str]:
    """Pages with zero inbound internal links."""
    inbound: dict[str, int] = {}
    for p in l0_pages:
        for link in p.get("internal_links", []):
            dest = link["dest_url"] if isinstance(link, dict) else link
            inbound[dest] = inbound.get(dest, 0) + 1
    return [p["url"] for p in l0_pages if inbound.get(p["url"], 0) == 0]


def entity_relation_graph(kg) -> tuple[list[str], list[tuple[str, str]]]:
    """Entity ids + (subject_id, object_id) edges from RelatesTo, for
    networkx-based KG quality metrics (section 10) and the top-50 SVG
    (section 3)."""
    nodes: set[str] = set()
    edges: list[tuple[str, str]] = []
    res = kg._exec("MATCH (a:Entity)-[:RelatesTo]->(b:Entity) RETURN a.id, b.id")
    while res.has_next():
        a, b = res.get_next()
        nodes.add(a)
        nodes.add(b)
        edges.append((a, b))
    # entities with zero relations still need to appear as isolated nodes
    res2 = kg._exec("MATCH (e:Entity) RETURN e.id")
    while res2.has_next():
        nodes.add(res2.get_next()[0])
    return sorted(nodes), edges
