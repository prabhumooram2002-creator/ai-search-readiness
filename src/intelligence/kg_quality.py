"""Section 10 — Knowledge Graph Quality [DERIVED: networkx metrics, cheap]"""
from __future__ import annotations

from rapidfuzz import fuzz

from .graph_stats import entity_relation_graph

COMPLETENESS_FORMULA = ("heuristic composite: 0.3*relation_coverage + "
                       "0.3*(1 - isolated_share) + 0.2*claims_per_entity_norm + "
                       "0.2*(1 - duplicate_share)")


def kg_quality_metrics(kg, entities: list[dict], claims_count: int,
                      dup_name_threshold: float = 90.0) -> dict:
    import networkx as nx

    nodes, edges = entity_relation_graph(kg)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    n = g.number_of_nodes() or 1
    density = round(nx.density(g), 4) if g.number_of_nodes() > 1 else 0.0
    avg_degree = round(sum(dict(g.degree()).values()) / n, 4)
    components = list(nx.connected_components(g))
    isolated = [x for x in nodes if g.degree(x) == 0]
    hubs = sorted(g.degree, key=lambda kv: -kv[1])[:10]

    # duplicate entity candidates: same type, fuzzy name match above threshold
    by_type: dict[str, list[dict]] = {}
    for e in entities:
        by_type.setdefault(e["type"], []).append(e)
    duplicates = []
    for etype, group in by_type.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                s = fuzz.token_set_ratio(group[i]["name"], group[j]["name"])
                if s >= dup_name_threshold:
                    duplicates.append({"a": group[i]["id"], "b": group[j]["id"],
                                      "type": etype, "similarity": round(s, 1)})

    n_with_relation = sum(1 for x in nodes if g.degree(x) > 0)
    relation_coverage = round(n_with_relation / n, 4)
    isolated_share = round(len(isolated) / n, 4)
    claims_per_entity = round(claims_count / n, 4) if n else 0.0
    dup_share = round(len(duplicates) / n, 4) if n else 0.0
    completeness = round(
        0.3 * relation_coverage + 0.3 * (1 - isolated_share)
        + 0.2 * min(1.0, claims_per_entity / 2) + 0.2 * (1 - min(1.0, dup_share)), 4)

    return {
        "density": density, "avg_degree": avg_degree,
        "n_connected_components": len(components),
        "n_isolated_nodes": len(isolated),
        "hub_nodes": [{"entity_id": e, "degree": d} for e, d in hubs],
        "duplicate_entity_candidates": duplicates,
        "claims_per_entity": claims_per_entity,
        "relation_coverage": relation_coverage,
        "completeness": {"value": completeness, "label": "heuristic",
                        "formula": COMPLETENESS_FORMULA},
    }
