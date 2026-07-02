"""Hybrid retrieval + broken path detection — Phase 3."""
from dataclasses import dataclass, field
from typing import Optional
from ..graph.kuzu_graph import get_graph
from ..vector.store import VectorStore, SearchResult

from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PathResult:
    source: str
    target: str
    full_path: list[str] = field(default_factory=list)
    hops: int = 0
    break_point: Optional[str] = None
    missing_rel: Optional[str] = None
    suggested_fix: Optional[str] = None
    entities_found: list[Optional[str]] = field(default_factory=list)


def broken_path_detect(
    question: str,
    entity_chain: list[str],
    graph=None,
) -> PathResult:
    """
    Implements the broken path detection algorithm from Section 5.
    
    Steps:
    1. Parse expected entity chain (provided as input)
    2. For each link, check if edge exists in graph
    3. First missing edge = break point
    4. Return path or break details + suggested fix
    """
    if graph is None:
        graph = get_graph()

    if len(entity_chain) < 2:
        return PathResult(
            source=entity_chain[0] if entity_chain else "",
            target=entity_chain[-1] if entity_chain else "",
            full_path=entity_chain,
            hops=0,
        )

    # For each adjacent pair, check if edge exists
    for i in range(len(entity_chain) - 1):
        src, tgt = entity_chain[i], entity_chain[i + 1]

        # Check path in graph (sync call)
        paths = graph.find_paths(src, tgt, max_hops=2)

        if not paths:
            # No path found — this is the break point
            # Check if both entities exist independently
            src_related = graph.query_related(src)
            tgt_related = graph.query_related(tgt)

            src_found = bool(src_related is not None)
            tgt_found = bool(tgt_related is not None)

            return PathResult(
                source=src,
                target=tgt,
                full_path=entity_chain[:i + 1],
                hops=i,
                break_point=f"{src} → {tgt}",
                missing_rel=f"No edge found between '{src}' and '{tgt}'",
                suggested_fix=(
                    f"Add content to explicitly connect '{src}' to '{tgt}'. "
                    f"Consider a sentence like: '[{src}] {get_suggested_rel(src, tgt)} [{tgt}].'"
                ),
                entities_found=[src if src_found else None, tgt if tgt_found else None],
            )

    # Full path found
    return PathResult(
        source=entity_chain[0],
        target=entity_chain[-1],
        full_path=entity_chain,
        hops=len(entity_chain) - 1,
    )


def get_suggested_rel(src: str, tgt: str) -> str:
    """Suggest a relationship type based on entity names."""
    return "is related to"


def hybrid_retrieve(
    question_embedding: list[float],
    graph=None,
    vector_store: Optional[VectorStore] = None,
    top_k: int = 5,
) -> list:
    """
    Phase 3: Combine vector search + graph traversal.
    Returns ranked results with both vector similarity and graph context.
    """
    if graph is None:
        graph = get_graph()
    if vector_store is None:
        return []

    # Get vector search results
    vector_results = vector_store.search(question_embedding, top_k=top_k * 2)

    # Get entity names from vector results for graph lookup
    entity_names = [r.title for r in vector_results[:3] if r.title]

    # Try to find graph paths between found entities
    graph_paths = []
    for i, name1 in enumerate(entity_names):
        for name2 in entity_names[i + 1:]:
            try:
                paths = graph.find_paths(name1, name2, max_hops=3)
                graph_paths.extend(paths)
            except Exception:
                pass

    # Merge and rerank (simple: vector score + path bonus)
    results = []
    for r in vector_results:
        path_bonus = 0.0
        for p in graph_paths:
            if r.title in p.get("path", ""):
                path_bonus = 0.1
                break
        adjusted_score = r.distance + path_bonus
        results.append((adjusted_score, r))

    results.sort(key=lambda x: x[0])
    return [r for _, r in results[:top_k]]