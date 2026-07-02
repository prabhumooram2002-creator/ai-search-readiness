"""
Semantic Connectivity Gate — Stage 0 of the v4 pipeline.

Determines whether an input query has any meaningful connection to the
crawled knowledge base before attempting chain extraction / root cause
analysis. Outputs a connectivity_score (0-1) and a tier classification.

Tier 1 (>=0.55): Connected — proceed to full v3 pipeline
Tier 2 (0.30-0.55): Weakly Connected — proceed but flag as tentative
Tier 3 (<0.30): Disconnected — skip to Disconnection Report
"""
from typing import Optional
from dataclasses import dataclass
from src.vector.store import VectorStore
from src.graph.kuzu_graph import get_graph
from src.core.logging import get_logger

logger = get_logger(__name__)

# Default thresholds (configurable)
TIER1_THRESHOLD = 0.55
TIER2_THRESHOLD = 0.30


@dataclass
class ConnectivityResult:
    """Result of the connectivity gate for one input query."""
    query: str
    connectivity_score: float
    tier: int  # 1, 2, or 3
    max_chunk_similarity: float
    max_entity_similarity: float
    nearest_chunk_title: str = ""
    nearest_chunk_url: str = ""
    nearest_entity_name: str = ""
    nearest_entity_type: str = ""


async def compute_connectivity(
    query: str,
    query_embedding: list[float],
    vs: VectorStore,
    graph=None,
    tier1_threshold: float = TIER1_THRESHOLD,
    tier2_threshold: float = TIER2_THRESHOLD,
) -> ConnectivityResult:
    """
    Compute the connectivity score for a single input query.

    1. Whole-corpus vector similarity via ChromaDB
    2. Entity-level similarity via Kuzu graph name/synonym matching
    3. Connectivity score = max(chunk_sim, entity_sim)
    4. Tier classification
    """
    # ── 1. Whole-corpus vector similarity ─────────────────────────────
    max_chunk_sim = 0.0
    nearest_chunk_title = ""
    nearest_chunk_url = ""

    if vs and vs.collection.count() > 0:
        try:
            results = vs.collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                include=["documents", "metadatas", "distances"],
            )
            if results["ids"][0]:
                # ChromaDB returns L2 distances. For L2-normalized vectors:
                # dist^2 = 2 - 2*cos(sim) → sim = 1 - dist^2 / 2
                for i in range(len(results["ids"][0])):
                    dist = results["distances"][0][i]
                    sim = 1.0 - (dist * dist) / 2.0
                    sim = max(0.0, min(1.0, sim))
                    if sim > max_chunk_sim:
                        max_chunk_sim = sim
                        nearest_chunk_title = results["metadatas"][0][i].get("title", "")
                        nearest_chunk_url = results["metadatas"][0][i].get("url", "")
        except Exception as e:
            logger.warning(f"Vector query failed: {e}")

    # ── 2. Entity-level similarity ──────────────────────────────────
    max_entity_sim = 0.0
    nearest_entity_name = ""
    nearest_entity_type = ""

    if graph:
        try:
            # Check entity names via CONTAINS matching on all entities
            q_lower = query.lower()
            result = graph._conn.execute(
                "MATCH (e:Entity) RETURN e.id, e.name, e.type, e.synonyms"
            )
            df = result.get_as_df()
            for _, row in df.iterrows():
                name = str(row["e.name"]).lower()
                synonyms = str(row.get("e.synonyms", "")).lower() if row.get("e.synonyms") else ""
                # Token overlap similarity (Jaccard-like)
                query_tokens = set(q_lower.split())
                name_tokens = set(name.split())
                syn_tokens = set(synonyms.replace(",", " ").split())
                all_tokens = name_tokens | syn_tokens
                if query_tokens and all_tokens:
                    overlap = len(query_tokens & all_tokens)
                    sim = overlap / max(len(query_tokens), 1)
                    # Boost exact substring matches
                    if q_lower in name or q_lower in synonyms:
                        sim = max(sim, 0.6)
                    if sim > max_entity_sim:
                        max_entity_sim = sim
                        nearest_entity_name = str(row["e.name"])
                        nearest_entity_type = str(row.get("e.type", ""))
        except Exception as e:
            logger.warning(f"Entity similarity check failed: {e}")

    # ── 3. Connectivity Score ───────────────────────────────────────
    connectivity_score = max(max_chunk_sim, max_entity_sim)
    connectivity_score = max(0.0, min(1.0, connectivity_score))

    # ── 4. Tier Classification ──────────────────────────────────────
    if connectivity_score >= tier1_threshold:
        tier = 1
    elif connectivity_score >= tier2_threshold:
        tier = 2
    else:
        tier = 3

    logger.info(
        f"Connectivity gate: query='{query}' score={connectivity_score:.4f} "
        f"tier={tier} chunk_sim={max_chunk_sim:.4f} entity_sim={max_entity_sim:.4f} "
        f"nearest_chunk='{nearest_chunk_title}' nearest_entity='{nearest_entity_name}'"
    )

    return ConnectivityResult(
        query=query,
        connectivity_score=connectivity_score,
        tier=tier,
        max_chunk_similarity=max_chunk_sim,
        max_entity_similarity=max_entity_sim,
        nearest_chunk_title=nearest_chunk_title,
        nearest_chunk_url=nearest_chunk_url,
        nearest_entity_name=nearest_entity_name,
        nearest_entity_type=nearest_entity_type,
    )
