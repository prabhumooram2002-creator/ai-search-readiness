"""Coverage scoring — Phase 4 implementation of Section 6 formulas."""
import math
from dataclasses import dataclass
from typing import Optional
from ..core.config import scoring
from ..graph.kuzu_graph import get_graph
from ..vector.store import VectorStore
from ..core.logging import get_logger

logger = get_logger(__name__)

@dataclass
class CoverageScore:
    entity_coverage: float       # 0-1
    relationship_coverage: float # 0-1
    semantic_coverage: float     # 0-1
    evidence_strength: float      # 0-1
    overall: float               # weighted average
    verdict: str                 # YES / PARTIALLY / NO
    kg_completeness: float = 0.0  # v5 — Knowledge Graph Completeness (0-1)

    def to_dict(self) -> dict:
        return {
            "entity_coverage": round(self.entity_coverage, 3),
            "relationship_coverage": round(self.relationship_coverage, 3),
            "semantic_coverage": round(self.semantic_coverage, 3),
            "evidence_strength": round(self.evidence_strength, 3),
            "overall": round(self.overall, 3),
            "verdict": self.verdict,
            "kg_completeness": round(self.kg_completeness, 3),
        }


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _verdict(score: float, thresholds: dict) -> str:
    """Map score to YES / PARTIALLY / NO."""
    if score >= thresholds.get("yes", 0.75):
        return "YES"
    elif score >= thresholds.get("partially", 0.40):
        return "PARTIALLY"
    else:
        return "NO"


async def compute_coverage_score(
    question: str,
    question_embedding: list[float],
    entity_chain: list[str],
    graph_edge_count: int,
    total_expected_edges: int,
    evidence_confidences: list[float],
    vector_store: VectorStore,
    graph=None,
    top_k: int = 5,
) -> CoverageScore:
    """
    Compute coverage scores per Section 6 formulas.
    
    entity_coverage = (# expected-chain entities found) / (# expected total)
    relationship_coverage = (# edges present) / (# edges total)
    semantic_coverage = avg cosine similarity of top-k chunks vs question
    evidence_strength = mean confidence of relationship edges used
    overall = weighted average
    """
    cfg = scoring()
    weights = cfg.get("weights", {})
    thresholds = cfg.get("verdict_thresholds", {})
    
    w_entity = weights.get("entity_coverage", 0.35)
    w_rel = weights.get("relationship_coverage", 0.35)
    w_sem = weights.get("semantic_coverage", 0.20)
    w_evidence = weights.get("evidence_strength", 0.10)
    
    # Entity coverage
    entities_found = 0
    if graph and entity_chain:
        for name in entity_chain:
            related = graph.query_related(name)
            if related:
                entities_found += 1
    entity_coverage = entities_found / len(entity_chain) if entity_chain else 0.0
    
    # Relationship coverage
    relationship_coverage = graph_edge_count / total_expected_edges if total_expected_edges > 0 else 0.0
    
    # Semantic coverage — vector similarity
    if vector_store:
        results = vector_store.search(question_embedding, top_k=top_k)
        if results and results[0].distance is not None:
            # ChromaDB returns L2 distance, convert to similarity
            # L2 distance of 0 = similarity 1, L2 of 2 = similarity 0
            avg_dist = sum(r.distance for r in results[:top_k]) / len(results)
            semantic_coverage = max(0.0, 1.0 - avg_dist / 2.0)
        else:
            semantic_coverage = 0.0
    else:
        semantic_coverage = 0.0
    
    # Evidence strength
    if evidence_confidences:
        evidence_strength = sum(evidence_confidences) / len(evidence_confidences)
    else:
        evidence_strength = 0.0
    
    # Overall weighted
    overall = (
        w_entity * entity_coverage +
        w_rel * relationship_coverage +
        w_sem * semantic_coverage +
        w_evidence * evidence_strength
    )
    
    verdict = _verdict(overall, thresholds)
    
    score = CoverageScore(
        entity_coverage=entity_coverage,
        relationship_coverage=relationship_coverage,
        semantic_coverage=semantic_coverage,
        evidence_strength=evidence_strength,
        overall=overall,
        verdict=verdict,
    )
    
    logger.info(
        f"Coverage scores — entity={entity_coverage:.2f}, rel={relationship_coverage:.2f}, "
        f"semantic={semantic_coverage:.2f}, evidence={evidence_strength:.2f} → "
        f"overall={overall:.2f} [{verdict}]"
    )
    
    return score


def generate_recommendations(score: CoverageScore, missing_entities: list[str] = None,
                               missing_rels: list[str] = None) -> list[str]:
    """
    Generate actionable content recommendations based on coverage scores.
    Phase 4 deliverable.
    """
    recommendations = []
    
    if score.entity_coverage < 0.75:
        recommendations.append(
            f"❌ Low entity coverage ({score.entity_coverage:.0%}). "
            f"Add more content explicitly naming your key entities. "
            f"Missing: {', '.join(missing_entities[:3]) if missing_entities else 'see graph'}"
        )
    
    if score.relationship_coverage < 0.75:
        recommendations.append(
            f"❌ Low relationship coverage ({score.relationship_coverage:.0%}). "
            f"Add explicit relationship statements. "
            f"e.g., 'Product X {REL_SUGGESTIONS}'"
        )
    
    if score.semantic_coverage < 0.5:
        recommendations.append(
            f"⚠️  Low semantic coverage ({score.semantic_coverage:.0%}). "
            f"Consider adding an FAQ section or detailed product descriptions "
            f"that directly address common questions."
        )
    
    if score.evidence_strength < 0.5:
        recommendations.append(
            f"⚠️  Low evidence strength ({score.evidence_strength:.0%}). "
            f"Add more cited claims and source references to boost "
            f"AI confidence in extracted relationships."
        )
    
    if score.overall >= 0.75:
        recommendations.append(
            f"✅ Strong overall coverage ({score.overall:.0%}). "
            f"This content is well-structured for AI search visibility."
        )
    
    return recommendations

REL_SUGGESTIONS = [
    "contains", "is made from", "supports", "is recommended for",
    "treats", "belongs to", "is part of", "is related to",
]