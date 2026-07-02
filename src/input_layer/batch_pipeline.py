"""Batch pipeline -- run NormalizedQuery list through full Q&A + scoring pipeline.
Phase 6 entry point: import <file> / analyze <text>"""
from dataclasses import dataclass, field
from typing import Optional
import asyncio
import os
import time

from ..core.logging import get_logger
from ..qa.pipeline import QAPipeline
from ..vector.store import VectorStore
from ..graph.kuzu_graph import get_graph
from ..score.coverage import compute_coverage_score, CoverageScore
from ..retrieve.hybrid import broken_path_detect, PathResult, get_suggested_rel
from . import parse_file, segment_text, NormalizedQuery, parse_expected_chain
from ..connectivity_gate import compute_connectivity

logger = get_logger(__name__)

# -- Result types ----------------------------------------------------------

@dataclass
class GapResult:
    """Single query's enriched break-detection result."""
    query: NormalizedQuery
    # Path detection
    path_result: PathResult
    coverage: Optional[CoverageScore]
    # Root cause (Phase 7)
    root_cause: str = "unknown"
    weak_edge: Optional[dict] = None
    weak_node: Optional[dict] = None
    alternative_path: list[str] = field(default_factory=list)
    evidence_used: list[str] = field(default_factory=list)
    # Opportunity (Phase 8)
    opportunity_score: float = 0.0
    recommendation: Optional[dict] = None


# -- Root Cause Classifier (Phase 7) ---------------------------------------

"""Batch pipeline -- run NormalizedQuery list through full Q&A + scoring pipeline.
Phase 6 entry point: import <file> / analyze <text>
Phase 7: weak_edge, weak_node, alternative_path, hop_count, root_cause (10 cats)
Phase 8: constrained recommendations (10 types + repairs), opportunity score (6-factor)
"""
from dataclasses import dataclass, field
from typing import Optional
import asyncio
import os
import time

from ..core.logging import get_logger
from ..qa.pipeline import QAPipeline
from ..vector.store import VectorStore
from ..graph.kuzu_graph import get_graph
from ..score.coverage import compute_coverage_score, CoverageScore
from ..retrieve.hybrid import broken_path_detect, PathResult, get_suggested_rel
from . import parse_file, segment_text, NormalizedQuery, parse_expected_chain

logger = get_logger(__name__)

# -- Result types --------------------------------------------------------------

@dataclass
class GapResult:
    """Single query's enriched break-detection result -- Phase 7/8."""
    query: NormalizedQuery
    # Path detection (Phase 3)
    path_result: PathResult
    coverage: Optional[CoverageScore]
    # Phase 7 -- Root cause + enrichment
    root_cause: str = "unknown"
    weak_edge: Optional[dict] = None      # {"src": ..., "tgt": ..., "confidence": float, "reason": str}
    weak_node: Optional[dict] = None      # {"node": ..., "degree": int, "罪名": str}
    alternative_path: list[str] = field(default_factory=list)
    hop_count: int = 0                    # total hops in full entity chain
    # Phase 8
    opportunity_score: float = 0.0
    recommendation: Optional[dict] = None  # {"action", "repairs", "detail", ...}
    # v5/v6 -- Page-level resolution & optimization
    page_resolutions: list[dict] = field(default_factory=list)  # from PageMapper
    page_optimization: Optional[dict] = None                     # decision + suggested new page
    # v4 -- Connectivity Gate fields
    connectivity_score: float = 0.0
    connectivity_tier: int = 0            # 0=notscored, 1=connected, 2=weak, 3=disconnected
    nearest_concept_name: str = ""
    nearest_concept_type: str = ""


# ==============================================================================
# PHASE 7 -- ROOT CAUSE CLASSIFIER (10 categories)
# ==============================================================================

def _node_degree(graph, node: str) -> int:
    """Return the degree (connection count) of a node, or 0 if not found."""
    try:
        related = graph.query_related(node)
        if related:
            return len(related)
    except Exception:
        pass
    return 0


def _node_schema_type(graph, node: str) -> Optional[str]:
    """Return schema type label for a node if detectable."""
    try:
        n = graph.query_node(node)
        if n:
            return n.get("schema_type") or n.get("type") or "Thing"
    except Exception:
        pass
    return None


def _classify_root_cause(
    gap: GapResult,
    graph,
    src_node, tgt_node,
    src_degree: int,
    tgt_degree: int,
) -> str:
    """
    Phase 7 -- Rule-based root cause classifier, all 10 categories.
    Evaluated in priority order (most specific first).
    """
    pr = gap.path_result
    hops = pr.hops
    missing_rel = pr.missing_rel

    # -- 1. missing_entity ---------------------------------------------
    # Neither endpoint exists in the graph at all
    if src_node is None and tgt_node is None:
        return "missing_entity"
    if src_node is None or tgt_node is None:
        return "missing_entity"

    # -- 2. conflicting_relationships ---------------------------------
    # Both entities exist but have edges to each other with contradictory labels
    # e.g. "X is spicy" vs "X is mild" -- needs multi-edge check
    try:
        src_rels = graph.query_related(src_node.get("name", pr.source))
        tgt_rels = graph.query_related(tgt_node.get("name", pr.target))
        # Heuristic: if both link to each other with different relationship labels
        if src_rels and tgt_rels:
            rel_labels = {r.get("relationship", "") for r in src_rels}
            if len(rel_labels) > 1:
                return "conflicting_relationships"
    except Exception:
        pass

    # -- 3. missing_relationship ----------------------------------------
    # Both entities present but no edge connects them
    if missing_rel:
        return "missing_relationship"

    # -- 4. weak_edge ---------------------------------------------------
    # An edge exists (hop ≥ 1) but is weak: low degree endpoints + low evidence
    if hops >= 1:
        avg_degree = (src_degree + tgt_degree) / 2
        evidence_weak = (
            gap.coverage is None or
            gap.coverage.evidence_strength < 0.4 or
            len(gap.path_result.entities_found) < 2
        )
        if avg_degree <= 2 or evidence_weak:
            return "weak_edge"

    # -- 5. weak_node ---------------------------------------------------
    # Entity exists but is a leaf/orphan (degree ≤ 1) -- hard to route through
    if src_degree <= 1 or tgt_degree <= 1:
        return "weak_node"

    # -- 6. isolated_graph_component ------------------------------------
    # Both entities exist independently but are in separate disconnected components
    if hops == 0 and src_node and tgt_node and missing_rel is None:
        # Check if they have any shared neighbours
        try:
            src_neighbors = {n["name"] for n in graph.query_related(pr.source) or []}
            tgt_neighbors = {n["name"] for n in graph.query_related(pr.target) or []}
            if not (src_neighbors & tgt_neighbors):
                # No shared neighbours = potentially separate components
                return "isolated_graph_component"
        except Exception:
            pass

    # -- 7. weak_topical_authority --------------------------------------
    # The target entity exists but has very few inbound citations/references
    # Proxy: target degree + coverage overall score
    if tgt_degree <= 3 and hops == 0:
        if gap.coverage and gap.coverage.overall < 0.4:
            return "weak_topical_authority"

    # -- 8. sparse_supporting_content ----------------------------------
    # Entity exists with decent degree but semantic coverage is weak
    if gap.coverage and gap.coverage.semantic_coverage < 0.3:
        return "sparse_supporting_content"

    # -- 9. missing_schema ---------------------------------------------
    # Entity exists but has no schema_type annotation
    src_schema = _node_schema_type(graph, pr.source)
    tgt_schema = _node_schema_type(graph, pr.target)
    if src_schema is None or tgt_schema is None:
        # Heuristic: informational query with no schema
        if gap.query.intent == "informational" and hops == 0:
            return "missing_schema"

    # -- 10. missing_faq -----------------------------------------------
    # Question-shaped query (what/how/why/which/where) with no direct answer
    # detected via source coverage
    if any(w in gap.query.original.lower() for w in ["what", "how", "why", "which", "where", "who", "when"]):
        if gap.coverage and gap.coverage.semantic_coverage < 0.5:
            return "missing_faq"

    # -- 11. weak_internal_linking --------------------------------------
    # Both entities exist but only via very long indirect paths (hops > 3)
    # suggesting poor internal linking structure
    if hops > 3:
        return "weak_internal_linking"

    # -- Default -------------------------------------------------------
    return "missing_relationship"


# -- Opportunity Score (Phase 8) -------------------------------------------

@dataclass
class OpportunityScorer:
    """Computes opportunity score per Section 4.2."""
    w_demand: float = 0.25
    w_business: float = 0.20
    w_centrality: float = 0.20
    w_reasoning: float = 0.15
    w_evidence: float = 0.10
    w_effort: float = 0.10
    # Business value overrides per entity (set during batch)
    entity_weights: dict = field(default_factory=dict)

    def compute(self, gap: GapResult, centrality_hit_count: int = 1) -> float:
        w = self
        gap_entities = {e.lower() for e in gap.query.detected_entities}

        # SearchDemand -- normalized volume (0-1) or frequency proxy
        demand = 0.5  # default if no volume
        if gap.query.volume:
            demand = min(1.0, gap.query.volume / 1000.0)

        # BusinessValue -- configurable, default 0.5
        business = 0.5
        for e in gap_entities:
            if e in self.entity_weights:
                business = max(business, self.entity_weights[e])
        # Hero products get a boost
        hero_entities = {"wickedgüd", "noodles", "pasta", "ramen", "combos"}
        if gap_entities & hero_entities:
            business = max(business, 0.7)

        # GraphCentrality -- how many other gaps share the same missing node/edge
        centrality = min(1.0, centrality_hit_count / 5.0)

        # AIReasoningImpact -- hops away from complete answer (fewer hops = higher)
        if gap.path_result.hops >= 3:
            reasoning = 0.2
        elif gap.path_result.hops == 2:
            reasoning = 0.5
        elif gap.path_result.hops == 1:
            reasoning = 0.8
        else:
            reasoning = 1.0  # 0 hops = already answered

        # EvidenceWeakness -- inverse of confidence
        conf = 0.7  # default
        if gap.coverage and gap.coverage.evidence_strength:
            conf = 1.0 - gap.coverage.evidence_strength
        evidence = min(1.0, conf)

        # ContentEffort -- hard to fix vs easy (FAQ=easy, new page=hard)
        effort = 0.5
        if gap.root_cause in ("missing_faq", "weak_evidence", "missing_schema"):
            effort = 0.2  # easy
        elif gap.root_cause in ("missing_entity", "missing_relationship"):
            effort = 0.5  # medium
        else:
            effort = 0.8  # hard

        score = (
            w.w_demand * demand +
            w.w_business * business +
            w.w_centrality * centrality +
            w.w_reasoning * reasoning +
            w.w_evidence * evidence +
            w.w_effort * (1.0 - effort)
        )
        return round(score, 4)


# -- Recommendation Generator (Phase 8) -----------------------------------

RECOMMENDATION_ACTIONS = [
    "create_relationship",
    "create_supporting_entity",
    "strengthen_evidence",
    "create_faq",
    "create_supporting_page",
    "improve_internal_links",
    "add_schema",
    "add_authoritative_reference",
    "connect_orphan_node",
    "increase_topic_authority",
    "add_product_section",
]

def generate_recommendation(gap: GapResult) -> dict:
    """
    Phase 8 -- Constrained recommendation with GAP-SPECIFIC repairs statement.
    Action is one of 10 types. repairs field explains exactly what semantic
    relationship is broken and how to repair it (never 'add keyword').
    """
    src = gap.path_result.source or ""
    tgt = gap.path_result.target or ""
    rc = gap.root_cause
    hops = gap.hop_count
    break_pt = gap.path_result.break_point or f"{src} -> {tgt}"

    # -- Action type mapping (one of 10) ---------------------------------
    action_map = {
        "missing_entity":              "create_supporting_entity",
        "missing_relationship":       "create_relationship",
        "weak_edge":                  "strengthen_evidence",
        "weak_node":                  "connect_orphan_node",
        "missing_faq":                "create_faq",
        "sparse_supporting_content":  "create_supporting_page",
        "weak_internal_linking":      "improve_internal_links",
        "missing_schema":             "add_schema",
        "weak_topical_authority":     "increase_topic_authority",
        "isolated_graph_component":  "connect_orphan_node",
        "conflicting_relationships":  "add_authoritative_reference",
    }
    action = action_map.get(rc, "create_relationship")

    # -- Repairs: gap-specific statement of what semantic path is broken --
    # Format: "Repair: [broken relationship]. By: [specific action]."
    repairs_map = {
        "missing_relationship": (
            f"Repair: no edge exists between '{src}' and '{tgt}' in the semantic graph. "
            f"By: add a content statement that makes the relationship explicit. "
            f"E.g. '[{src}] {get_suggested_rel(src, tgt)} [{tgt}].'"
        ),
        "missing_entity": (
            f"Repair: '{tgt}' does not exist as a node in the semantic graph. "
            f"By: create a dedicated page or section for '{tgt}' that includes "
            f"at least 3 relationships to existing nodes."
        ),
        "weak_edge": (
            f"Repair: the edge between '{src}' and '{tgt}' has low confidence "
            f"(low endpoint degree + weak evidence). "
            f"By: add 2-3 attribute statements and citations that reinforce the "
            f"'{src} -> {tgt}' relationship."
        ),
        "weak_node": (
            f"Repair: '{tgt}' is a leaf node (degree ≤ 1) and cannot be routed to. "
            f"By: connect '{tgt}' to at least 2 other relevant nodes via "
            f"explicit relationship statements."
        ),
        "missing_faq": (
            f"Repair: the query '{gap.query.original[:60]}' is a question-type "
            f"search but no FAQ content directly answers it. "
            f"By: add an FAQ entry on the '{src}' page that explicitly answers "
            f"the question and includes the '{src} -> {tgt}' relationship."
        ),
        "sparse_supporting_content": (
            f"Repair: semantic coverage for '{src} -> {tgt}' is weak (low citation density). "
            f"By: expand the '{tgt}' page with 2-3 additional supporting sections "
            f"covering attributes, comparisons, or use-cases."
        ),
        "weak_internal_linking": (
            f"Repair: '{src}' and '{tgt}' require {hops} hops to connect -- "
            f"internal link structure is too shallow. "
            f"By: add direct hyperlinks from the '{src}' page to '{tgt}' "
            f"and from at least one intermediate page."
        ),
        "missing_schema": (
            f"Repair: the '{src}' -> '{tgt}' relationship has no structured data markup. "
            f"By: add JSON-LD (Product/FAQ/HowTo schema) to the '{src}' page "
            f"marking up the relationship with type, properties, and expected answer."
        ),
        "weak_topical_authority": (
            f"Repair: '{tgt}' has low inbound reference density -- "
            f"appears unconnected to the broader topical graph. "
            f"By: create 2-3 supporting articles that reference '{tgt}' "
            f"with natural mentions and cross-links."
        ),
        "isolated_graph_component": (
            f"Repair: '{src}' and '{tgt}' exist but belong to disconnected "
            f"graph components (no shared neighbours). "
            f"By: on the '{src}' page, add links to at least 2 nodes from "
            f"the main component to bridge the two subgraphs."
        ),
        "conflicting_relationships": (
            f"Repair: '{src}' has multiple outgoing edges with contradictory "
            f"relationship labels -- AI engines will detect inconsistency. "
            f"By: audit all edges from '{src}' and resolve to a single "
            f"consistent relationship type, then add authoritative citation."
        ),
    }
    repairs = repairs_map.get(
        rc,
        f"Repair: semantic path between '{src}' and '{tgt}' is broken or missing. "
        f"By: add explicit relationship content."
    )

    # -- Detail: implementation guide (what exactly to write/create/link) -
    detail_map = {
        "create_relationship": (
            f"Write a content block: 'About {src}: [{src}] {get_suggested_rel(src, tgt)} [{tgt}]. "
            f"Add this to the primary '{src}' page and link to the '{tgt}' page."
        ),
        "create_supporting_entity": (
            f"Create a new page titled '{tgt}' with: (1) type/category, (2) key attributes, "
            f"(3) comparisons, (4) at least 2 natural mentions of '{src}' with links."
        ),
        "strengthen_evidence": (
            f"Add to the existing '{src}' page: (1) 2-3 citations from authoritative sources, "
            f"(2) a direct attribute statement, (3) an FAQ or comparison table including '{tgt}'."
        ),
        "create_faq": (
            f"Add FAQ schema to the '{src}' page: Q='{gap.query.original[:80]}', "
            f"A='[{src}] {get_suggested_rel(src, tgt)} [{tgt}].' Also add a human-readable "
            f"FAQ section with the same answer and a link to '{tgt}'."
        ),
        "add_product_section": (
            f"Add a new section titled 'For {tgt}' to the existing '{src}' product page. "
            f"Content: audience framing, benefit statement, link to product.\n"
        ),
        "create_supporting_page": (
            f"Create a supporting page about '{src}' with sections: Overview, Attributes, "
            f"Use-cases, Comparisons (include '{tgt}'), Related Links. "
            f"Include natural mentions with internal links to '{tgt}'."
        ),
        "improve_internal_links": (
            f"From the '{src}' page, add: (1) a hyperlink anchor text='{tgt}' "
            f"pointing to the '{tgt}' page, (2) reciprocal link from '{tgt}' page. "
            f"Use descriptive anchor text, not raw URLs."
        ),
        "add_schema": (
            f"Add JSON-LD to the '{src}' page <head>: use type Product or FAQPage. "
            f"Properties: name='{src}', description should mention '{tgt}', "
            f"url must point to the canonical '{src}' page URL."
        ),
        "add_authoritative_reference": (
            f"Find an authoritative external source that mentions the '{src} -> {tgt}' "
            f"relationship (official docs, recognized authority). Add as citation with "
            f"anchor text and structured citation schema."
        ),
        "connect_orphan_node": (
            f"Identify 2-3 relevant existing pages in the main graph component. "
            f"From each, add a natural mention of '{tgt}' with an internal link. "
            f"E.g. 'Related: [{tgt}]({tgt} page).'"
        ),
        "increase_topic_authority": (
            f"Write 2-3 supporting articles (500+ words each) that naturally mention "
            f"'{src}' and '{tgt}'. Topics: how-to, comparison, use-case. "
            f"Include internal links and FAQ schema."
        ),
    }
    detail = detail_map.get(
        action,
        f"Fix the semantic gap between '{src}' and '{tgt}' by adding "
        f"explicit relationship content on the relevant page."
    )

    return {
        "action": action,
        "repairs": repairs,
        "detail": detail,
        "target_entities": [e for e in [src, tgt] if e],
        "root_cause": rc,
        "priority": "high" if rc in (
            "missing_relationship", "missing_entity", "conflicting_relationships"
        ) else "medium" if rc in (
            "weak_edge", "weak_node", "missing_faq"
        ) else "low",
        # v5/v6 -- Page-level resolution
        "page_resolutions": gap.page_resolutions,
        "page_decision": gap.page_optimization["recommendation_type"] if gap.page_optimization else None,
        "page_decision_reason": gap.page_optimization["reason"] if gap.page_optimization else None,
        "suggested_new_page": gap.page_optimization["suggested_new_page"] if gap.page_optimization else None,
        "suggested_edits": gap.page_optimization["suggested_edits"] if gap.page_optimization else [],
    }


# -- Batch Pipeline --------------------------------------------------------

async def run_batch_audit(
    queries: list[NormalizedQuery],
    top_k: int = 5,
) -> list[GapResult]:
    """
    Run a batch of NormalizedQuery objects through the full pipeline:
    embed -> vector search -> hybrid retrieval -> broken path detection
    -> root cause classification -> coverage scoring -> opportunity scoring
    -> recommendation generation.

    Returns list[GapResult], one per query.
    """
    if not queries:
        return []

    logger.info(f"Batch pipeline: {len(queries)} queries")

    vs = VectorStore()
    try:
        graph = get_graph()
        graph_available = True
    except Exception as e:
        logger.warning(f"Graph not available: {e}")
        graph = None
        graph_available = False

    qa = QAPipeline(vs, neo4j=None)

    results: list[GapResult] = []
    # Count how many gaps share the same missing node for centrality
    missing_node_counts: dict[str, int] = {}

    for i, nq in enumerate(queries):
        try:
            gap = await _process_single_query(nq, vs, qa, graph, top_k)
            results.append(gap)

            # Track missing node frequency
            if gap.path_result.missing_rel:
                key = f"{gap.path_result.source}->{gap.path_result.target}"
                missing_node_counts[key] = missing_node_counts.get(key, 0) + 1

        except Exception as e:
            logger.warning(f"Query {i+1} failed: {e}")
            results.append(GapResult(
                query=nq,
                path_result=PathResult(source=nq.original, target=""),
                coverage=None,
                root_cause="pipeline_error",
            ))

    # Phase 8: compute opportunity scores (needs centrality pass)
    scorer = OpportunityScorer()
    for gap in results:
        if gap.root_cause == "disconnected":
            gap.opportunity_score = 0.0
            gap.recommendation = None
            continue
        key = f"{gap.path_result.source}->{gap.path_result.target}"
        centrality = missing_node_counts.get(key, 1)
        gap.opportunity_score = scorer.compute(gap, centrality_hit_count=centrality)
        gap.recommendation = generate_recommendation(gap)

    results.sort(key=lambda g: g.opportunity_score, reverse=True)
    return results


async def _process_single_query(
    nq: NormalizedQuery,
    vs: VectorStore,
    qa: QAPipeline,
    graph,
    top_k: int,
) -> GapResult:
    """Process one NormalizedQuery through the full Phase 3-7 pipeline."""
    from ..retrieve.hybrid import get_suggested_rel

    # Embed question
    q_emb = await qa.embed_question_async(nq.canonical)
    if not q_emb:
        return GapResult(
            query=nq,
            path_result=PathResult(source=nq.original, target=""),
            coverage=None,
        )

    # -- v4 Stage 0: Semantic Connectivity Gate ------------------------
    gate_result = await compute_connectivity(
        nq.canonical, q_emb, vs, graph
    )
    gap = GapResult(
        query=nq,
        path_result=PathResult(source=nq.original, target=""),
        coverage=None,
        connectivity_score=gate_result.connectivity_score,
        connectivity_tier=gate_result.tier,
        nearest_concept_name=gate_result.nearest_entity_name or gate_result.nearest_chunk_title,
        nearest_concept_type=gate_result.nearest_entity_type or "chunk",
    )

    # Tier 3 -- Disconnected: skip chain extraction, return disconnection report
    if gate_result.tier >= 3:
        gap.root_cause = "disconnected"
        gap.path_result = PathResult(
            source=nq.original,
            target=gate_result.nearest_entity_name or gate_result.nearest_chunk_title,
            full_path=[nq.original],
            hops=0,
        )
        return gap

    # Tier 1/2 -- Connected/Weak: proceed to full v3 pipeline
    gap.connectivity_tier = gate_result.tier

    # Vector search
    sources = vs.search(q_emb, top_k=top_k)

    # Get expected entity chain
    entity_chain = parse_expected_chain(nq, graph)

    # Phase 3: broken path detection
    path_result: PathResult
    if graph and entity_chain:
        try:
            path_result = broken_path_detect(nq.canonical, entity_chain, graph)
        except Exception as e:
            logger.warning(f"Path detection failed: {e}")
            path_result = PathResult(source=nq.original, target="")
    else:
        path_result = PathResult(source=nq.original, target="")

    # Pre-compute graph metadata for classifier
    src_node, tgt_node = None, None
    src_degree, tgt_degree = 0, 0
    if graph:
        try:
            src_node = graph.query_node(path_result.source)
        except Exception:
            pass
        try:
            tgt_node = graph.query_node(path_result.target)
        except Exception:
            pass
        src_degree = _node_degree(graph, path_result.source) if path_result.source else 0
        tgt_degree = _node_degree(graph, path_result.target) if path_result.target else 0

    # Set v3 fields on the existing gap (created by connectivity gate above)
    gap.path_result = path_result
    gap.hop_count = len(entity_chain) - 1 if entity_chain else 0

    # Phase 7: root cause -- now passes full gap + graph metadata
    root_cause = _classify_root_cause(gap, graph, src_node, tgt_node, src_degree, tgt_degree)
    gap.root_cause = root_cause

    # -- Populate weak_edge ---------------------------------------------------
    if root_cause == "weak_edge" and src_node and tgt_node:
        gap.weak_edge = {
            "src": path_result.source,
            "tgt": path_result.target,
            "avg_degree": round((src_degree + tgt_degree) / 2, 1),
            "evidence_strength": gap.coverage.evidence_strength if gap.coverage else 0.0,
            "reason": (
                f"Edge between '{path_result.source}' and '{path_result.target}' "
                f"has low confidence (avg endpoint degree={round((src_degree+tgt_degree)/2,1)}, "
                f"evidence={gap.coverage.evidence_strength if gap.coverage else 0.0:.2f})"
            ),
        }

    # -- Populate weak_node --------------------------------------------------
    if root_cause in ("weak_node", "isolated_graph_component") and (src_node or tgt_node):
        weak_name = path_result.source if src_degree <= 1 else path_result.target
        weak_degree = src_degree if src_degree <= 1 else tgt_degree
        罪名 = "leaf/orphan node (degree ≤ 1)" if root_cause == "weak_node" else "disconnected component"
        gap.weak_node = {
            "node": weak_name,
            "degree": weak_degree,
            "罪名": 罪名,
            "reason": f"Node '{weak_name}' is a {罪名} -- cannot be routed to/from by AI engines.",
        }

    # -- Populate alternative_path -----------------------------------------
    if graph and path_result.source and path_result.target:
        try:
            alt_paths = graph.find_paths(path_result.source, path_result.target, max_hops=3)
            if alt_paths:
                # Pick the shortest alternative that differs from the broken path
                for p in alt_paths:
                    if isinstance(p, dict) and p.get("path") and p["path"] != path_result.full_path:
                        gap.alternative_path = p["path"]
                        break
                    elif isinstance(p, list) and p != path_result.full_path:
                        gap.alternative_path = p
                        break
        except Exception as e:
            logger.debug(f"Alternative path search failed: {e}")

    # Phase 4: coverage scoring
    coverage = None
    if graph and entity_chain:
        try:
            coverage = await compute_coverage_score(
                question=nq.canonical,
                question_embedding=q_emb,
                entity_chain=entity_chain,
                graph_edge_count=path_result.hops,
                total_expected_edges=max(1, len(entity_chain) - 1),
                evidence_confidences=[0.7] * max(0, path_result.hops),
                vector_store=vs,
                graph=graph,
            )
            # v5/v6 -- Knowledge Graph Completeness + page-level resolution
            if coverage and entity_chain:
                from ..page_mapper import PageMapper
                mapper = PageMapper([])  # lightweight -- just needs graph queries
                chain_resolved = mapper.resolve_chain(entity_chain, graph)
                nodes_present = sum(1 for r in chain_resolved if r.status == "found")
                coverage.kg_completeness = nodes_present / max(len(entity_chain), 1)
                # Store page-level resolution on gap for generate_recommendation
                gap.page_resolutions = [
                    {
                        "entity_name": r.entity_name,
                        "status": r.status,
                        "supporting_pages": r.supporting_pages[:3],
                        "chunk_count": r.chunk_count,
                    }
                    for r in chain_resolved
                ]
                opt = mapper.decide_edit_or_create(chain_resolved)
                gap.page_optimization = {
                    "recommendation_type": opt.recommendation_type,
                    "reason": opt.reason,
                    "suggested_new_page": opt.suggested_new_page,
                    "suggested_edits": opt.suggested_edits,
                }
        except Exception as e:
            logger.warning(f"Coverage scoring failed: {e}")
    gap.coverage = coverage

    # -- Phase 7: re-classify once coverage is available -----------------
    # Re-run root cause now that coverage is populated
    gap.root_cause = _classify_root_cause(gap, graph, src_node, tgt_node, src_degree, tgt_degree)

    return gap


# -- CLI-facing helpers ----------------------------------------------------

async def audit_file(path: str) -> list[GapResult]:
    """Load and process a file -- auto-detects format."""
    logger.info(f"Auditing file: {path}")
    queries = parse_file(path)
    logger.info(f"Parsed {len(queries)} queries from {path}")
    results = await run_batch_audit(queries)
    # Persist for gap-report
    import json
    report_data = _serialize_results(results)
    out_path = "data/gap_batch_results.json"
    os.makedirs("data", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Results cached: {out_path}")
    return results


async def audit_text(text: str) -> list[GapResult]:
    """Segment pasted text and process as batch."""
    queries = segment_text(text)
    logger.info(f"Segmented {len(queries)} queries from pasted text")
    return await run_batch_audit(queries)


def print_batch_summary(results: list[GapResult]) -> None:
    """Print compact summary of batch results to console -- v4 three-bucket."""
    if not results:
        print("No results.")
        return

    disconnected = sum(1 for g in results if g.root_cause == "disconnected")
    covered = sum(1 for g in results if g.coverage and g.coverage.verdict == "YES")
    partial = sum(1 for g in results if g.coverage and g.coverage.verdict == "PARTIALLY")
    gaps = len(results) - covered - partial - disconnected
    connected = len(results) - disconnected
    avg_score = sum(g.opportunity_score for g in results if g.root_cause != "disconnected") / max(connected, 1)

    print(f"\n{'='*60}")
    print(f"  BATCH SUMMARY -- {len(results)} queries")
    print(f"{'='*60}")
    print(f"  Connected / Analyzed:      {connected:>4}")
    print(f"  Disconnected (Tier 3):     {disconnected:>4}")
    print(f"  Full coverage (YES):       {covered:>4}")
    print(f"  Partial coverage:          {partial:>4}")
    print(f"  Gaps found:                {gaps:>4}")
    print(f"  Average opportunity score: {avg_score:.3f}")
    print(f"{'='*60}\n")

    # Top gaps by opportunity score (connected only)
    connected_results = [g for g in results if g.root_cause != "disconnected"]
    if connected_results:
        print(f"  TOP 10 GAPS BY OPPORTUNITY SCORE:")
        print(f"  {'#':<4} {'OpScore':<8} {'Root Cause':<25} {'Query':<30}")
        print(f"  {'-'*4} {'-'*8} {'-'*25} {'-'*30}")
        for i, g in enumerate(connected_results[:10], 1):
            query_short = g.query.original[:28] + ".." if len(g.query.original) > 30 else g.query.original
            print(f"  {i:<4} {g.opportunity_score:<8.4f} {g.root_cause:<25} {query_short}")

    # Action type summary (connected only)
    action_counts: dict[str, int] = {}
    for g in connected_results:
        if g.recommendation:
            a = g.recommendation["action"]
            action_counts[a] = action_counts.get(a, 0) + 1

    if action_counts:
        print(f"\n  RECOMMENDATION BREAKDOWN:")
        for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            print(f"    {action:<30} {count:>4}")

    # Disconnected summary
    if disconnected:
        print(f"\n  DISCONNECTED CONCEPTS (Tier 3):")
        print(f"  {'#':<4} {'Score':<8} {'Query':<35} {'Nearest Concept':<25}")
        print(f"  {'-'*4} {'-'*8} {'-'*35} {'-'*25}")
        disconnected_by_score = sorted(results, key=lambda g: g.connectivity_score, reverse=True)
        for i, g in enumerate([r for r in disconnected_by_score if r.root_cause == "disconnected"][:10], 1):
            q_short = g.query.original[:33] + ".." if len(g.query.original) > 35 else g.query.original
            nc = g.nearest_concept_name[:23] + ".." if len(g.nearest_concept_name) > 25 else g.nearest_concept_name
            print(f"  {i:<4} {g.connectivity_score:<8.4f} {q_short:<35} {nc:<25}")


def _serialize_results(results: list[GapResult]) -> list[dict]:
    """Serialize list[GapResult] to JSON-serializable list[dict]."""
    out = []
    for g in results:
        out.append({
            "query": {
                "original": g.query.original,
                "canonical": g.query.canonical,
                "intent": g.query.intent,
                "detected_entities": g.query.detected_entities,
                "source": g.query.source,
                "line_number": g.query.line_number,
            },
            "path_result": {
                "source": g.path_result.source,
                "target": g.path_result.target,
                "full_path": g.path_result.full_path,
                "hops": g.path_result.hops,
                "break_point": g.path_result.break_point,
                "missing_rel": g.path_result.missing_rel,
                "suggested_fix": g.path_result.suggested_fix,
                "entities_found": g.path_result.entities_found,
            },
            "coverage": None if g.coverage is None else {
                "entity_coverage": g.coverage.entity_coverage,
                "relationship_coverage": g.coverage.relationship_coverage,
                "semantic_coverage": g.coverage.semantic_coverage,
                "evidence_strength": g.coverage.evidence_strength,
                "overall": g.coverage.overall,
                "verdict": g.coverage.verdict,
            },
            # Phase 7 -- enriched break detection
            "root_cause": g.root_cause,
            "weak_edge": g.weak_edge,
            "weak_node": g.weak_node,
            "alternative_path": g.alternative_path,
            "hop_count": g.hop_count,
            # Phase 8
            "opportunity_score": g.opportunity_score,
            "recommendation": g.recommendation,
            # v4 -- Connectivity Gate
            "connectivity_score": g.connectivity_score,
            "connectivity_tier": g.connectivity_tier,
            "nearest_concept_name": g.nearest_concept_name,
            "nearest_concept_type": g.nearest_concept_type,
        })
    return out