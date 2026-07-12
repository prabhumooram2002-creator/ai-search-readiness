"""LAYER 3 — Explainability report over the Trace (CLAUDE.md).

PURE INTROSPECTION — no new inference. Every row traces to a concrete
upstream value recorded by Layers 0-2; anything else is a bug. Citation
probability is always labeled "simulated" (observed rates only exist if real
engine runs were logged — never estimated here).
"""
from __future__ import annotations

from .core.logging import get_logger

logger = get_logger(__name__)

WEAK_EVIDENCE_FLOOR = 0.5


def _step(trace, name):
    return next((s for s in trace.steps if s.name == name), None)


def explain_query(trace, kg=None) -> dict:
    """Build the per-query explainability rows from a Layer-2 Trace."""
    rows: dict = {"query": trace.query}

    # Why this answer? -> synthesis + citations
    rows["answer"] = trace.answer
    synth = _step(trace, "answer_synthesis")
    rows["why_this_answer"] = {
        "citations": trace.citations,
        "n_source_chunks": synth.inputs.get("n_sources", 0) if synth else 0,
    }

    # Why this page? -> rerank scores of winning chunks
    rows["why_these_chunks"] = trace.reranked

    # Why not another page? -> retrieved-but-lost with their losing scores
    kept = {r["chunk_id"] for r in trace.reranked}
    rows["retrieved_but_lost"] = [
        r for r in trace.retrieved if r["chunk_id"] not in kept]

    # Which chunks lost confidence? -> drops recorded at drop time
    rows["dropped_with_reasons"] = {
        s.name: s.dropped for s in trace.steps if s.dropped}

    # Missing entities: query-constraint entities with no Entity node in KG
    missing_entities = []
    q_entities = (trace.intent or {}).get("constraints", {}).get("entities", []) or []
    if kg is not None:
        for name in q_entities:
            res = kg._exec(
                "MATCH (e:Entity) WHERE lower(e.name) = $n "
                "OR CONTAINS(lower(e.synonyms), $n) RETURN e.id LIMIT 1",
                {"n": str(name).lower()})
            if not res.has_next():
                missing_entities.append(name)
    trace.missing_entities = missing_entities
    rows["missing_entities"] = missing_entities

    # Weak evidence: evidence entries below the floor
    weak = [e for e in trace.evidence
            if e["nli_label"] != "entailment" or e["nli_conf"] < WEAK_EVIDENCE_FLOOR]
    trace.weak_evidence = weak
    rows["weak_evidence"] = weak

    # Circular references: RelatesTo cycles among answer entities (Kùzu)
    circular = []
    if kg is not None:
        res = kg._exec(
            "MATCH (a:Entity)-[:RelatesTo]->(b:Entity)-[:RelatesTo]->(a) "
            "RETURN DISTINCT a.id, b.id LIMIT 20")
        while res.has_next():
            r = res.get_next()
            circular.append({"a": r[0], "b": r[1]})
    trace.circular_refs = circular
    rows["circular_references"] = circular

    # Hallucination risk -> unsupported sentences surfaced directly
    rows["hallucination_risk"] = trace.unsupported_sentences

    # Retrieval dead ends (recorded by the retriever at floor time)
    rows["retrieval_dead_ends"] = trace.retrieval_dead_ends

    # Trust bottlenecks: winning chunks with low authority despite confidence
    conf_step = _step(trace, "confidence")
    breakdown = trace.confidence_breakdown or {}
    bottlenecks = []
    if trace.confidence and trace.confidence > 0.5 and \
            breakdown.get("source_authority", 0.2) < 0.1:  # 0.2*authority < 0.1
        for r in trace.reranked:
            bottlenecks.append({"chunk_id": r["chunk_id"],
                                "reason": "high confidence but low source authority"})
    trace.trust_bottlenecks = bottlenecks
    rows["trust_bottlenecks"] = bottlenecks

    # Citation probability — SIMULATED, clearly labeled; observed only if logged
    trace.citation_probability_simulated = trace.confidence
    rows["citation_probability"] = {
        "simulated (internal heuristic confidence)": trace.confidence,
        "observed": trace.citation_probability_observed,  # None unless real runs logged
    }

    # Recommendations — rule engine per failure pattern, with exact locations
    recs = []
    for name in missing_entities:
        recs.append({"rule": "missing_entity",
                     "action": f'Add a section covering "{name}" — no Entity '
                               f'node exists for it in the site graph.'})
    for w in weak:
        recs.append({"rule": "weak_evidence", "chunk_id": w["chunk_id"],
                     "action": "Add supporting data/citations near this chunk "
                               f"(NLI {w['nli_label']} {w['nli_conf']})."})
    for de in trace.retrieval_dead_ends:
        sub = de["sub_query"] if isinstance(de, dict) else de
        w = f" (fan-out weight {de['weight']})" if isinstance(de, dict) else ""
        recs.append({"rule": "retrieval_dead_end",
                     "action": f'Real content gap: no chunk answers "{sub}"{w} '
                               f"above the score floor. Create content for it."})
    for s in trace.unsupported_sentences:
        recs.append({"rule": "unsupported_answer_sentence",
                     "action": f'Answer sentence "{s[:80]}..." has no source '
                               "support — strengthen or remove the claim on-site."})
    for b in bottlenecks:
        recs.append({"rule": "trust_bottleneck", "chunk_id": b["chunk_id"],
                     "action": "Add author credentials / earn citations for "
                               "this winning-but-low-authority page."})
    rows["recommendations"] = recs
    return rows


def render_markdown(rows: dict) -> str:
    """Render one query's explainability rows as markdown."""
    L = [f"## Query: {rows['query']}", "",
         f"**Answer:** {rows['answer']}", ""]
    L.append(f"**Citations:** {len(rows['why_this_answer']['citations'])} | "
             f"**Citation probability (simulated, heuristic):** "
             f"{rows['citation_probability']['simulated (internal heuristic confidence)']}")
    if rows["hallucination_risk"]:
        L.append("\n### Hallucination risk (unsupported sentences)")
        L += [f"- {s}" for s in rows["hallucination_risk"]]
    if rows["retrieved_but_lost"]:
        L.append("\n### Retrieved but lost (losing scores)")
        L += [f"- {r['chunk_id']}: combined={r['combined']}"
              for r in rows["retrieved_but_lost"][:10]]
    if rows["dropped_with_reasons"]:
        L.append("\n### Dropped along the way (reason logged at drop time)")
        for step, drops in rows["dropped_with_reasons"].items():
            L += [f"- [{step}] {d['reason']}" for d in drops[:10]]
    if rows["retrieval_dead_ends"]:
        L.append("\n### Retrieval dead ends (real content gaps, weighted)")
        for de in rows["retrieval_dead_ends"]:
            if isinstance(de, dict):
                L.append(f"- {de['sub_query']} (weight {de['weight']})")
            else:
                L.append(f"- {de}")
    if rows["missing_entities"]:
        L.append("\n### Missing entities")
        L += [f"- {e}" for e in rows["missing_entities"]]
    if rows["recommendations"]:
        L.append("\n### Recommendations")
        L += [f"- **{r['rule']}**: {r['action']}" for r in rows["recommendations"]]
    return "\n".join(L)
