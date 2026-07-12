"""LAYER 2 — AI Reasoning Simulator (CLAUDE.md).

Per-query pipeline; every node appends a StepTrace to the query's Trace. All
LLM calls go through ``providers.llm_complete`` (local by default, zero keys).

Nodes: intent detection -> query expansion -> hybrid retriever (rank_bm25 +
BGE-M3, 0.4/0.6) -> chunk rerank (bge-reranker-v2-m3, MiniLM fallback) ->
graph traversal (Kùzu 2-hop) -> evidence validation (NLI) -> contradiction
detection (pairwise NLI) -> confidence (heuristic, clamped, with breakdown) ->
answer synthesis (context-only, temp 0) -> citation selection (per-sentence
NLI; unsupported sentences surfaced, never hidden).
"""
from __future__ import annotations

import json as _json
import os
import re
from typing import Optional

from . import providers
from .claims import segment_sentences
from .evidence import _score_pairs  # NLI seam (label order read from config)
from .core.logging import get_logger

logger = get_logger(__name__)

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_FALLBACK = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BM25_WEIGHT, EMBED_WEIGHT = 0.4, 0.6     # CLAUDE.md starting weights
TOP_K_RETRIEVE, TOP_K_RERANK = 20, 5
DEAD_END_FLOOR = 0.25                     # combined-score floor -> content gap
INTENT_CATEGORIES = ["definition", "comparison", "how-to", "list",
                     "fact-lookup", "opinion"]

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        from sentence_transformers import CrossEncoder
        try:
            logger.info(f"Loading reranker {RERANKER_MODEL}...")
            _reranker = CrossEncoder(RERANKER_MODEL)
        except Exception as e:
            logger.warning(f"{RERANKER_MODEL} failed ({e}); falling back to {RERANKER_FALLBACK}")
            _reranker = CrossEncoder(RERANKER_FALLBACK)
    return _reranker


def _tok(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _json_obj(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.startswith("json") else text
    try:
        return _json.loads(text.strip())
    except _json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}") + 1
        if 0 <= s < e:
            try:
                return _json.loads(text[s:e])
            except _json.JSONDecodeError:
                pass
    return {}


def run_query(
    query: str,
    chunks: list[dict],
    chunk_embeddings: list[list[float]],
    kg=None,
    page_signals: Optional[dict] = None,
    trace=None,
):
    """Run one query through Layer 2. Returns the populated Trace."""
    from .trace import Trace
    trace = trace or Trace(query=query)
    by_id = {str(c["chunk_id"]): c for c in chunks}
    emb_by_id = {str(c["chunk_id"]): chunk_embeddings[i]
                 for i, c in enumerate(chunks) if i < len(chunk_embeddings)}

    # ── intent detection ─────────────────────────────────────────────────
    with trace.start_step("intent", query=query) as st:
        raw = providers.llm_complete(
            f"Classify this search query.\nQuery: {query}\n"
            f"Categories: {', '.join(INTENT_CATEGORIES)}\n"
            'Return ONLY JSON: {"category": "<one category>", '
            '"constraints": {"entities": [], "price": null, "date": null}}',
            json=True, temperature=0,
            system="You classify search intents. Output ONLY valid JSON.")
        data = _json_obj(raw)
        cat = data.get("category", "fact-lookup")
        trace.intent = {"query": query,
                        "category": cat if cat in INTENT_CATEGORIES else "fact-lookup",
                        "constraints": data.get("constraints", {}) or {}}
        st.outputs["intent"] = trace.intent

    # ── query expansion — Phase 3c stochastic fan-out (weighted, cached) ──
    fanout_weights: dict[str, float] = {}
    with trace.start_step("query_expansion", query=query,
                          method="stochastic_fanout") as st:
        from .fanout import fanout_distribution
        dist = fanout_distribution(query, use_cache=True)
        trace.expanded_queries = [d["sub_query"] for d in dist]
        fanout_weights = {d["sub_query"]: d["weight"] for d in dist}
        st.outputs["fanout_distribution"] = dist
        st.scores["n_sub_intents"] = float(len(dist))
        if not dist:
            st.note("fan-out produced nothing; retrieving on the original only")

    # ── hybrid retriever: rank_bm25 + BGE cosine, 0.4/0.6 ────────────────
    with trace.start_step("retriever", n_chunks=len(chunks),
                          weights=f"bm25={BM25_WEIGHT}/embed={EMBED_WEIGHT}") as st:
        from rank_bm25 import BM25Okapi
        corpus_ids = [str(c["chunk_id"]) for c in chunks]
        bm25 = BM25Okapi([_tok(c["content"]) for c in chunks])
        queries = [query] + trace.expanded_queries
        qvecs = providers.embed_texts([q[:512] for q in queries])
        best: dict[str, dict] = {}
        for qi, q in enumerate(queries):
            bscores = bm25.get_scores(_tok(q))
            bmax = max(bscores) if len(bscores) and max(bscores) > 0 else 1.0
            qv = qvecs[qi]
            for ci, cid in enumerate(corpus_ids):
                ev = emb_by_id.get(cid)
                cos = sum(a * b for a, b in zip(qv, ev)) if ev else 0.0
                b_norm = float(bscores[ci]) / bmax
                combined = BM25_WEIGHT * b_norm + EMBED_WEIGHT * max(0.0, cos)
                cur = best.get(cid)
                if cur is None or combined > cur["combined"]:
                    best[cid] = {"chunk_id": cid, "bm25": round(b_norm, 4),
                                 "embed": round(max(0.0, cos), 4),
                                 "combined": round(combined, 4),
                                 "url": by_id[cid].get("url", ""),
                                 "matched_query": q}
        ranked = sorted(best.values(), key=lambda r: -r["combined"])[:TOP_K_RETRIEVE]
        trace.retrieved = ranked
        st.outputs["n_candidates"] = len(ranked)
        st.scores["top_combined"] = ranked[0]["combined"] if ranked else 0.0
        # Weighted cluster coverage (Phase 3c): each sub-intent carries a fan-out
        # weight; a covered sub-intent has a chunk above the floor. Dead ends on
        # high-weight sub-intents outrank low-weight ones.
        covered_w = total_w = 0.0
        covered_n = 0
        for q in queries:
            top_q = max((r["combined"] for r in best.values()
                         if r["matched_query"] == q), default=0.0)
            w = fanout_weights.get(q, 0.0) if q != query else 0.0
            total_w += w
            if top_q < DEAD_END_FLOOR:
                trace.retrieval_dead_ends.append(
                    {"sub_query": q, "weight": round(w, 4),
                     "top_combined": round(top_q, 4)})
                st.note(f"dead end (w={w:.2f}): {q!r} top={top_q:.2f} < {DEAD_END_FLOOR}")
            else:
                covered_w += w
                covered_n += 1
        n_intents = len(fanout_weights) or 1
        st.outputs["cluster_coverage"] = {
            "sub_intents_covered": covered_n, "sub_intents_total": len(fanout_weights),
            "weighted_coverage": round(covered_w / total_w, 4) if total_w else None}
        st.scores["weighted_coverage"] = round(covered_w / total_w, 4) if total_w else 0.0

    # ── rerank: bge-reranker-v2-m3 cross-encoder, keep top-5 ─────────────
    with trace.start_step("rerank", n_in=len(trace.retrieved),
                          model=RERANKER_MODEL) as st:
        if trace.retrieved:
            rr = _get_reranker()
            pairs = [(query, by_id[r["chunk_id"]]["content"][:1500])
                     for r in trace.retrieved]
            scores = [float(s) for s in rr.predict(pairs, show_progress_bar=False)]
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            trace.reranked = [
                {"chunk_id": trace.retrieved[i]["chunk_id"],
                 "rerank": round(scores[i], 4)}
                for i in order[:TOP_K_RERANK]]
            for i in order[TOP_K_RERANK:]:
                st.drop({"chunk_id": trace.retrieved[i]["chunk_id"],
                         "rerank": round(scores[i], 4)}, "below top-5 rerank cut")
        st.outputs["kept"] = [r["chunk_id"] for r in trace.reranked]

    # ── graph traversal (Kùzu 2-hop from top chunks' entities) ───────────
    with trace.start_step("graph_traversal", enabled=kg is not None) as st:
        if kg is not None and trace.reranked:
            seed_ids: list[str] = []
            for r in trace.reranked:
                res = kg._exec(
                    "MATCH (c:Chunk {id: $cid})-[:MentionsEntity]->(e:Entity) "
                    "RETURN e.id", {"cid": r["chunk_id"]})
                while res.has_next():
                    seed_ids.append(res.get_next()[0])
            hops = kg.two_hop_entities(list(dict.fromkeys(seed_ids))[:10])
            trace.graph_paths = hops
            extra = {h["chunk_id"] for h in hops} - {r["chunk_id"] for r in trace.reranked}
            st.outputs["graph_supported_chunks"] = sorted(extra)
            st.outputs["n_paths"] = len(hops)
        else:
            st.note("no knowledge graph available — traversal skipped")

    # ── evidence validation (NLI: does each top chunk support the query?) ─
    with trace.start_step("evidence_validation", n_in=len(trace.reranked)) as st:
        survivors = []
        if trace.reranked:
            hyp = f"This text contains an answer to the question: {query}"
            pairs = [(by_id[r["chunk_id"]]["content"][:1500], hyp)
                     for r in trace.reranked]
            for r, (label, conf) in zip(trace.reranked, _score_pairs(pairs)):
                trace.evidence.append({"claim_id": None, "chunk_id": r["chunk_id"],
                                       "nli_label": label, "nli_conf": conf})
                if label == "contradiction":
                    st.drop({"chunk_id": r["chunk_id"], "nli_conf": conf},
                            "chunk contradicts the query premise")
                elif label == "neutral" and conf > 0.9:
                    st.drop({"chunk_id": r["chunk_id"], "nli_conf": conf},
                            "no evidence relation to query (high-confidence neutral)")
                else:
                    survivors.append(r)
            if not survivors and trace.reranked:
                survivors = [trace.reranked[0]]
                st.note("NLI dropped everything; kept best reranked chunk "
                        "(recorded, not hidden)")
        st.outputs["survivors"] = [s["chunk_id"] for s in survivors]

    # ── contradiction detection (pairwise NLI among survivors) ───────────
    with trace.start_step("contradiction_detection", n_in=len(survivors)) as st:
        if len(survivors) > 1:
            texts = [by_id[s["chunk_id"]]["content"][:1000] for s in survivors]
            ids = [s["chunk_id"] for s in survivors]
            pairs, meta = [], []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    pairs.append((texts[i], texts[j]))
                    meta.append((ids[i], ids[j]))
            for (a, b), (label, conf) in zip(meta, _score_pairs(pairs)):
                if label == "contradiction":
                    trace.contradictions.append(
                        {"chunk_a": a, "chunk_b": b, "nli_conf": conf})
        st.outputs["contradictions"] = trace.contradictions
        if trace.contradictions:
            st.note("conflicting sources flagged explicitly — not silently resolved")

    # ── confidence (heuristic, clamped, logged with breakdown) ────────────
    with trace.start_step("confidence") as st:
        sigs = page_signals or {}
        top = survivors[0] if survivors else None
        retr = next((r["combined"] for r in trace.retrieved
                     if top and r["chunk_id"] == top["chunk_id"]), 0.0)
        ent_conf = max((e["nli_conf"] for e in trace.evidence
                        if e["nli_label"] == "entailment"), default=0.0)
        url = by_id[top["chunk_id"]].get("url", "") if top else ""
        authority = sigs.get(url, {}).get("source_authority", 0.5)
        freshness = sigs.get(url, {}).get("temporal_freshness", 0.5)
        if not sigs:
            st.note("no page signals provided — authority/freshness neutral 0.5")
        penalty = 0.3 if trace.contradictions else 0.0
        breakdown = {"combined_retrieval": round(0.35 * retr, 4),
                     "evidence_entailment": round(0.25 * ent_conf, 4),
                     "source_authority": round(0.2 * authority, 4),
                     "temporal_freshness": round(0.2 * freshness, 4),
                     "contradiction_penalty": -penalty}
        trace.confidence = max(0.0, min(1.0, sum(breakdown.values())))
        trace.confidence_breakdown = breakdown
        st.scores["confidence"] = trace.confidence  # heuristic
        st.outputs["breakdown"] = breakdown

    # ── answer synthesis (context-only, temp 0) ──────────────────────────
    with trace.start_step("answer_synthesis", n_sources=len(survivors)) as st:
        if not survivors:
            trace.answer = "No relevant content found in the indexed pages."
        else:
            ctx = "\n\n".join(
                f"[chunk {s['chunk_id']}] {by_id[s['chunk_id']]['content'][:1500]}"
                for s in survivors)
            trace.answer = providers.llm_complete(
                f"Answer the question using ONLY the provided context. If the "
                f"context does not contain the answer, say you don't know. Do "
                f"not use outside knowledge.\n\nContext:\n{ctx}\n\n"
                f"Question: {query}\nAnswer:",
                temperature=0,
                system="You answer strictly from provided context.").strip()
        st.outputs["answer_len"] = len(trace.answer)

    # ── citation selection (per-sentence NLI; unsupported surfaced) ───────
    with trace.start_step("citation_selection") as st:
        sents = segment_sentences(trace.answer) if survivors else []
        for s in sents:
            pairs = [(by_id[sv["chunk_id"]]["content"][:1500], s["text"])
                     for sv in survivors]
            scored = _score_pairs(pairs)
            best_i = max(range(len(scored)), key=lambda i: (
                scored[i][0] == "entailment", scored[i][1]), default=None)
            if best_i is not None and scored[best_i][0] == "entailment":
                cid = survivors[best_i]["chunk_id"]
                trace.citations.append({
                    "sentence_span": [s["char_start"], s["char_end"]],
                    "chunk_id": cid, "page_url": by_id[cid].get("url", "")})
            else:
                trace.unsupported_sentences.append(s["text"])
        st.outputs["n_citations"] = len(trace.citations)
        st.outputs["unsupported_sentences"] = trace.unsupported_sentences
        if trace.unsupported_sentences:
            st.note("hallucination risk: answer sentences without source support")

    return trace
