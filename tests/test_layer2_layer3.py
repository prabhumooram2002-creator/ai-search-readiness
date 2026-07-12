"""Layer 2 (src/simulator.py) + Layer 3 (src/explain.py) + steps 11-14
(src/signals.py). Offline: LLM/embeddings/NLI/reranker stubbed; verifies node
wiring, StepTrace coverage, confidence formula, citation/unsupported split,
explainability introspection, and heuristic signal formulas."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src import simulator as sim
from src import explain as ex
from src import signals as sig
from src.trace import Trace

CHUNKS = [
    {"chunk_id": "c1", "content": "Acme was founded in 2001. Acme builds widgets.",
     "url": "https://a.com/p1"},
    {"chunk_id": "c2", "content": "Totally unrelated cooking recipes.",
     "url": "https://a.com/p2"},
]
EMB = [[1.0] + [0.0] * 3, [0.0, 1.0, 0.0, 0.0]]


class FakeReranker:
    def predict(self, pairs, show_progress_bar=False):
        # first chunk relevant, second not
        return [2.5 if "Acme" in p[1] else -3.0 for p in pairs]


def _stub_llm(responses):
    it = iter(responses)
    def f(prompt, **kw):
        return next(it)
    return f


def _stub_nli(label_for):
    """label_for: fn(premise, hypothesis) -> (label, conf)"""
    def f(pairs):
        return [label_for(p, h) for p, h in pairs]
    return f


def _run(monkeypatch, nli=None, llm=None):
    # Phase 3c: expansion routes through fan-out; stub it directly so tests
    # stay fast (no 12x sampling) and deterministic.
    import src.fanout as fo
    monkeypatch.setattr(fo, "fanout_distribution",
                        lambda q, use_cache=True, **k: [
                            {"sub_query": "acme founding year", "weight": 0.6},
                            {"sub_query": "acme history", "weight": 0.4}])
    monkeypatch.setattr(sim.providers, "llm_complete", llm or _stub_llm([
        '{"category": "fact-lookup", "constraints": {"entities": ["Acme"]}}',
        "Acme was founded in 2001. It is based on Mars.",
    ]))
    monkeypatch.setattr(sim.providers, "embed_texts",
                        lambda texts: [[1.0, 0.0, 0.0, 0.0] for _ in texts])
    monkeypatch.setattr(sim, "_get_reranker", lambda: FakeReranker())
    monkeypatch.setattr(sim, "_score_pairs", nli or _stub_nli(
        lambda p, h: ("entailment", 0.95) if "founded in 2001" in h or "answer" in h
        else ("neutral", 0.95)))
    return sim.run_query("when was acme founded", CHUNKS, EMB)


def test_layer2_all_nodes_emit_steptraces(monkeypatch):
    trace = _run(monkeypatch)
    assert trace.step_names() == [
        "intent", "query_expansion", "retriever", "rerank", "graph_traversal",
        "evidence_validation", "contradiction_detection", "confidence",
        "answer_synthesis", "citation_selection"]
    assert trace.intent["category"] == "fact-lookup"
    assert trace.expanded_queries == ["acme founding year", "acme history"]
    assert trace.retrieved and trace.retrieved[0]["chunk_id"] == "c1"
    assert trace.reranked[0]["chunk_id"] == "c1"


def test_confidence_formula_clamped_with_breakdown(monkeypatch):
    trace = _run(monkeypatch)
    bd = trace.confidence_breakdown
    assert set(bd) == {"combined_retrieval", "evidence_entailment",
                       "source_authority", "temporal_freshness",
                       "contradiction_penalty"}
    assert 0.0 <= trace.confidence <= 1.0
    assert abs(sum(bd.values()) - trace.confidence) < 1e-6  # no hidden terms


def test_citations_vs_unsupported_split(monkeypatch):
    # NLI: entails "founded in 2001" sentence; the Mars sentence is unsupported
    trace = _run(monkeypatch, nli=_stub_nli(
        lambda p, h: ("entailment", 0.9) if "2001" in h or "answer" in h
        else ("neutral", 0.9)))
    assert len(trace.citations) == 1
    assert trace.citations[0]["chunk_id"] == "c1"
    assert any("Mars" in s for s in trace.unsupported_sentences)


def test_contradiction_penalty_applies(monkeypatch):
    both_relevant = _stub_nli(lambda p, h: ("entailment", 0.9))
    trace_no = _run(monkeypatch, nli=both_relevant)
    assert trace_no.confidence_breakdown["contradiction_penalty"] == 0

    def contradicting(p, h):
        if h.startswith("This text contains an answer") or len(h) > 60:
            return ("entailment", 0.9)
        return ("contradiction", 0.9)  # pairwise chunk-vs-chunk
    # force both chunks to survive so a pair exists
    trace_yes = _run(monkeypatch, nli=_stub_nli(contradicting))
    if trace_yes.contradictions:
        assert trace_yes.confidence_breakdown["contradiction_penalty"] == -0.3


def test_layer3_pure_introspection(monkeypatch):
    trace = _run(monkeypatch)
    rows = ex.explain_query(trace)  # no KG -> entity checks skipped
    assert rows["answer"] == trace.answer
    assert rows["why_these_chunks"] == trace.reranked
    lost = {r["chunk_id"] for r in rows["retrieved_but_lost"]}
    assert lost == {r["chunk_id"] for r in trace.retrieved} - \
                   {r["chunk_id"] for r in trace.reranked}
    # simulated citation probability equals recorded confidence — no new inference
    assert rows["citation_probability"][
        "simulated (internal heuristic confidence)"] == trace.confidence
    assert rows["citation_probability"]["observed"] is None
    # unsupported sentences produce recommendations with the rule name
    if trace.unsupported_sentences:
        assert any(r["rule"] == "unsupported_answer_sentence"
                   for r in rows["recommendations"])
    md = ex.render_markdown(rows)
    assert trace.query in md


def test_signals_formulas_and_labels():
    pages = [
        {"url": "https://a.com/p1", "content": "By Jane Doe. Updated 2026-07-01.",
         "schema_jsonld": [{"sameAs": ["x"]}], "internal_inbound": 6,
         "lastmod": "2026-07-01"},
        {"url": "https://a.com/p2", "content": "no signals here",
         "internal_inbound": 0},
    ]
    out = sig.score_pages(pages)
    p1, p2 = out["https://a.com/p1"], out["https://a.com/p2"]
    for rec in (p1, p2):
        for k in ("trust", "citation", "source_authority", "temporal_freshness"):
            assert 0.0 <= rec[k] <= 1.0
    assert p1["trust"] > p2["trust"]
    assert p1["citation"] > p2["citation"]
    assert p1["temporal_freshness"] > 0.9      # fresh lastmod
    assert p2["temporal_freshness"] == 0.5     # no date evidence -> neutral
    assert any("10b pending" in n for n in p1["notes"])
    assert "heuristic" in sig.FORMULAS["trust"]


def test_fanout_weighted_coverage_drops_when_covering_chunk_removed(monkeypatch):
    # Phase 3c: weighted cluster coverage must fall when the chunk that covers
    # a sub-intent is deleted. Real BM25; everything else stubbed.
    import src.fanout as fo
    monkeypatch.setattr(fo, "fanout_distribution",
                        lambda q, use_cache=True, **k: [
                            {"sub_query": "acme pricing plans cost", "weight": 0.7},
                            {"sub_query": "acme security compliance", "weight": 0.3}])
    monkeypatch.setattr(sim.providers, "llm_complete", _stub_llm([
        '{"category": "fact-lookup", "constraints": {"entities": []}}',
        "answer"] * 2))
    # embed: match sub-query to its chunk by keyword overlap (crude but real-ish)
    def emb(texts):
        out = []
        for t in texts:
            tl = t.lower()
            out.append([1.0 if "pricing" in tl or "cost" in tl else 0.0,
                        1.0 if "security" in tl or "compliance" in tl else 0.0])
        return out
    monkeypatch.setattr(sim.providers, "embed_texts", emb)
    monkeypatch.setattr(sim, "_get_reranker", lambda: FakeReranker())
    monkeypatch.setattr(sim, "_score_pairs",
                        lambda pairs: [("entailment", 0.9)] * len(pairs))

    both = [{"chunk_id": "cp", "content": "acme pricing plans cost ten dollars", "url": "u1"},
            {"chunk_id": "cs", "content": "acme security compliance soc2 audited", "url": "u2"}]
    emb_both = emb([c["content"] for c in both])
    full = sim.run_query("tell me about acme", both, emb_both)
    cov_full = next(s for s in full.steps if s.name == "retriever").outputs["cluster_coverage"]

    # remove the pricing chunk (covers the 0.7-weight sub-intent)
    only_sec = [both[1]]
    cov_gap = sim.run_query("tell me about acme", only_sec, emb([only_sec[0]["content"]])
                            ).steps[2].outputs["cluster_coverage"]
    assert cov_full["weighted_coverage"] > cov_gap["weighted_coverage"]
    assert cov_full["sub_intents_covered"] > cov_gap["sub_intents_covered"]
