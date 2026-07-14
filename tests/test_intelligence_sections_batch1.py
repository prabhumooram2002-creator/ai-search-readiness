"""Phase 12 sections 3,4,5,8,9,10,13,14,16 — light coverage over fakes/fixtures,
consistent with the shared-utils test pattern."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.intelligence import (relationships, topics_intel, chunks_intel,
                              crawl_intel, crawlability, kg_quality,
                              query_sim, reasoning_path, content_intel)
from src.trace import Trace


class _FakeResult:
    def __init__(self, rows):
        self._rows = list(rows)
        self._i = 0
    def has_next(self):
        return self._i < len(self._rows)
    def get_next(self):
        row = self._rows[self._i]; self._i += 1; return row


class _FakeKG:
    def __init__(self, *row_sets):
        self.row_sets = list(row_sets)
        self._n = 0
    def _exec(self, query, params=None):
        rows = self.row_sets[self._n] if self._n < len(self.row_sets) else []
        self._n += 1
        return _FakeResult(rows)


# ── Section 3 ────────────────────────────────────────────────────────────
def test_relationship_triplets_and_weak_edge_finding():
    kg = _FakeKG([("e1", "Acme", "founded by", "e2", "Dario", 1, 0.4)])
    triplets = relationships.relationship_triplets(kg)
    assert triplets[0]["relation"] == "founded by"
    findings = relationships.relationship_findings(triplets)
    assert findings["n_weak"] == 1
    assert findings["missing_edges"]["available"] is False


def test_render_top_entities_svg_is_valid_svg(monkeypatch):
    kg = _FakeKG([("e1", "e2")], [("e1",), ("e2",)], [("e1", "Acme"), ("e2", "Dario")])
    svg = relationships.render_top_entities_svg(kg, top_n=10)
    assert svg.startswith("<svg")
    assert "</svg>" in svg


# ── Section 4 ────────────────────────────────────────────────────────────
def test_topic_stats_roles_and_share():
    kg = _FakeKG(
        [("t1", "Pricing", "c1", "u1"), ("t1", "Pricing", "c2", "u1"),
         ("t2", "Support", "c3", "u2")],
        [("t1", "e1")])
    out = topics_intel.topic_stats(kg, page_signals={})
    labels = {t["label"]: t for t in out["topics"]}
    assert labels["Pricing"]["role"] in ("primary", "secondary", "supporting")
    assert out["n_topics"] == 2


# ── Section 5 ────────────────────────────────────────────────────────────
def test_chunk_dossiers_won_lost_and_entities():
    t = Trace(query="q")
    t.retrieved = [{"chunk_id": "c1", "combined": 0.9}, {"chunk_id": "c2", "combined": 0.5}]
    t.reranked = [{"chunk_id": "c1", "rerank": 2.0}]
    chunks = [{"chunk_id": "c1", "content": "text", "url": "u1"},
             {"chunk_id": "c2", "content": "text2", "url": "u2"}]
    entities = [{"name": "Acme", "mentions": [{"chunk_id": "c1"}]}]
    out = chunks_intel.chunk_dossiers(chunks, entities, {}, {}, {}, {}, [t])
    by_id = {d["chunk_id"]: d for d in out}
    assert by_id["c1"]["queries_won"] == ["q"]
    assert by_id["c2"]["queries_lost"] == ["q"]
    assert by_id["c1"]["entities"] == ["Acme"]


# ── Section 8 ────────────────────────────────────────────────────────────
def test_crawl_manifest_dead_pages_and_duplicates():
    class R:
        def __init__(self, url, status, success, depth):
            self.url, self.status, self.success, self.depth = url, status, success, depth
    results = [R("u1", 200, True, 0), R("u2", 404, False, 1)]
    pages = [{"url": "u1", "markdown_hash": "h1"}, {"url": "u3", "markdown_hash": "h1"}]
    out = crawl_intel.crawl_manifest({"sitemap_urls": 10, "captured": 9, "capture_rate": 0.9},
                                     pages, results)
    assert out["dead_pages"] == ["u2"]
    assert len(out["duplicate_url_groups"]) == 1
    assert out["canonical_conflicts"]["available"] is False


# ── Section 9 ────────────────────────────────────────────────────────────
def test_crawlability_report_no_llms_txt_note():
    inv = {"worst_pages": [{"url": "u1", "per_bot": {"GPTBot": 1.0}}]}
    out = crawlability.crawlability_report(inv, [{"url": "u1"}], {}, None)
    assert out["llms_txt"]["present"] is False
    assert out["per_bot_table"][0]["bot"] == "GPTBot"


# ── Section 10 ───────────────────────────────────────────────────────────
def test_kg_quality_metrics_basic():
    kg = _FakeKG([("e1", "e2")], [("e1",), ("e2",), ("e3",)])
    entities = [{"id": "e1", "name": "Acme", "type": "Org"},
               {"id": "e2", "name": "Acme Inc", "type": "Org"},
               {"id": "e3", "name": "Zeta", "type": "Org"}]
    out = kg_quality.kg_quality_metrics(kg, entities, claims_count=10)
    assert out["n_isolated_nodes"] == 1  # e3
    assert out["completeness"]["label"] == "heuristic"
    assert len(out["duplicate_entity_candidates"]) >= 1  # Acme / Acme Inc


# ── Section 13 ───────────────────────────────────────────────────────────
def test_full_query_dossiers_includes_skipped():
    t = Trace(query="q", answer="a")
    out = query_sim.full_query_dossiers([t], skipped_queries=[{"query": "off-brand"}])
    assert out["n_queries"] == 1
    assert out["skipped_irrelevant"] == [{"query": "off-brand"}]


# ── Section 14 ───────────────────────────────────────────────────────────
def test_reasoning_path_is_byte_identical_for_same_trace():
    t = Trace(query="when?", intent={"category": "fact-lookup"},
             expanded_queries=["a", "b"],
             retrieved=[{"chunk_id": "c1"}] * 3, reranked=[{"chunk_id": "c1"}],
             evidence=[{"nli_label": "entailment"}], confidence=0.71,
             confidence_breakdown={"combined_retrieval": 0.5, "evidence_entailment": 0.21},
             citations=[{"page_url": "u1", "chunk_id": "c1"}])
    n1 = reasoning_path.render_reasoning_path(t)
    n2 = reasoning_path.render_reasoning_path(t)
    assert n1 == n2
    assert "when?" in n1 and "0.71" in n1


# ── Section 16 ───────────────────────────────────────────────────────────
def test_page_content_intel_computes_completeness_and_citation_potential():
    out = content_intel.page_content_intel(
        "u1", "This is a simple readable page about noodles and pasta.",
        n_claims=2, n_entities=3, structure_score=0.8, unsupported_rate=0.1,
        sub_intents_covered=3, sub_intents_total=5, freshness=0.9)
    assert out["answer_completeness"] == 0.6
    assert out["citation_potential"]["label"] == "heuristic"
    assert out["readability_flesch"] is not None


def test_near_duplicate_chunks_finds_similar_pairs():
    chunks = [{"chunk_id": "c1"}, {"chunk_id": "c2"}, {"chunk_id": "c3"}]
    embs = [[1.0, 0.0], [1.0, 0.001], [0.0, 1.0]]
    out = content_intel.near_duplicate_chunks(chunks, embs, threshold=0.99)
    assert {"c1", "c2"} == {out[0]["a"], out[0]["b"]}
