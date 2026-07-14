"""Phase 12 final assembly: build_intelligence_report() wiring + html_renderer.py.
Integration-style test over a comprehensive fake KG/ctx -- catches the kind
of wiring bugs (wrong import path, wrong query shape) unit tests on
individual sections can't."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.intelligence import report as rpt
from src.intelligence import html_renderer as hr
from src.trace import Trace


class _FakeResult:
    def __init__(self, rows):
        self._rows = list(rows)
        self._i = 0
    def has_next(self):
        return self._i < len(self._rows)
    def get_next(self):
        row = self._rows[self._i]; self._i += 1; return row


# Query -> canned rows, matched by a distinguishing substring so one fake
# KG object can serve every section's differently-shaped query.
_QUERY_ROWS = [
    ("OPTIONAL MATCH (e)<-[m:MentionsEntity]-(c:Chunk)",
     [("e1", "Acme", "Organization", 0.9, "chunk-zzz1", "https://site.com/a")]),
    ("RelatesTo]->(b:Entity) RETURN a.id, b.id\n", []),  # graph_stats degree pass (entities.py)
    ("RETURN a.id, a.name, r.relation", []),  # relationships.relationship_triplets
    ("MATCH (a:Entity)-[:RelatesTo]->(b:Entity) RETURN a.id, b.id", []),
    ("MATCH (e:Entity) RETURN e.id", [("e1",)]),
    ("BelongsToTopic]->(t:Topic) RETURN t.id, t.label, c.id, p.url", []),
    ("MentionsEntity]->(e:Entity) RETURN t.id, e.id", []),
    ("d:ExternalDomain", []),
    ("SupportsClaim]->(cl:Claim)", []),
    ("BelongsToTopic]->(t:Topic) RETURN c.id, t.label", []),
]


class _FakeKG:
    def _exec(self, query, params=None):
        for marker, rows in _QUERY_ROWS:
            if marker in query:
                return _FakeResult(rows)
        return _FakeResult([])  # unmatched query -> empty, never crash


def _make_trace(query="noodles healthy?"):
    return Trace(query=query, answer="answer text", confidence=0.7,
                retrieved=[{"chunk_id": "chunk-zzz1", "combined": 0.8}],
                reranked=[{"chunk_id": "chunk-zzz1", "rerank": 1.0}],
                citations=[{"page_url": "https://site.com/a", "chunk_id": "chunk-zzz1"}],
                evidence=[{"nli_label": "entailment", "nli_conf": 0.9}])


def test_build_intelligence_report_all_19_sections_present(monkeypatch):
    from src import providers
    monkeypatch.setattr(providers, "embed_texts", lambda texts: [[1.0, 0.0]] * len(texts))

    ctx = {
        "kg": _FakeKG(),
        "l0_pages": [{"url": "https://site.com/a", "content": "Some page content.",
                     "internal_links": [], "schema_jsonld": [], "access_gaps": []}],
        "chunks": [{"chunk_id": "chunk-zzz1", "content": "chunk text", "url": "https://site.com/a"}],
        "page_signals": {"https://site.com/a": {"source_authority": 0.5,
                                                "temporal_freshness": 0.5}},
        "embeddings": [[1.0, 0.0]],
        "site_report": {"url": "https://site.com", "sitemap": {}},
        "invisibility": {"worst_pages": []},
    }
    traces = [_make_trace()]
    explanations = [{"recommendations": [
        {"rule": "missing_entity", "finding_id": "entity:x", "action": "a"}]}]
    struct = [{"chunk_id": "chunk-zzz1", "structure_score": 0.7, "flags": []}]

    payload = rpt.build_intelligence_report(
        ctx, traces, explanations, struct, fx=None, query_weights={},
        skipped_queries=[{"query": "off-brand"}])

    for key, _ in hr.SECTION_TITLES:
        assert key in payload, f"missing {key}"
    assert payload["section_13_query_simulation"]["n_queries"] == 1
    assert payload["section_13_query_simulation"]["skipped_irrelevant"] == [{"query": "off-brand"}]
    assert len(payload["section_14_reasoning_paths"]) == 1


def test_claim_embedding_provider_outage_degrades_not_crashes(monkeypatch):
    """Real bug found running the actual verification: an NVIDIA API 500
    (surviving embedder.py's own 4-attempt retry) crashed the whole report.
    build_intelligence_report() must catch that and degrade section 11/12
    to 'no embedding available' rather than take down every other section."""
    from src import providers

    def failing_embed(texts):
        raise RuntimeError("NVIDIA NIM embed failed after 4 attempts: 500")
    monkeypatch.setattr(providers, "embed_texts", failing_embed)

    class _KGWithOneClaim(_FakeKG):
        def _exec(self, query, params=None):
            if "SupportsClaim]->(cl:Claim)" in query:
                return _FakeResult([("chunk-zzz1", "cl1", "Noodles cost Rs. 240.", "entailment", 0.9)])
            return super()._exec(query, params)

    ctx = {
        "kg": _KGWithOneClaim(),
        "l0_pages": [{"url": "https://site.com/a", "content": "text",
                     "internal_links": [], "schema_jsonld": [], "access_gaps": []}],
        "chunks": [{"chunk_id": "chunk-zzz1", "content": "chunk", "url": "https://site.com/a"}],
        "page_signals": {}, "embeddings": [[1.0, 0.0]],
        "site_report": {"url": "https://site.com", "sitemap": {}},
        "invisibility": {"worst_pages": []},
    }
    payload = rpt.build_intelligence_report(
        ctx, [_make_trace()], [{"recommendations": []}],
        [{"chunk_id": "chunk-zzz1", "structure_score": 0.5, "flags": []}],
        fx=None, query_weights={}, skipped_queries=None)

    records = payload["section_11_12_claims"]["records"]
    assert len(records) == 1
    assert records[0]["note"] == "no embedding available"
    # every other section still assembled despite the embedding failure
    for key, _ in hr.SECTION_TITLES:
        assert key in payload


def test_render_intelligence_html_has_sidebar_and_search(monkeypatch):
    from src import providers
    monkeypatch.setattr(providers, "embed_texts", lambda texts: [[1.0, 0.0]] * len(texts))
    ctx = {
        "kg": _FakeKG(),
        "l0_pages": [{"url": "https://site.com/a", "content": "text",
                     "internal_links": [], "schema_jsonld": [], "access_gaps": []}],
        "chunks": [{"chunk_id": "chunk-zzz1", "content": "chunk", "url": "https://site.com/a"}],
        "page_signals": {}, "embeddings": [[1.0, 0.0]],
        "site_report": {"url": "https://site.com", "sitemap": {}},
        "invisibility": {"worst_pages": []},
    }
    payload = rpt.build_intelligence_report(
        ctx, [_make_trace()], [{"recommendations": []}],
        [{"chunk_id": "chunk-zzz1", "structure_score": 0.5, "flags": []}],
        fx=None, query_weights={}, skipped_queries=None)
    html = hr.render_intelligence_html(payload)
    assert 'id="searchbox"' in html
    assert 'href="#section_1_identity"' in html
    assert "</html>" in html
    # 19 sections all got a heading
    for key, _ in hr.SECTION_TITLES:
        assert f'id="{key}"' in html


def test_render_intelligence_html_never_prints_bare_ids(monkeypatch):
    """Brief hard rule: 'bare chunk/entity ids are a build failure'. A row
    with a chunk_id/entity_id must render as a link + snippet, never the
    raw opaque id string, found via hand-verification against a real run
    (section 3's relationship triplets rendered '042833ef040abf9f' as
    plain text with no link)."""
    from src import providers
    monkeypatch.setattr(providers, "embed_texts", lambda texts: [[1.0, 0.0]] * len(texts))

    class _KGWithTriplet(_FakeKG):
        def _exec(self, query, params=None):
            if "RETURN a.id, a.name, r.relation" in query:
                return _FakeResult([("e1", "Acme", "founded by", "e9", "Dario",
                                    0.9, "chunk-zzz1")])
            return super()._exec(query, params)

    ctx = {
        "kg": _KGWithTriplet(),
        "l0_pages": [{"url": "https://site.com/a", "content": "Some page content.",
                     "internal_links": [], "schema_jsonld": [], "access_gaps": []}],
        "chunks": [{"chunk_id": "chunk-zzz1", "content": "Acme was founded by Dario in 2020.",
                   "url": "https://site.com/a"}],
        "page_signals": {}, "embeddings": [[1.0, 0.0]],
        "site_report": {"url": "https://site.com", "sitemap": {}},
        "invisibility": {"worst_pages": []},
    }
    payload = rpt.build_intelligence_report(
        ctx, [_make_trace()], [{"recommendations": []}],
        [{"chunk_id": "chunk-zzz1", "structure_score": 0.5, "flags": []}],
        fx=None, query_weights={}, skipped_queries=None)

    assert payload["section_3_relationships"]["triplets"][0]["source_chunk_id"] == "chunk-zzz1"

    html = hr.render_intelligence_html(payload)
    assert "chunk-zzz1" not in html  # bare chunk id never appears in the rendered report
    assert "e9" not in html  # bare entity id (object_id) never appears either
    assert "Acme was founded by Dario" in html  # resolved snippet instead
    assert 'href="https://site.com/a"' in html
