"""PHASE 10a — the impact chain (src/impact.py). Verifies: content-fix
findings get a real predicted coverage delta via a stubbed what-if; technical
fixes (jsonld/llms_txt) get a qualitative note, never a fabricated delta;
findings with no matching fix artifact are left untouched; results are
cached per (finding_id, fix_content_hash)."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src import impact


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


def test_content_fix_gets_predicted_coverage_delta(tmp_path, monkeypatch):
    faq_path = _write(tmp_path, "faq_pricing.md", "## Pricing\nOur plans start at $10/month.")
    recs = [{"rule": "retrieval_dead_end", "finding_id": "deadend:pricing-tiers",
            "action": "gap"}]
    manifest = [{"finding_id": "deadend:pricing-tiers", "type": "faq_draft",
                "path": faq_path, "valid": True}]

    def fake_whatif(draft_path, queries, base_chunks, base_embs, persist_paths=None, trace=None):
        assert draft_path == faq_path
        return {"per_query": [{"coverage_before": 0.3, "coverage_after": 0.75,
                               "coverage_delta": 0.45, "newly_covered": 2}]}
    monkeypatch.setattr("src.whatif.whatif", fake_whatif)

    out = impact.attach_predicted_impact(
        recs, manifest, "what are your pricing tiers", [], [],
        cache_path=tmp_path / "cache.json")
    imp = out[0]["predicted_impact"]
    assert imp["kind"] == "coverage"
    assert imp["delta"] == 0.45
    assert imp["label"] == "predicted (simulated)"


def test_technical_fix_gets_qualitative_note_not_a_fabricated_delta(tmp_path, monkeypatch):
    jsonld_path = _write(tmp_path, "schema.jsonld", "{}")
    recs = [{"rule": "missing_entity", "finding_id": "schema:site-com-pricing",
            "action": "gap"}]
    manifest = [{"finding_id": "schema:site-com-pricing", "type": "jsonld",
                "path": jsonld_path, "valid": True}]

    def fail_whatif(*a, **kw):
        raise AssertionError("technical fixes must not go through the coverage overlay")
    monkeypatch.setattr("src.whatif.whatif", fail_whatif)

    out = impact.attach_predicted_impact(
        recs, manifest, "q", [], [], cache_path=tmp_path / "cache.json")
    imp = out[0]["predicted_impact"]
    assert imp["kind"] == "technical"
    assert "delta" not in imp
    assert imp["label"] == "predicted (simulated)"


def test_finding_with_no_fix_artifact_is_untouched(tmp_path):
    recs = [{"rule": "trust_bottleneck", "finding_id": "structure:xyz", "action": "gap"}]
    out = impact.attach_predicted_impact(
        recs, [], "q", [], [], cache_path=tmp_path / "cache.json")
    assert "predicted_impact" not in out[0]


def test_impact_is_cached_by_finding_and_fix_hash(tmp_path, monkeypatch):
    faq_path = _write(tmp_path, "faq.md", "content v1")
    recs = [{"rule": "retrieval_dead_end", "finding_id": "deadend:x", "action": "gap"}]
    manifest = [{"finding_id": "deadend:x", "type": "faq_draft", "path": faq_path, "valid": True}]
    cache_path = tmp_path / "cache.json"

    calls = {"n": 0}
    def fake_whatif(*a, **kw):
        calls["n"] += 1
        return {"per_query": [{"coverage_before": 0.1, "coverage_after": 0.2,
                               "coverage_delta": 0.1, "newly_covered": 1}]}
    monkeypatch.setattr("src.whatif.whatif", fake_whatif)

    impact.attach_predicted_impact(recs, manifest, "q", [], [], cache_path=cache_path)
    assert calls["n"] == 1
    # second call, same fix content -> cache hit, no recompute
    impact.attach_predicted_impact(recs, manifest, "q", [], [], cache_path=cache_path)
    assert calls["n"] == 1

    # fix content changes -> cache miss, recompute
    Path(faq_path).write_text("content v2 — different", encoding="utf-8")
    impact.attach_predicted_impact(recs, manifest, "q", [], [], cache_path=cache_path)
    assert calls["n"] == 2


def test_whatif_failure_is_caught_not_fatal(tmp_path, monkeypatch):
    faq_path = _write(tmp_path, "faq.md", "content")
    recs = [{"rule": "retrieval_dead_end", "finding_id": "deadend:x", "action": "gap"}]
    manifest = [{"finding_id": "deadend:x", "type": "faq_draft", "path": faq_path, "valid": True}]

    def broken_whatif(*a, **kw):
        raise RuntimeError("boom")
    monkeypatch.setattr("src.whatif.whatif", broken_whatif)

    out = impact.attach_predicted_impact(
        recs, manifest, "q", [], [], cache_path=tmp_path / "cache.json")
    assert out[0]["predicted_impact"]["kind"] == "unknown"
