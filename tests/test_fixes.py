"""Phase 8 — fix generation (src/fixes.py). LLM stubbed; JSON-LD/llms.txt real."""
import sys
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src import fixes as fx
from src.trace import Trace


def test_generated_jsonld_validates():
    art = fx.generate_jsonld({"url": "https://a.com/p", "title": "Pricing Guide",
                              "content": "How our pricing works."})
    ok, errs = fx.validate_jsonld(art)
    assert ok, errs
    assert art["@context"] == "https://schema.org" and art["@type"] == "Article"


def test_faq_page_when_question_headings():
    art = fx.generate_jsonld({"url": "https://a.com/faq", "title": "FAQ",
                              "headings": [{"level": 2, "text": "What is X?"},
                                           {"level": 2, "text": "How much does X cost?"}]})
    ok, errs = fx.validate_jsonld(art)
    assert ok, errs
    assert art["@type"] == "FAQPage" and len(art["mainEntity"]) == 2


def test_validator_rejects_bad_blocks():
    bad_ctx = {"@context": "http://example.com", "@type": "Article",
               "headline": "x", "url": "u"}
    ok, errs = fx.validate_jsonld(bad_ctx)
    assert not ok and any("@context" in e for e in errs)

    missing = {"@context": "https://schema.org", "@type": "WebPage"}
    ok, errs = fx.validate_jsonld(missing)
    assert not ok and any("requires 'name'" in e for e in errs)


def test_llms_txt_only_real_urls():
    txt = fx.generate_llms_txt(
        "Acme", pages=[{"url": "https://a.com/1", "title": "One", "pagerank": 0.5},
                       {"url": "javascript:void(0)", "title": "Bad"},
                       {"url": "/relative", "title": "AlsoBad"}],
        topics=[{"label": "Pricing"}], description="Acme site")
    assert "https://a.com/1" in txt
    assert "javascript:void(0)" not in txt and "/relative" not in txt
    assert "## Topics" in txt and "Pricing" in txt
    assert fx.DRAFT_BANNER in txt


def test_generate_fixes_manifest_maps_findings(tmp_path, monkeypatch):
    monkeypatch.setattr(fx.providers, "llm_complete",
                        lambda p, **k: '{"heading":"Refunds","qa":[{"q":"?","a":"."}]}'
                        if k.get("json") else "rewritten table")
    trace = Trace(query="q")
    res = fx.generate_fixes(
        7, fixes_dir=tmp_path,
        pages_without_schema=[{"url": "https://a.com/p", "title": "P", "content": "c"}],
        site={"name": "Acme", "pages": [{"url": "https://a.com/p", "title": "P",
                                         "pagerank": 0.5}],
              "topics": [{"label": "Pricing"}]},
        dead_ends=[{"sub_query": "refund policy", "source_claims": ["30 day returns"]}],
        structure_flagged=[{"chunk_id": "c1", "text": "wall", "flags":
                            [{"flag": "paragraph_too_long"}]}],
        trace=trace)
    man = res["manifest"]
    types = {m["type"] for m in man}
    assert types == {"jsonld", "llms_txt", "faq_draft", "block_rewrite"}
    # every artifact exists on disk and is referenced by a finding id
    for m in man:
        assert Path(m["path"]).exists()
        assert m["finding_id"]
    # emitted JSON-LD is valid
    assert next(m for m in man if m["type"] == "jsonld")["valid"] is True
    # manifest file written + a fix maps to the structure finding by chunk id
    manifest_file = json.loads((Path(res["run_dir"]) / "manifest.json").read_text())
    assert any(m["finding_id"] == "structure:c1" for m in manifest_file)
    # every emitted draft carries the human-review banner
    for m in man:
        if m["path"].endswith((".md", ".txt")):
            assert fx.DRAFT_BANNER in Path(m["path"]).read_text(encoding="utf-8")
    assert trace.step_names() == ["fix_generation"]
