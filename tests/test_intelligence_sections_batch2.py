"""Sections 6, 17 — Keyword Intelligence + Trust Intelligence."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.intelligence import keywords as kw
from src.intelligence import trust as tr


# ── Section 17 ───────────────────────────────────────────────────────────
def test_trust_checklist_all_pass():
    pages = [{"url": "u1", "content": "By Jane Doe, our nutritionist.",
             "schema_jsonld": [{"@type": "Review"}]},
            {"url": "u2/privacy-policy", "content": ""},
            {"url": "u3/about-us", "content": ""}]
    entities = [{"name": "ISO 9001", "type": "Certification"}]
    contacts = {"u1": {"emails": ["a@b.com"], "phones": []}}
    out = tr.trust_checklist(pages, entities, contacts, org_schema_complete=True)
    assert out["checklist"]["author_bylines_present"]["pass"] is True
    assert out["checklist"]["review_rating_schema"]["pass"] is True
    assert out["checklist"]["certifications_or_awards"]["pass"] is True
    assert out["score"]["label"] == "heuristic"


def test_trust_checklist_flags_missing_evidence():
    out = tr.trust_checklist([], [], {}, org_schema_complete=False)
    assert out["checklist"]["author_bylines_present"]["pass"] is False
    assert out["checklist"]["author_bylines_present"]["evidence"] == "missing"
    assert out["score"]["value"] < 0.5


# ── Section 6 ────────────────────────────────────────────────────────────
def test_parse_gsc_csv_basic(tmp_path):
    p = tmp_path / "gsc.csv"
    p.write_text("Query,Clicks,Impressions,CTR,Position\n"
                "best noodles,12,300,4%,3.2\n", encoding="utf-8")
    rows = kw.parse_gsc_csv(p)
    assert rows[0]["query"] == "best noodles"
    assert rows[0]["clicks"] == 12.0
    assert abs(rows[0]["ctr"] - 0.04) < 1e-9


def test_parse_gsc_csv_missing_query_column_returns_empty(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("Clicks,Impressions\n1,2\n", encoding="utf-8")
    assert kw.parse_gsc_csv(p) == []


def test_keyword_cannibalization_detects_competing_urls():
    rows = [{"query": "noodles", "ranking_url": "u1"},
           {"query": "noodles", "ranking_url": "u2"},
           {"query": "pasta", "ranking_url": "u3"}]
    out = kw.keyword_cannibalization(rows)
    assert len(out) == 1
    assert out[0]["query"] == "noodles"
    assert set(out[0]["competing_urls"]) == {"u1", "u2"}


def test_missing_and_long_tail_split_by_variant_type():
    gkp = [{"keyword": "noodles", "variant_type": None},
          {"keyword": "healthy noodle options for kids", "variant_type": "conversational"}]
    best = {"noodles": 0.1, "healthy noodle options for kids": 0.05}
    out = kw.missing_and_long_tail(gkp, best, retrieval_floor=0.25)
    assert len(out["missing_keywords"]) == 1
    assert len(out["long_tail_opportunities"]) == 1


def test_keyword_topic_mapping_picks_best_topic():
    keywords = [{"query": "pricing plans"}]
    centroids = {"t1": [1.0, 0.0], "t2": [0.0, 1.0]}
    def embed_fn(texts):
        return [[1.0, 0.0]] * len(texts)
    out = kw.keyword_topic_mapping(keywords, centroids, embed_fn, threshold=0.5)
    assert out[0]["best_topic"] == "t1"


def test_excluded_metrics_carry_reasons():
    assert kw.EXCLUDED_METRICS["difficulty"]["available"] is False
    assert "paid" in kw.EXCLUDED_METRICS["cpc"]["reason"]
