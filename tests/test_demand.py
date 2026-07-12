"""Phase 4 — query demand ingestion (src/demand.py). LLM stubbed; GKP/PAA real."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src import demand as dm
from src.trace import Trace


GKP_CSV = """Search terms report
Generated: 2026-07-01

Keyword,Avg. monthly searches,Competition
best crm small business,1300,High
free project tool,90,Low
enterprise crm,12000,High
best crm small business,1300,High
"""


def test_volume_bucket_and_weight_monotonic():
    assert dm.volume_bucket(0) == "0"
    assert dm.volume_bucket(5) == "1-10"
    assert dm.volume_bucket(1300) == "1K-10K"
    assert dm.volume_bucket(200000) == "100K+"
    assert dm.volume_weight(12000) > dm.volume_weight(1300) > dm.volume_weight(90)


def test_parse_gkp_skips_preamble_and_dedups(tmp_path):
    f = tmp_path / "gkp.csv"
    f.write_text(GKP_CSV, encoding="utf-8")
    rows = dm.parse_gkp_csv(f)
    kws = [r["keyword"] for r in rows]
    assert kws == ["best crm small business", "free project tool", "enterprise crm"]
    ent = next(r for r in rows if r["keyword"] == "enterprise crm")
    assert ent["searches"] == 12000 and ent["volume_bucket"] == "10K-100K"


def test_parse_gkp_utf16_tsv(tmp_path):
    f = tmp_path / "gkp.tsv"
    content = "Report\n\nKeyword\tAvg. monthly searches\nlogo maker\t5400\n"
    f.write_bytes(content.encode("utf-16"))   # real GKP exports carry the BOM
    rows = dm.parse_gkp_csv(f)
    assert rows[0]["keyword"] == "logo maker" and rows[0]["searches"] == 5400


def test_conversationalize_every_seed_gets_2plus_variants(monkeypatch):
    seeds = [{"keyword": "best crm small business", "searches": 1300,
              "volume_bucket": "1K-10K", "weight": 0.62},
             {"keyword": "free project tool", "searches": 90,
              "volume_bucket": "10-100", "weight": 0.4}]
    monkeypatch.setattr(dm.providers, "llm_complete", lambda p, **k:
                        '{"questions": ["Which CRM suits a 5-person agency?", '
                        '"What is the simplest CRM for a small team?", '
                        '"Is there an affordable CRM for tiny businesses?"]}')
    trace = Trace(query="q")
    qs = dm.conversationalize(seeds, trace=trace)
    by_seed = {}
    for q in qs:
        by_seed.setdefault(q["seed"], []).append(q)
        assert q["variant_type"] == "conversational"
        assert q["weight"] == next(s["weight"] for s in seeds if s["keyword"] == q["seed"])
    assert all(len(v) >= 2 for v in by_seed.values())     # every seed >= 2 variants
    assert trace.step_names() == ["conversationalize"]


def test_parse_paa(tmp_path):
    f = tmp_path / "paa.txt"
    f.write_text("- How much does a CRM cost?\n2. What is the best CRM?\n"
                 "Random short\nDoes a CRM integrate with email?\n", encoding="utf-8")
    out = dm.parse_paa(f)
    qs = [o["query"] for o in out]
    assert "How much does a CRM cost?" in qs
    assert "What is the best CRM?" in qs
    assert all(o["variant_type"] == "paa" for o in out)


def test_build_query_set_orders_by_weight(tmp_path, monkeypatch):
    f = tmp_path / "gkp.csv"
    f.write_text(GKP_CSV, encoding="utf-8")
    monkeypatch.setattr(dm.providers, "llm_complete", lambda p, **k:
                        '{"questions": ["Q one for this?", "Q two for this?"]}')
    qs = dm.build_query_set(gkp_path=str(f))
    weights = [q["weight"] for q in qs]
    assert weights == sorted(weights, reverse=True)   # high-volume first
    # enterprise crm (12000) variants outrank free project tool (90) variants
    assert qs[0]["seed"] == "enterprise crm"
