"""Phase 6 — manual calibration panel (src/calibration.py)."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src import calibration as cal


FILLED = """# Calibration — best crm for small teams
query_id: q001
weight: 0.9

## ChatGPT
HubSpot and Zoho are good options for small teams.
### Cited URLs
- https://www.hubspot.com/products/crm
- https://competitor.com/blog

## Perplexity
Several CRMs work well.
### Cited URLs
- https://competitor.com/list

## Gemini
<paste the answer text here>
### Cited URLs
-
"""


def test_generate_templates(tmp_path):
    root = cal.generate_templates(3, [{"query": "a b c", "weight": 0.5},
                                       {"query": "d e", "weight": 0.9}],
                                  calib_dir=tmp_path)
    files = sorted(root.glob("q*.md"))
    assert len(files) == 2
    body = files[0].read_text(encoding="utf-8")
    for eng in cal.ENGINES:
        assert f"## {eng}" in body
    assert "### Cited URLs" in body


def test_parse_extracts_answer_and_urls(tmp_path):
    f = tmp_path / "q001.md"
    f.write_text(FILLED, encoding="utf-8")
    rec = cal.parse_calibration_file(f)
    assert rec["query"] == "best crm for small teams"
    assert rec["query_id"] == "q001" and rec["weight"] == 0.9
    cg = rec["engines"]["ChatGPT"]
    assert "hubspot.com" in cg["cited_domains"]
    assert "competitor.com" in cg["cited_domains"]
    assert cg["answer"].startswith("HubSpot")
    # Gemini left as placeholder -> empty answer, no urls
    assert rec["engines"]["Gemini"]["answer"] == ""
    assert rec["engines"]["Gemini"]["cited_urls"] == []


def test_agreement_math_hand_checked(tmp_path):
    f = tmp_path / "q001.md"
    f.write_text(FILLED, encoding="utf-8")
    calibration = [cal.parse_calibration_file(f)]
    # simulator cited a hubspot.com page for this query
    wins = {"best crm for small teams": ["https://www.hubspot.com/products/crm"]}
    ag = cal.compute_agreement(calibration, wins)
    # ChatGPT cited hubspot.com -> agree (1.0); Perplexity didn't -> 0.0;
    # Gemini not filled -> skipped
    assert ag["per_engine_agreement"]["ChatGPT"] == 1.0
    assert ag["per_engine_agreement"]["Perplexity"] == 0.0
    assert ag["per_engine_agreement"]["Gemini"] is None
    assert ag["overall_agreement"] == 0.5      # 1 hit / 2 filled engines
    assert ag["label"] == "observed"


def test_below_floor_triggers_tuning_note(tmp_path):
    f = tmp_path / "q001.md"
    f.write_text(FILLED, encoding="utf-8")
    # simulator cited a page NOBODY cited -> 0 agreement -> below floor
    ag = cal.compute_agreement([cal.parse_calibration_file(f)],
                               {"best crm for small teams": ["https://nobody.com/x"]})
    assert ag["overall_agreement"] == 0.0
    assert ag["below_floor"] is True
    assert "tune retriever" in ag["note"]


def test_simulated_and_observed_never_merged(tmp_path):
    # the observed structure is labeled 'observed' and carries no simulated field
    f = tmp_path / "q001.md"
    f.write_text(FILLED, encoding="utf-8")
    ag = cal.compute_agreement([cal.parse_calibration_file(f)],
                               {"best crm for small teams": ["https://www.hubspot.com/x"]})
    assert ag["label"] == "observed"
    assert "simulated" not in ag
    assert "confidence" not in ag
