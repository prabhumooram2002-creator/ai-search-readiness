"""PHASE 10 (src/report/decision_report.py) — verdict/grade formula, action-plan
ranking + URL resolution, battle-card verdicts, and the 10c renderer rules
(jargon ban, no bare chunk-id patterns, predicted labels preserved). All pure
functions over small synthetic fixtures — no models, no DB."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.report import decision_report as dr


# ─────────────────────────────────────────────────────────────────────────────
# S5 — trend chunk_structure URL resolution
# ─────────────────────────────────────────────────────────────────────────────
def test_chunk_structure_regression_resolves_to_url():
    diff = {"regressions": [{"kind": "chunk_structure", "chunk_id": "abc123",
                             "from": 0.8, "to": 0.4, "cause": "structure changed"}],
           "improvements": []}
    out = dr.resolve_chunk_structure_urls(diff, CHUNK_LOOKUP)
    assert out["regressions"][0]["url"] == "https://site.com/pricing"


def test_chunk_structure_regression_dropped_when_unresolvable():
    diff = {"regressions": [{"kind": "chunk_structure", "chunk_id": "ghost",
                             "from": 0.8, "to": 0.4, "cause": "structure changed"}],
           "improvements": []}
    out = dr.resolve_chunk_structure_urls(diff, CHUNK_LOOKUP)
    assert out["regressions"] == []


def test_non_chunk_structure_items_pass_through_unchanged():
    diff = {"regressions": [{"kind": "page_invisibility", "url": "https://site.com/x",
                             "from": 0.1, "to": 0.5, "cause": "js-only"}],
           "improvements": []}
    out = dr.resolve_chunk_structure_urls(diff, {})
    assert out["regressions"] == diff["regressions"]


# ─────────────────────────────────────────────────────────────────────────────
# S1 — grade + verdict
# ─────────────────────────────────────────────────────────────────────────────
def test_site_grade_bands_and_formula_logged():
    good = dr.compute_site_grade(invisibility=0.05, mean_coverage=0.9,
                                 mean_structure=0.9, mean_authority=0.9)
    bad = dr.compute_site_grade(invisibility=0.9, mean_coverage=0.1,
                                mean_structure=0.1, mean_authority=0.1)
    assert good["grade"] == "A"
    assert bad["grade"] == "F"
    assert good["label"] == "heuristic"
    assert "invisibility" in good["formula"]


def test_verdict_predicted_lift_capped_and_labeled():
    v = dr.build_verdict(
        invisibility={"site_invisibility": 0.44, "label": "measured", "headline": "44% invisible"},
        queries=[{"weighted_coverage": 0.3}, {"weighted_coverage": 0.5}],
        mean_structure=0.6, mean_authority=0.5,
        top_action_deltas=[0.4, 0.3, 0.9])
    assert v["mean_weighted_coverage"] == 0.4
    assert "predicted, simulated" in v["predicted_lift_sentence"]
    # lift is capped at 1.0 (the true ceiling) even though 0.4+0.4+0.3+0.9 would exceed it
    assert "100%" in v["predicted_lift_sentence"]


def test_verdict_lift_never_below_starting_coverage():
    """Real bug found running the actual penny-test: an old 0.95 sanity cap
    produced "lift from 98% to 95%" whenever measured coverage already
    exceeded 95% — a nonsensical negative "lift". The cap must never push
    the predicted "after" number below the measured "before" number."""
    v = dr.build_verdict(
        invisibility={"site_invisibility": 0.02, "label": "measured", "headline": "2% invisible"},
        queries=[{"weighted_coverage": 0.98}], mean_structure=0.8, mean_authority=0.8,
        top_action_deltas=[0.1, 0.05])
    assert v["mean_weighted_coverage"] == 0.98
    assert "98%" in v["predicted_lift_sentence"]
    assert "lift from 98% to 95%" not in v["predicted_lift_sentence"]
    # "after" must be >= "before"
    before = float(v["mean_weighted_coverage"])
    import re as _re
    after_pct = int(_re.search(r"to (\d+)%", v["predicted_lift_sentence"]).group(1))
    assert after_pct >= round(before * 100)


def test_verdict_no_fixes_run_gives_honest_message():
    v = dr.build_verdict(invisibility={}, queries=[], mean_structure=0, mean_authority=0,
                         top_action_deltas=[])
    assert "--fixes" in v["predicted_lift_sentence"]


# ─────────────────────────────────────────────────────────────────────────────
# S2 — action plan
# ─────────────────────────────────────────────────────────────────────────────
CHUNK_LOOKUP = {
    "abc123": {"url": "https://site.com/pricing", "heading_path": ["Pricing", "Plans"],
              "text": "Our plans start at $10/month for the basic tier."},
}


def test_action_plan_rows_all_have_urls():
    recs = [
        {"rule": "weak_evidence", "chunk_id": "abc123", "finding_id": "structure:abc123",
         "action": "Add supporting data (NLI neutral 0.9)."},
        {"rule": "retrieval_dead_end", "finding_id": "deadend:pricing-tiers",
         "action": 'Real content gap: no chunk answers "what are your pricing tiers" above the score floor.'},
    ]
    rows = dr.build_action_plan(
        query_recommendations=[("pricing tiers", 0.8, recs)],
        structure_flags=[], invisibility_worst_pages=[], schema_missing_pages=[],
        visited_but_invisible=[], chunk_lookup=CHUNK_LOOKUP)
    assert len(rows) >= 1
    for r in rows:
        assert r["url"], f"row has no resolvable URL: {r}"


def test_action_plan_drops_unresolvable_rows():
    recs = [{"rule": "missing_entity", "finding_id": "entity:ghost-brand",
            "action": "irrelevant"}]  # no chunk_id, no page_url -> unresolvable
    rows = dr.build_action_plan(
        query_recommendations=[("q", 0.5, recs)],
        structure_flags=[], invisibility_worst_pages=[], schema_missing_pages=[],
        visited_but_invisible=[], chunk_lookup={})
    assert rows == []


def test_action_plan_priority_orders_by_weight_and_severity():
    recs_hi = [{"rule": "weak_evidence", "chunk_id": "abc123", "finding_id": "structure:abc123",
               "action": "x", "predicted_impact": {"kind": "coverage", "delta": 0.5, "label": "predicted (simulated)"}}]
    recs_lo = [{"rule": "weak_evidence", "chunk_id": "abc123", "finding_id": "structure:abc123",
               "action": "x", "predicted_impact": {"kind": "coverage", "delta": 0.05, "label": "predicted (simulated)"}}]
    rows = dr.build_action_plan(
        query_recommendations=[("high-weight q", 0.9, recs_hi), ("low-weight q", 0.1, recs_lo)],
        structure_flags=[], invisibility_worst_pages=[], schema_missing_pages=[],
        visited_but_invisible=[], chunk_lookup=CHUNK_LOOKUP)
    assert rows[0]["source_query"] == "high-weight q"


def test_technical_visibility_items_included():
    rows = dr.build_action_plan(
        query_recommendations=[], structure_flags=[],
        invisibility_worst_pages=[{"url": "https://site.com/js-page",
                                   "mean_invisible_ratio": 0.8, "pagerank": 0.02}],
        schema_missing_pages=[{"url": "https://site.com/no-schema"}],
        visited_but_invisible=["https://site.com/visited-invisible"],
        chunk_lookup={})
    cats = {r["category"] for r in rows}
    assert cats == {"technical_visibility"}
    assert len(rows) == 3


# ─────────────────────────────────────────────────────────────────────────────
# S3 — battle cards
# ─────────────────────────────────────────────────────────────────────────────
def test_battle_card_verdict_badges():
    assert dr._verdict_badge(citations=0, confidence=0.3) == "INVISIBLE"
    assert dr._verdict_badge(citations=3, confidence=0.7) == "ANSWERABLE"
    assert dr._verdict_badge(citations=1, confidence=0.55) == "PARTIAL"


def test_battle_cards_ranked_by_weight_and_capped():
    records = [{"query": f"q{i}", "weight": i / 10, "answer": "a", "confidence": 0.6,
               "n_citations": 1} for i in range(15)]
    cards = dr.build_battle_cards(records, action_plan=[], top_n=10)
    assert len(cards) == 10
    assert cards[0]["query"] == "q14"  # highest weight first


def test_battle_card_links_to_top_action():
    action_plan = [{"source_query": "pricing tiers", "priority": 0.9,
                    "problem": "no content about pricing tiers", "url": "https://site.com/x"}]
    records = [{"query": "pricing tiers", "weight": 1.0, "answer": "I don't know.",
               "confidence": 0.3, "n_citations": 0,
               "retrieval_dead_ends": [{"sub_query": "pricing tiers", "weight": 0.8}]}]
    cards = dr.build_battle_cards(records, action_plan, top_n=10)
    assert cards[0]["verdict"] == "INVISIBLE"
    assert cards[0]["top_action"]["problem"] == "no content about pricing tiers"
    assert cards[0]["missing_sub_intents"]


# ─────────────────────────────────────────────────────────────────────────────
# 10c — renderer rules
# ─────────────────────────────────────────────────────────────────────────────
def _minimal_ctx():
    v = dr.build_verdict(
        invisibility={"site_invisibility": 0.3, "label": "measured", "headline": "30% invisible"},
        queries=[{"weighted_coverage": 0.5}], mean_structure=0.6, mean_authority=0.5,
        top_action_deltas=[0.2])
    rows = dr.build_action_plan(
        query_recommendations=[("pricing tiers", 0.8, [
            {"rule": "weak_evidence", "chunk_id": "abc123", "finding_id": "structure:abc123",
             "action": "Add supporting data (NLI neutral 0.9).",
             "predicted_impact": {"kind": "coverage", "coverage_before": 0.3,
                                  "coverage_after": 0.6, "delta": 0.3,
                                  "label": "predicted (simulated)"}}])],
        structure_flags=[], invisibility_worst_pages=[], schema_missing_pages=[],
        visited_but_invisible=[], chunk_lookup=CHUNK_LOOKUP)
    cards = dr.build_battle_cards(
        [{"query": "pricing tiers", "weight": 0.8, "answer": "We have three tiers.",
         "confidence": 0.7, "n_citations": 2}], rows, top_n=10)
    technical = dr.build_technical_section(
        invisibility={"worst_pages": [{"per_bot": {"GPTBot": 0.4}}]},
        robots_conflicts=0, orphan_pages=[], missing_links=[],
        llms_txt_present=True, serverlogs=None)
    return {"url": "https://site.com", "verdict": v, "action_plan": rows,
            "battle_cards": cards, "technical": technical, "trend": None}


def test_rendered_report_has_no_jargon():
    html = dr.render_report_html(_minimal_ctx())
    violations = dr.enforce_renderer_rules(html)
    assert violations == [], f"renderer rule violations: {violations}"


def test_rendered_report_preserves_predicted_label():
    html = dr.render_report_html(_minimal_ctx())
    assert "predicted (simulated)" in html


def test_rendered_report_every_action_row_has_a_url():
    ctx = _minimal_ctx()
    html = dr.render_report_html(ctx)
    for row in ctx["action_plan"]:
        assert row["url"] in html


def test_number_formatting_rules():
    assert dr._pct(0.4366) == "44%"
    assert dr._pct(1.0) == "100%"
    assert dr._score(0.123456) == "0.12"
    assert dr._score(2) == "2"


def test_translate_finding_never_uses_banned_jargon():
    recs = [
        {"rule": "missing_entity", "finding_id": "entity:widget-pricing"},
        {"rule": "weak_evidence", "finding_id": "structure:abc"},
        {"rule": "retrieval_dead_end", "action": 'no chunk answers "how much does it cost"'},
        {"rule": "unsupported_answer_sentence"},
        {"rule": "trust_bottleneck"},
    ]
    for rec in recs:
        text = dr.translate_finding(rec, snippet="some snippet")
        assert not dr._JARGON_RE.search(text), f"jargon in translated text: {text!r}"
