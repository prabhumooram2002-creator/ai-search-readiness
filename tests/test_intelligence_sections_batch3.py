"""Sections 7, 18, 19 — Backlinks (degraded), Citation Readiness, Recommendation chains."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.intelligence import backlinks, citation_readiness as cr, recommendations as rec


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


# ── Section 7 ────────────────────────────────────────────────────────────
def test_backlink_report_degraded_without_enrichment():
    kg = _FakeKG([])  # no ExternalDomain rows
    pages = [{"url": "a", "internal_links": [{"dest_url": "b"}]},
            {"url": "b", "internal_links": []}]
    out = backlinks.backlink_report(kg, pages)
    assert out["enriched"] is False
    assert "Common Crawl" in out["degraded_note"]
    assert out["excluded_metrics"]["authority_scores"]["available"] is False
    assert out["internal_pagerank_top_pages"]


def test_backlink_report_enriched_includes_external_links():
    kg = _FakeKG([("host1",)], [("p1", "ep1", "click here", 0.8)])
    out = backlinks.backlink_report(kg, [{"url": "p1", "internal_links": []}])
    assert out["enriched"] is True
    assert out["external_links"][0]["to"] == "ep1"


# ── Section 18 ───────────────────────────────────────────────────────────
def test_citation_readiness_row_computes_when_all_present():
    row = cr.citation_readiness_row(
        "u1", structure_score=0.8, evidence_strength=0.9, invisible_ratio=0.1,
        freshness=0.9, authority=0.7, simulated_retrieval_success=True)
    assert row["citation_readiness_score"]["value"] is not None
    assert row["citation_readiness_score"]["label"] == "heuristic"
    assert row["simulated_retrieval_success"]["label"] == "simulated"
    assert row["observed_citations"]["label"] == "observed"
    assert "blank until" in row["observed_citations"]["note"]


def test_citation_readiness_row_missing_input_is_none_not_fabricated():
    row = cr.citation_readiness_row(
        "u1", structure_score=None, evidence_strength=0.9, invisible_ratio=0.1,
        freshness=0.9, authority=0.7, simulated_retrieval_success=None)
    assert row["citation_readiness_score"]["value"] is None


def test_citation_readiness_columns_never_merged():
    row = cr.citation_readiness_row(
        "u1", 0.8, 0.9, 0.1, 0.9, 0.7, True, observed_citations={"chatgpt": 2})
    labels = {row["citation_readiness_score"]["label"],
             row["simulated_retrieval_success"]["label"],
             row["observed_citations"]["label"]}
    assert labels == {"heuristic", "simulated", "observed"}


# ── Section 19 ───────────────────────────────────────────────────────────
def test_causal_chains_group_by_shared_entity():
    recs = [
        {"rule": "missing_entity", "finding_id": "entity:acme", "action": "a1"},
        {"rule": "trust_bottleneck", "finding_id": "entity:acme", "chunk_id": "c1",
         "action": "a2", "predicted_impact": {"kind": "coverage", "delta": 0.2}},
    ]
    chains = rec.build_causal_chains(recs)
    assert len(chains) == 1
    assert chains[0]["n_steps"] == 2
    assert chains[0]["steps"][0]["rule"] == "missing_entity"  # ordered before trust_bottleneck
    assert chains[0]["predicted_impact"]["delta"] == 0.2


def test_causal_chains_ranked_by_predicted_impact():
    recs = [
        {"rule": "weak_evidence", "chunk_id": "c1", "finding_id": "structure:c1",
         "predicted_impact": {"kind": "coverage", "delta": 0.1}},
        {"rule": "weak_evidence", "chunk_id": "c2", "finding_id": "structure:c2",
         "predicted_impact": {"kind": "coverage", "delta": 0.5}},
    ]
    chains = rec.build_causal_chains(recs)
    assert chains[0]["chain_root"] == "chunk:c2"


def test_causal_chains_handle_missing_impact_gracefully():
    recs = [{"rule": "missing_entity", "finding_id": "entity:x", "action": "a"}]
    chains = rec.build_causal_chains(recs)
    assert chains[0]["predicted_impact"] is None
