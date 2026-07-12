"""Phase 5 — snapshots & regression diff (src/snapshots.py)."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src import snapshots as sn


def _snap(run_id, pages, chunks, queries):
    return {"run_id": run_id, "ts": "t", "url": "http://x",
            "pages": pages, "chunks": chunks, "queries": queries}


BASELINE = _snap(1,
    pages={"http://x/a": {"invisibility": 0.0, "pagerank": 0.05,
                          "signals": {"trust": 0.8}}},
    chunks={"c1": 0.9, "c2": 0.7},
    queries={"q1": {"confidence": 0.6, "weighted_coverage": 0.8,
                    "weight": 0.9, "citation_chunks": ["c1", "c2"],
                    "winning_pages": ["http://x/a"]}})


def test_no_change_no_regressions():
    d = sn.diff_snapshots(BASELINE, BASELINE)
    assert d["regressions"] == [] and d["improvements"] == []
    assert d["ci_status"] == "pass"


def test_deleted_chunk_traced_as_coverage_cause():
    # break ONE thing: delete c1 (a covering chunk) -> q1 coverage drops
    after = _snap(2,
        pages=BASELINE["pages"],
        chunks={"c2": 0.7},                    # c1 deleted
        queries={"q1": {"confidence": 0.6, "weighted_coverage": 0.4,
                        "weight": 0.9, "citation_chunks": ["c2"],
                        "winning_pages": ["http://x/a"]}})
    d = sn.diff_snapshots(BASELINE, after)
    cov = next(r for r in d["regressions"] if r["kind"] == "query_weighted_coverage")
    assert cov["query"] == "q1"
    assert cov["from"] == 0.8 and cov["to"] == 0.4
    assert "c1" in cov["cause"]                # traced to the deleted chunk
    assert cov["severity"] == round(0.4 * 0.9, 4)   # magnitude x query weight
    assert d["ci_status"] == "fail"            # above threshold -> CI fails


def test_invisibility_regression_and_pagerank_severity():
    after = _snap(2,
        pages={"http://x/a": {"invisibility": 0.6, "pagerank": 0.05,
                              "signals": {"trust": 0.8}}},
        chunks=BASELINE["chunks"], queries=BASELINE["queries"])
    d = sn.diff_snapshots(BASELINE, after)
    inv = next(r for r in d["regressions"] if r["kind"] == "page_invisibility")
    assert inv["from"] == 0.0 and inv["to"] == 0.6
    assert "JS" in inv["cause"]                # cause: new JS template
    assert inv["severity"] > 0                 # weighted by pagerank
    assert d["ci_status"] == "fail"


def test_improvement_not_flagged_as_regression():
    after = _snap(2, pages=BASELINE["pages"], chunks={"c1": 0.9, "c2": 0.95},
                  queries={"q1": {**BASELINE["queries"]["q1"],
                                  "weighted_coverage": 0.95}})
    d = sn.diff_snapshots(BASELINE, after)
    assert d["regressions"] == []
    kinds = {i["kind"] for i in d["improvements"]}
    assert "query_weighted_coverage" in kinds and "chunk_structure" in kinds


def test_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(sn, "SNAP_DIR", tmp_path / "snaps")
    rid = sn.save_snapshot(dict(BASELINE))
    assert rid == 1
    assert sn.list_snapshots() == [1]
    assert sn.load_snapshot(1)["queries"]["q1"]["confidence"] == 0.6
    rid2 = sn.save_snapshot(dict(BASELINE))
    assert rid2 == 2 and sn.list_snapshots() == [1, 2]


def test_render_diff_contains_cause():
    after = _snap(2, pages=BASELINE["pages"], chunks={"c2": 0.7},
                  queries={"q1": {**BASELINE["queries"]["q1"],
                                  "weighted_coverage": 0.4, "citation_chunks": ["c2"]}})
    md = sn.render_diff(sn.diff_snapshots(BASELINE, after))
    assert "Regressions" in md and "c1" in md and "FAIL" in md
