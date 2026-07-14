"""Section 15 — Competitor Knowledge Graph (src/intelligence/competitor.py).
Only the comparison/path logic is testable offline; run_competitor_index()
itself needs a real competitor URL + a full Layers 0-2 crawl, not exercised
here (see its docstring for the known VectorStore-collision gap)."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.intelligence import competitor as comp
from src.trace import Trace


def test_competitor_kg_paths_are_namespaced_per_domain(tmp_path):
    p1 = comp.competitor_kg_paths("https://rival.com/", tmp_path)
    p2 = comp.competitor_kg_paths("https://other.com/", tmp_path)
    assert p1["slug"] != p2["slug"]
    assert "competitors" in p1["kg_path"]
    assert p1["kg_path"] != p2["kg_path"]


def test_comparison_table_includes_this_site_and_competitors():
    site = {"pages": 89, "chunks": 393, "entities": 428, "l0_pages": []}
    competitors = [{"domain": "rival.com", "pages": 50, "chunks": 200,
                    "entities": 100, "l0_pages": []}]
    rows = comp.comparison_table(site, competitors)
    assert rows[0]["domain"] == "this site"
    assert rows[1]["domain"] == "rival.com"
    assert "l0_pages" not in rows[0]  # internal-only field stripped before output


def test_comparison_table_computes_pagerank_concentration():
    pages = [{"url": "a", "internal_links": [{"dest_url": "b"}]},
            {"url": "b", "internal_links": [{"dest_url": "a"}]}]
    site = {"pages": 2, "l0_pages": pages}
    rows = comp.comparison_table(site, [])
    assert rows[0]["pagerank_concentration_top3"] is not None


def test_per_query_competitor_diff_flags_winner():
    ours = Trace(query="q", confidence=0.8, reranked=[{"chunk_id": "c1"}])
    theirs = Trace(query="q", confidence=0.5, reranked=[{"chunk_id": "t1"}])
    out = comp.per_query_competitor_diff(ours, theirs)
    assert out["we_win"] is True
    assert out["our_winning_chunks"] == ["c1"]
    assert out["their_winning_chunks"] == ["t1"]
