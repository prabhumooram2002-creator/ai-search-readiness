"""Phase 12 shared helpers: core_utils.py + graph_stats.py."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.intelligence import core_utils as cu
from src.intelligence import graph_stats as gs


def test_cosine_identical_vectors_is_one():
    assert abs(cu.cosine([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9


def test_cosine_orthogonal_is_zero():
    assert cu.cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_pct_and_score2_formatting():
    assert cu.pct(0.4366) == "44%"
    assert cu.score2(0.12345) == "0.12"
    assert cu.pct(None) == "—"


def test_excluded_carries_reason():
    e = cu.excluded("needs a paid rank-data provider")
    assert e["available"] is False
    assert "paid" in e["reason"]


def test_collapse_identical_groups_and_counts():
    rows = [{"flag": "no_schema", "url": "a"}, {"flag": "no_schema", "url": "a"},
           {"flag": "other", "url": "b"}]
    out = cu.collapse_identical(rows, ("flag", "url"))
    assert len(out) == 2
    assert out[0]["count"] == 2
    assert out[1]["count"] == 1


def test_orphan_pages_zero_inbound():
    pages = [{"url": "a", "internal_links": [{"dest_url": "b"}]},
            {"url": "b", "internal_links": []},
            {"url": "c", "internal_links": []}]
    assert gs.orphan_pages(pages) == ["a", "c"]


def test_page_pagerank_returns_scores_for_linked_graph():
    pages = [{"url": "a", "internal_links": [{"dest_url": "b"}]},
            {"url": "b", "internal_links": [{"dest_url": "a"}]}]
    pr = gs.page_pagerank(pages)
    assert set(pr) == {"a", "b"}
    assert abs(sum(pr.values()) - 1.0) < 1e-6


class _FakeResult:
    def __init__(self, rows):
        self._rows = list(rows)
        self._i = 0
    def has_next(self):
        return self._i < len(self._rows)
    def get_next(self):
        row = self._rows[self._i]; self._i += 1; return row


class _FakeKG:
    def __init__(self, edge_rows, entity_rows):
        self.edge_rows, self.entity_rows = edge_rows, entity_rows
        self._n = 0
    def _exec(self, query, params=None):
        self._n += 1
        return _FakeResult(self.edge_rows if self._n == 1 else self.entity_rows)


def test_entity_relation_graph_includes_isolated_entities():
    kg = _FakeKG(edge_rows=[("e1", "e2")], entity_rows=[("e1",), ("e2",), ("e3",)])
    nodes, edges = gs.entity_relation_graph(kg)
    assert nodes == ["e1", "e2", "e3"]
    assert edges == [("e1", "e2")]
