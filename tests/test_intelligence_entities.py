"""Section 2 — entities.py. Regression test for the 424-vs-428 real-KG
discrepancy: an Entity with zero MentionsEntity edges must still be counted
(query must start FROM Entity with OPTIONAL MATCH out, not FROM the edge)."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.intelligence import entities as entities_mod


class _FakeResult:
    def __init__(self, rows):
        self._rows = list(rows)
        self._i = 0
    def has_next(self):
        return self._i < len(self._rows)
    def get_next(self):
        row = self._rows[self._i]; self._i += 1; return row


class _FakeKG:
    def __init__(self, mention_rows, relates_rows=()):
        self.mention_rows = mention_rows
        self.relates_rows = relates_rows
        self._n = 0
    def _exec(self, query, params=None):
        self._n += 1
        if self._n == 1:
            return _FakeResult(self.mention_rows)
        return _FakeResult(self.relates_rows)


def test_entity_with_zero_mentions_still_counted():
    # e2 has no MentionsEntity edge at all -- the real bug found against
    # data/kg.kuzu (rs-610 etc.), which an INNER match from the edge drops.
    kg = _FakeKG(mention_rows=[
        ("e1", "Acme", "Organization", 0.9, "c1", "https://site.com/a"),
        ("e2", "Orphan Corp", "Organization", None, None, None),
    ])
    out = entities_mod.entity_stats(kg, l0_pages=[])
    assert out["n_entities"] == 2
    by_id = {e["id"]: e for e in out["all"]}
    assert by_id["e2"]["frequency"] == 0
    assert by_id["e2"]["mentions"] == []
    assert by_id["e1"]["frequency"] == 1


def test_mentioned_chunk_with_missing_page_link_still_counted():
    # c1 mentions e1 but HasChunk->Page is stale/missing (url=None) -- must
    # not crash, and must not be dropped from the entity's mention count.
    kg = _FakeKG(mention_rows=[
        ("e1", "Acme", "Organization", 0.8, "c1", None),
    ])
    out = entities_mod.entity_stats(kg, l0_pages=[])
    e1 = out["all"][0]
    assert e1["frequency"] == 1
    assert e1["n_pages"] == 0
    assert e1["mentions"] == [{"chunk_id": "c1", "url": None}]
