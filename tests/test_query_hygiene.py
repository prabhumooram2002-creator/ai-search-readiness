"""PHASE 12 P2 — query hygiene (src/query_hygiene.py): dedupe + topic-relevance
parking. Offline: fake KG + fake embed function, no models."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src import query_hygiene as qh


def test_dedupe_case_and_whitespace_insensitive():
    out = qh.dedupe_queries(["What is X?", "what is x?", "  What   is X?  ", "Different Q"])
    assert out == ["What is X?", "Different Q"]


def test_dedupe_preserves_first_seen_order():
    out = qh.dedupe_queries(["b", "a", "b", "c", "a"])
    assert out == ["b", "a", "c"]


class FakeResult:
    def __init__(self, rows):
        self._rows = list(rows)
        self._i = 0

    def has_next(self):
        return self._i < len(self._rows)

    def get_next(self):
        row = self._rows[self._i]
        self._i += 1
        return row


class FakeKG:
    def __init__(self, rows):
        self._rows = rows

    def _exec(self, query, params=None):
        return FakeResult(self._rows)


def test_topic_centroids_averages_member_chunk_embeddings():
    kg = FakeKG([
        ("t1", [1.0, 0.0]), ("t1", [1.0, 2.0]),   # mean -> [1.0, 1.0]
        ("t2", [0.0, 4.0]),
    ])
    c = qh.topic_centroids(kg)
    assert c["t1"] == [1.0, 1.0]
    assert c["t2"] == [0.0, 4.0]


def test_topic_centroids_skips_null_embeddings():
    kg = FakeKG([("t1", [1.0, 1.0]), ("t1", None)])
    c = qh.topic_centroids(kg)
    assert c["t1"] == [1.0, 1.0]


def test_score_query_relevance_splits_on_threshold():
    centroids = {"t1": [1.0, 0.0]}
    def embed_fn(texts):
        # "pricing" queries align with the centroid; "banana" queries don't
        return [[1.0, 0.0] if "pricing" in t.lower() else [0.0, 1.0] for t in texts]
    out = qh.score_query_relevance(
        ["What is our pricing?", "Best banana bread recipe?"], centroids, embed_fn, threshold=0.5)
    assert out["relevant"] == ["What is our pricing?"]
    assert len(out["skipped"]) == 1
    assert out["skipped"][0]["query"] == "Best banana bread recipe?"
    assert out["skipped"][0]["max_similarity"] == 0.0


def test_score_query_relevance_no_centroids_keeps_everything():
    out = qh.score_query_relevance(["q1", "q2"], {}, lambda t: [[1, 0]] * len(t))
    assert out["relevant"] == ["q1", "q2"]
    assert out["skipped"] == []


def test_clean_query_set_full_pipeline():
    kg = FakeKG([("t1", [1.0, 0.0])])
    def embed_fn(texts):
        return [[1.0, 0.0] if "pricing" in t.lower() else [0.0, 1.0] for t in texts]
    result = qh.clean_query_set(
        ["Pricing?", "pricing?", "Banana bread?"], kg=kg, embed_fn=embed_fn, threshold=0.5)
    assert result["n_duplicates_removed"] == 1
    assert result["relevant"] == ["Pricing?"]
    assert len(result["skipped"]) == 1


def test_clean_query_set_without_kg_only_dedupes():
    result = qh.clean_query_set(["a", "a", "b"])
    assert result["relevant"] == ["a", "b"]
    assert result["skipped"] == []
    assert result["n_duplicates_removed"] == 1
