"""Sections 11/12 — Claim Intelligence (src/claim_intel.py)."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src import claim_intel as ci


def test_is_superlative_claim_detects_marketing_language():
    assert ci.is_superlative_claim("WickedGud is India's No.1 noodle brand")
    assert ci.is_superlative_claim("Our best-selling product")
    assert not ci.is_superlative_claim("The noodles cost Rs. 240")


def test_cross_page_evidence_splits_supporting_and_contradicting():
    claims = [{"id": "cl1", "text": "Noodles are healthy", "source_chunk_id": "c0"}]
    chunks_by_id = {"c1": {"content": "text1", "url": "u1"},
                   "c2": {"content": "text2", "url": "u2"}}
    claim_embeddings = {"cl1": [1.0, 0.0]}
    chunk_embeddings = {"c1": [1.0, 0.0], "c2": [0.9, 0.1]}

    def score_fn(pairs):
        return [("entailment", 0.9), ("contradiction", 0.8)]

    out = ci.cross_page_evidence(claims, chunks_by_id, claim_embeddings,
                                 chunk_embeddings, score_fn, top_k=2)
    rec = out[0]
    assert rec["evidence_count"] == 2
    assert len(rec["supporting"]) == 1
    assert len(rec["contradicting"]) == 1
    assert rec["confidence"]["label"] == "heuristic"


def test_cross_page_evidence_excludes_source_chunk():
    claims = [{"id": "cl1", "text": "x", "source_chunk_id": "c1"}]
    chunks_by_id = {"c1": {"content": "self", "url": "u1"}, "c2": {"content": "other", "url": "u2"}}
    claim_embeddings = {"cl1": [1.0, 0.0]}
    chunk_embeddings = {"c1": [1.0, 0.0], "c2": [1.0, 0.0]}
    seen = []
    def score_fn(pairs):
        seen.extend(pairs)
        return [("entailment", 0.9)] * len(pairs)
    ci.cross_page_evidence(claims, chunks_by_id, claim_embeddings, chunk_embeddings,
                           score_fn, top_k=5)
    assert all(p[0] == "other" for p in seen)


def test_cross_page_evidence_missing_embedding_is_honest():
    claims = [{"id": "cl1", "text": "x"}]
    out = ci.cross_page_evidence(claims, {}, {}, {}, lambda p: [])
    assert out[0]["note"] == "no embedding available"
    assert out[0]["evidence_count"] == 0


def test_rank_claims_finds_naked_and_contradicted():
    records = [
        {"claim_id": "c1", "text": "best", "is_superlative": True,
         "evidence_count": 0, "supporting": [], "contradicting": [], "confidence": None},
        {"claim_id": "c2", "text": "costs 10", "is_superlative": False,
         "evidence_count": 2, "supporting": [{}], "contradicting": [{}],
         "confidence": {"value": 0.5}},
    ]
    out = ci.rank_claims(records)
    assert out["naked_superlative_claims"][0]["claim_id"] == "c1"
    assert out["contradicted_claims"][0]["claim_id"] == "c2"
