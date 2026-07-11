"""Layer 1 step 4 — GLiREL relation extraction (src/relations.py).

Offline tests with a stubbed GLiREL model: verify token mapping, the
CLAUDE.md endpoint-validation rule (no hallucinated entities — drops recorded
with reasons), dedup, and Trace emission. The live-model run is in BUILD_LOG.md.
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src import relations as rel
from src.trace import Trace


def _entity(eid, name, typ, chunk_id, start, end):
    return {
        "id": eid, "type": typ, "name": name, "synonyms": [],
        "source_chunk_ids": [chunk_id], "score": 0.9,
        "mentions": [{"chunk_id": chunk_id, "char_start": start,
                      "char_end": end, "score": 0.9}],
    }


TEXT = "Acme was founded by Jane Roe in Berlin."
#       0123456789012345678901234567890123456789
# Acme: 0-4   Jane Roe: 20-28   Berlin: 32-38
CHUNK = {"content": TEXT, "chunk_id": "c0"}
ENTS = [
    _entity("acme", "Acme", "Organization", "c0", 0, 4),
    _entity("jane-roe", "Jane Roe", "Person", "c0", 20, 28),
    _entity("berlin", "Berlin", "Location", "c0", 32, 38),
]


class FakeGlirel:
    def __init__(self, preds):
        self.preds = preds
        self.seen = {}

    def predict_relations(self, tokens, labels, threshold=0.5, ner=None, top_k=-1):
        self.seen = {"tokens": tokens, "labels": labels, "ner": ner}
        return self.preds


def test_char_to_token_span_mapping():
    tokens, offsets = rel._tokenize_with_offsets(TEXT)
    assert tokens[0] == "Acme"
    assert rel._char_span_to_token_span(0, 4, offsets) == (0, 0)      # Acme
    assert rel._char_span_to_token_span(20, 28, offsets) == (4, 5)    # Jane Roe
    assert rel._char_span_to_token_span(32, 38, offsets) == (7, 7)    # Berlin.


def test_valid_triplets_and_ner_spans(monkeypatch):
    fake = FakeGlirel([
        {"head_pos": [0, 1], "tail_pos": [4, 6], "head_text": ["Acme"],
         "tail_text": ["Jane", "Roe"], "label": "founded by", "score": 0.93},
        {"head_pos": [0, 1], "tail_pos": [7, 8], "head_text": ["Acme"],
         "tail_text": ["Berlin."], "label": "located in", "score": 0.81},
    ])
    monkeypatch.setattr(rel, "_get_glirel", lambda: fake)

    trace = Trace(query="q")
    out = rel.extract_relations([CHUNK], ENTS, trace=trace)

    assert {(t["subject_entity_id"], t["relation"], t["object_entity_id"]) for t in out} == {
        ("acme", "founded by", "jane-roe"),
        ("acme", "located in", "berlin"),
    }
    assert all(t["source_chunk_id"] == "c0" for t in out)
    # NER spans handed to GLiREL came from step-3 mentions
    assert [s[3] for s in fake.seen["ner"]] == ["Acme", "Jane Roe", "Berlin"]
    # Trace step emitted with scores
    assert trace.step_names() == ["relation_extraction"]
    assert trace.steps[0].scores["n_triplets"] == 2.0


def test_hallucinated_endpoint_dropped_with_reason(monkeypatch):
    fake = FakeGlirel([
        # tail_pos 2 maps to no step-3 mention -> must be dropped
        {"head_pos": [0, 1], "tail_pos": [2, 3], "head_text": ["Acme"],
         "tail_text": ["founded"], "label": "uses", "score": 0.99},
    ])
    monkeypatch.setattr(rel, "_get_glirel", lambda: fake)

    trace = Trace(query="q")
    out = rel.extract_relations([CHUNK], ENTS, trace=trace)
    assert out == []
    dropped = trace.steps[0].dropped
    assert len(dropped) == 1
    assert dropped[0]["reason"] == "endpoint not in step-3 entities for this chunk"


def test_dedup_keeps_best_score_and_skips_single_entity_chunks(monkeypatch):
    fake = FakeGlirel([
        {"head_pos": [0, 1], "tail_pos": [4, 6], "head_text": ["Acme"],
         "tail_text": ["Jane", "Roe"], "label": "founded by", "score": 0.60},
        {"head_pos": [0, 1], "tail_pos": [4, 6], "head_text": ["Acme"],
         "tail_text": ["Jane", "Roe"], "label": "founded by", "score": 0.95},
    ])
    monkeypatch.setattr(rel, "_get_glirel", lambda: fake)
    out = rel.extract_relations([CHUNK], ENTS)
    assert len(out) == 1 and out[0]["score"] == 0.95

    # A chunk with <2 step-3 entities is skipped entirely (no model call needed)
    lone = {"content": "Acme alone.", "chunk_id": "solo"}
    lone_ents = [_entity("acme", "Acme", "Organization", "solo", 0, 4)]
    assert rel.extract_relations([lone], lone_ents) == []


def test_trace_step_appended_even_on_model_error(monkeypatch):
    class Boom:
        def predict_relations(self, *a, **k):
            raise RuntimeError("model exploded")
    monkeypatch.setattr(rel, "_get_glirel", lambda: Boom())

    trace = Trace(query="q")
    try:
        rel.extract_relations([CHUNK], ENTS, trace=trace)
        assert False, "should have raised"
    except RuntimeError:
        pass
    assert trace.step_names() == ["relation_extraction"]
    assert any("ERROR" in n for n in trace.steps[0].notes)
