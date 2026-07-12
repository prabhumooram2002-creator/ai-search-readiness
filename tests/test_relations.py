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


def test_precision_floor_and_cross_chunk_dedup(monkeypatch):
    # backlog: over-generation control. Two chunks each state the SAME
    # (acme, founded by, jane-roe); a third weak/nonsense edge is below floor.
    ents2 = [
        _entity("acme", "Acme", "Organization", "c0", 0, 4),
        _entity("jane-roe", "Jane Roe", "Person", "c0", 20, 28),
        _entity("berlin", "Berlin", "Location", "c0", 32, 38),
    ]
    # add mentions of acme/jane in a second chunk c1 so dedup can corroborate
    for e in ents2:
        e["mentions"].append({"chunk_id": "c1", "char_start": e["mentions"][0]["char_start"],
                              "char_end": e["mentions"][0]["char_end"], "score": 0.9})

    def preds_for(tokens, labels, threshold=0.5, ner=None, top_k=-1):
        # same strong founded-by in both chunks (0.9), plus one TYPE-VALID but
        # weak edge (Acme[org] located-in Berlin[location] @0.4) -> below floor
        return [
            {"head_pos": [0, 1], "tail_pos": [4, 6], "head_text": ["Acme"],
             "tail_text": ["Jane", "Roe"], "label": "founded by", "score": 0.90},
            {"head_pos": [0, 1], "tail_pos": [7, 8], "head_text": ["Acme"],
             "tail_text": ["Berlin."], "label": "located in", "score": 0.40},  # weak
        ]
    fake = FakeGlirel([]); fake.predict_relations = preds_for
    monkeypatch.setattr(rel, "_get_glirel", lambda: fake)

    chunks = [{"content": TEXT, "chunk_id": "c0"}, {"content": TEXT, "chunk_id": "c1"}]
    trace = Trace(query="q")
    out = rel.extract_relations(chunks, ents2, trace=trace)

    fb = [t for t in out if t["relation"] == "founded by"]
    assert len(fb) == 1                              # deduped across c0 + c1
    assert fb[0]["support_count"] == 2
    assert set(fb[0]["source_chunk_ids"]) == {"c0", "c1"}
    # the weak located-in edge (0.40, type-valid) is below the floor -> dropped
    assert all(t["relation"] != "located in" for t in out)
    assert any("below precision floor" in d["reason"] for d in trace.steps[0].dropped)
    assert trace.steps[0].outputs["n_type_dropped"] == 0   # both were type-valid
    assert trace.steps[0].outputs["n_deduped"] >= len(out)


def test_type_constraints_kill_wrong_direction_and_type(monkeypatch):
    # entities with types: org, person, location
    ents = [
        _entity("acme", "Acme", "Organization", "c0", 0, 4),
        _entity("jane-roe", "Jane Roe", "Person", "c0", 20, 28),
        _entity("berlin", "Berlin", "Location", "c0", 32, 38),
    ]
    def preds_for(tokens, labels, threshold=0.5, ner=None, top_k=-1):
        return [
            # correct: Acme(org) founded by Jane(person) -> KEPT
            {"head_pos": [0, 1], "tail_pos": [4, 6], "label": "founded by", "score": 0.9},
            # wrong direction: Jane(person) founded by Acme(org) -> type violation
            {"head_pos": [4, 5], "tail_pos": [0, 1], "label": "founded by", "score": 0.9},
            # nonsense: Berlin(location) founded by Jane(person) -> type violation
            {"head_pos": [7, 8], "tail_pos": [4, 6], "label": "founded by", "score": 0.9},
            # correct: Acme(org) located in Berlin(location) -> KEPT
            {"head_pos": [0, 1], "tail_pos": [7, 8], "label": "located in", "score": 0.9},
            # nonsense: Acme(org) located in Jane(person) -> tail not Location
            {"head_pos": [0, 1], "tail_pos": [4, 6], "label": "located in", "score": 0.9},
        ]
    fake = FakeGlirel([]); fake.predict_relations = preds_for
    monkeypatch.setattr(rel, "_get_glirel", lambda: fake)

    trace = Trace(query="q")
    out = rel.extract_relations([CHUNK], ents, trace=trace)
    got = {(t["subject_entity_id"], t["relation"], t["object_entity_id"]) for t in out}
    assert got == {("acme", "founded by", "jane-roe"),
                   ("acme", "located in", "berlin")}     # only the 2 valid ones
    assert trace.steps[0].outputs["n_type_dropped"] == 3
    assert any("type constraint" in d["reason"] for d in trace.steps[0].dropped)


def test_symmetric_relation_canonicalized(monkeypatch):
    ents = [_entity("claude", "Claude", "Product", "c0", 0, 6),
            _entity("gemini", "Gemini", "Product", "c0", 20, 26)]
    def preds_for(tokens, labels, threshold=0.5, ner=None, top_k=-1):
        # both directions of a symmetric relation -> collapse to one edge
        return [
            {"head_pos": [0, 1], "tail_pos": [4, 5], "label": "competes with", "score": 0.8},
            {"head_pos": [4, 5], "tail_pos": [0, 1], "label": "competes with", "score": 0.7},
        ]
    fake = FakeGlirel([]); fake.predict_relations = preds_for
    # both entities need >=2-token presence; map token 0->claude, 4->gemini
    ents[1]["mentions"][0]["char_start"] = 20; ents[1]["mentions"][0]["char_end"] = 26
    monkeypatch.setattr(rel, "_get_glirel", lambda: fake)
    out = rel.extract_relations([{"content": "Claude and also the Gemini model here.",
                                  "chunk_id": "c0"}], ents)
    comp = [t for t in out if t["relation"] == "competes with"]
    assert len(comp) == 1 and comp[0]["support_count"] == 2   # A<->B merged
