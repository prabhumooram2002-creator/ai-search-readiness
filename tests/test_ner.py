"""Layer 1 step 3 — GLiNER entity extraction (src/ner.py).

Offline tests: GLiNER/spaCy are stubbed so the suite stays fast and
network-free — these verify wiring, dedup/merge, Trace emission, and the
cross-check math. The live-model run on real input is logged in BUILD_LOG.md.
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src import ner
from src.trace import Trace


class FakeGliner:
    """Deterministic stand-in for GLiNER.predict_entities."""
    def __init__(self, preds_by_text):
        self.preds_by_text = preds_by_text

    def predict_entities(self, text, labels, threshold=0.4):
        return self.preds_by_text.get(text, [])


def test_default_labels_match_claude_md():
    assert ner.DEFAULT_LABELS == [
        "person", "organization", "product", "feature", "price", "date", "location",
    ]


def test_extraction_dedup_merge_and_trace(monkeypatch):
    fake = FakeGliner({
        "Acme builds the Widget.": [
            {"text": "Acme", "label": "organization", "score": 0.91, "start": 0, "end": 4},
            {"text": "Widget", "label": "product", "score": 0.80, "start": 16, "end": 22},
        ],
        "ACME shipped widget v2 on 2026-01-01.": [
            {"text": "ACME", "label": "organization", "score": 0.95, "start": 0, "end": 4},
            {"text": "2026-01-01", "label": "date", "score": 0.77, "start": 26, "end": 36},
        ],
    })
    monkeypatch.setattr(ner, "_get_gliner", lambda: fake)

    chunks = [
        {"content": "Acme builds the Widget.", "chunk_id": "c1"},
        {"content": "ACME shipped widget v2 on 2026-01-01.", "chunk_id": "c2"},
    ]
    trace = Trace(query="q")
    ents = ner.extract_entities(chunks, trace=trace)

    by_id = {e["id"]: e for e in ents}
    # "Acme" and "ACME" share slug+label -> merged into one entity
    assert set(by_id) == {"acme", "widget", "2026-01-01"}
    acme = by_id["acme"]
    assert acme["type"] == "Organization"
    assert acme["source_chunk_ids"] == ["c1", "c2"]
    assert acme["synonyms"] == ["ACME"]          # variant kept as synonym
    assert acme["score"] == 0.95                  # max mention score (heuristic)
    assert len(acme["mentions"]) == 2

    # Trace: one StepTrace named entity_extraction with outputs + scores
    assert trace.step_names() == ["entity_extraction"]
    st = trace.steps[0]
    assert st.scores["n_entities"] == 3.0
    assert st.outputs["n_raw_mentions"] == 4
    assert st.inputs["model"] == ner.GLINER_MODEL


def test_trace_step_appended_even_on_model_error(monkeypatch):
    class Boom:
        def predict_entities(self, *a, **k):
            raise RuntimeError("model exploded")
    monkeypatch.setattr(ner, "_get_gliner", lambda: Boom())

    trace = Trace(query="q")
    try:
        ner.extract_entities([{"content": "x y z", "chunk_id": "c"}], trace=trace)
        assert False, "should have raised"
    except RuntimeError:
        pass
    # Stage still emitted its StepTrace, with the error noted
    assert trace.step_names() == ["entity_extraction"]
    assert any("ERROR" in n for n in trace.steps[0].notes)


def test_crosscheck_disagreement_math(monkeypatch):
    class FakeEnt:
        def __init__(self, text, label):
            self.text, self.label_ = text, label

    class FakeDoc:
        def __init__(self, ents):
            self.ents = ents

    class FakeNlp:
        def __call__(self, text):
            if "Acme" in text:
                return FakeDoc([FakeEnt("Acme", "ORG")])       # full agreement
            return FakeDoc([FakeEnt("Paris", "GPE")])           # full disagreement
    monkeypatch.setattr(ner, "_get_spacy", lambda: FakeNlp())

    chunks = [
        {"content": "Acme builds widgets.", "chunk_id": "a"},
        {"content": "Nothing matches here.", "chunk_id": "b"},
    ]
    gliner_entities = [{
        "id": "acme", "type": "Organization", "name": "Acme", "synonyms": [],
        "source_chunk_ids": ["a"], "score": 0.9,
        "mentions": [{"chunk_id": "a", "char_start": 0, "char_end": 4, "score": 0.9}],
    }]
    xc = ner.crosscheck_with_spacy(chunks, gliner_entities)
    per = {p["chunk_id"]: p for p in xc["per_chunk"]}
    assert per["a"]["disagreement"] == 0.0 and per["a"]["flagged"] is False
    assert per["b"]["disagreement"] == 1.0 and per["b"]["flagged"] is True
    assert xc["flagged_chunk_ids"] == ["b"]
    assert xc["mean_disagreement"] == 0.5
