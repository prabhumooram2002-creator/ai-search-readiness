"""Layer 1 step 6 — NLI evidence extraction (src/evidence.py).

Offline tests with a stubbed cross-encoder: ±2-sentence window construction,
label mapping read from model config, two-pass premise scoring (source
sentence, then ±2 span — see BUILD_LOG step-6 diagnostic), the CLAUDE.md
contradiction flag-and-re-run rule, drop reasons, and Trace emission. The
live-model run is in BUILD_LOG.md.
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src import evidence as ev
from src.trace import Trace

# 5 sentences so the ±2 window is meaningful
TEXT = ("One fact here. Two facts here. The target sentence is here. "
        "Four facts here. Five facts here.")
CHUNK = {"content": TEXT, "chunk_id": "c0"}

ENTAIL = [0.0, 8.0, 0.0]
CONTRA = [8.0, 0.0, 0.0]
NEUTRAL = [0.0, 0.0, 8.0]


def _claim(cid, text, num):
    return {"id": cid, "text": text, "source_sentence_id": f"c0:s{num}",
            "source_chunk_id": "c0", "sentence_text": "", "char_start": 0, "char_end": 1}


class FakeConfig:
    id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}


class QueueNli:
    """Returns queued logit-row batches, one batch per predict() call."""
    def __init__(self, batches):
        self.batches = list(batches)
        self.model = type("M", (), {"config": FakeConfig()})()
        self.calls = []

    def predict(self, pairs, convert_to_numpy=True, show_progress_bar=False):
        self.calls.append(list(pairs))
        return self.batches.pop(0)


def test_evidence_span_is_plus_minus_two():
    sents = ev.segment_sentences(TEXT, "c0")
    assert len(sents) == 5
    span = ev._evidence_span(sents, 3)
    assert span == ("One fact here. Two facts here. The target sentence is here. "
                    "Four facts here. Five facts here.")
    # window clamps at the edges
    assert ev._evidence_span(sents, 1) == "One fact here. Two facts here. The target sentence is here."


def test_pass1_entailment_uses_source_sentence_premise(monkeypatch):
    fake = QueueNli([[ENTAIL]])
    monkeypatch.setattr(ev, "_get_nli", lambda: fake)

    trace = Trace(query="q")
    out = ev.extract_evidence([CHUNK], [_claim("cl1", "supported claim", 3)], trace=trace)
    rec = out[0]
    assert rec["entailment_label"] == "entailment"
    assert rec["entailment_confidence"] > 0.99
    assert rec["premise_used"] == "source_sentence"
    assert rec["contradiction_flagged"] is False
    # pass 1 premise was the exact source sentence, not the span
    assert fake.calls[0][0][0] == "The target sentence is here."
    # the recorded evidence_text is still the ±2 span
    assert rec["evidence_text"].startswith("One fact here.")
    assert trace.steps[0].outputs["label_counts"] == {"entailment": 1}


def test_neutral_pass1_retries_against_span(monkeypatch):
    fake = QueueNli([[NEUTRAL], [ENTAIL]])
    monkeypatch.setattr(ev, "_get_nli", lambda: fake)

    out = ev.extract_evidence([CHUNK], [_claim("cl1", "needs context", 3)])
    rec = out[0]
    assert rec["entailment_label"] == "entailment"
    assert rec["premise_used"] == "evidence_span"
    assert len(fake.calls) == 2
    assert fake.calls[1][0][0].startswith("One fact here.")  # span premise


def test_contradiction_flagged_and_rerun_unresolved(monkeypatch):
    # sentence: contra -> span: contra -> full-chunk re-run: contra
    fake = QueueNli([[CONTRA], [CONTRA], [CONTRA]])
    monkeypatch.setattr(ev, "_get_nli", lambda: fake)

    trace = Trace(query="q")
    out = ev.extract_evidence([CHUNK], [_claim("cl1", "false claim", 3)], trace=trace)
    rec = out[0]
    assert rec["contradiction_flagged"] is True
    assert rec["entailment_label"] == "contradiction"
    assert rec["rerun_label"] == "contradiction"
    assert len(fake.calls) == 3
    assert fake.calls[2][0][0] == TEXT                        # re-run premise = full chunk
    assert trace.steps[0].outputs["contradiction_flags"] == ["cl1"]


def test_contradiction_rerun_resolved_keeps_flag(monkeypatch):
    # sentence: contra -> span: contra -> full-chunk re-run: entailment
    fake = QueueNli([[CONTRA], [CONTRA], [ENTAIL]])
    monkeypatch.setattr(ev, "_get_nli", lambda: fake)

    out = ev.extract_evidence([CHUNK], [_claim("cl1", "ambiguous claim", 3)])
    rec = out[0]
    assert rec["entailment_label"] == "entailment"   # wider context won
    assert rec["contradiction_flagged"] is True       # but the flag survives
    assert rec["rerun_label"] == "entailment"


def test_bad_claims_dropped_with_reason(monkeypatch):
    fake = QueueNli([[ENTAIL]])
    monkeypatch.setattr(ev, "_get_nli", lambda: fake)

    claims = [
        _claim("ok", "good", 3),
        {**_claim("bad-chunk", "x", 1), "source_chunk_id": "nope"},
        {**_claim("bad-sent", "y", 1), "source_sentence_id": "c0:s99"},
        {**_claim("bad-id", "z", 1), "source_sentence_id": "garbage"},
    ]
    trace = Trace(query="q")
    out = ev.extract_evidence([CHUNK], claims, trace=trace)
    assert [r["claim_id"] for r in out] == ["ok"]
    reasons = sorted(d["reason"] for d in trace.steps[0].dropped)
    assert reasons == ["source chunk not provided",
                       "source sentence not found in chunk",
                       "unparseable source_sentence_id"]


def test_trace_step_appended_even_on_model_error(monkeypatch):
    class Boom:
        model = type("M", (), {"config": FakeConfig()})()
        def predict(self, *a, **k):
            raise RuntimeError("nli exploded")
    monkeypatch.setattr(ev, "_get_nli", lambda: Boom())

    trace = Trace(query="q")
    try:
        ev.extract_evidence([CHUNK], [_claim("cl1", "any", 3)], trace=trace)
        assert False, "should have raised"
    except RuntimeError:
        pass
    assert trace.step_names() == ["evidence_extraction"]
    assert any("ERROR" in n for n in trace.steps[0].notes)
