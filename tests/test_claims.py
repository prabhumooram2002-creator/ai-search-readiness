"""Layer 1 step 5 — claim extraction (src/claims.py).

Offline tests: the LLM is stubbed (wiring, sentence mapping, the CLAUDE.md
"every claim maps to a real sentence" rule, drop reasons, Trace emission).
pysbd runs for real — it's a light pure-python dep. The live-qwen run on real
input is logged in BUILD_LOG.md.
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src import claims as cl
from src.trace import Trace

TEXT = "Acme was founded in 2001. It employs 50 people. Contact us today!"
CHUNK = {"content": TEXT, "chunk_id": "c0"}


def test_segment_sentences_offsets_and_ids():
    sents = cl.segment_sentences(TEXT, "c0")
    assert [s["sentence_id"] for s in sents] == ["c0:s1", "c0:s2", "c0:s3"]
    assert sents[0]["text"] == "Acme was founded in 2001."
    # offsets are real slices of the source text
    for s in sents:
        assert TEXT[s["char_start"]:s["char_end"]].strip() == s["text"]


def test_claims_map_to_real_sentences(monkeypatch):
    def fake_llm(prompt, *, json=False, temperature=0, system=None, max_tokens=None):
        assert json is True and temperature == 0
        return ('{"claims": ['
                '{"claim": "Acme was founded in 2001", "sentence": 1},'
                '{"claim": "Acme employs 50 people", "sentence": 2}]}')
    monkeypatch.setattr(cl.providers, "llm_complete", fake_llm)

    trace = Trace(query="q")
    out = cl.extract_claims([CHUNK], trace=trace)

    assert len(out) == 2
    assert out[0]["source_sentence_id"] == "c0:s1"
    assert out[0]["source_chunk_id"] == "c0"
    assert out[0]["sentence_text"] == "Acme was founded in 2001."
    assert TEXT[out[0]["char_start"]:out[0]["char_end"]].strip() == out[0]["sentence_text"]
    assert out[0]["id"].startswith("claim_")
    # deterministic id
    assert out[0]["id"] == cl._claim_id("c0", 1, "Acme was founded in 2001")
    # Trace
    assert trace.step_names() == ["claim_extraction"]
    assert trace.steps[0].scores["n_claims"] == 2.0


def test_invalid_sentence_reference_dropped_with_reason(monkeypatch):
    def fake_llm(prompt, **kw):
        return ('{"claims": ['
                '{"claim": "Ghost claim", "sentence": 99},'
                '{"claim": "Stringy ref", "sentence": "2"},'
                '{"claim": "Acme was founded in 2001", "sentence": 1}]}')
    monkeypatch.setattr(cl.providers, "llm_complete", fake_llm)

    trace = Trace(query="q")
    out = cl.extract_claims([CHUNK], trace=trace)
    assert [c["text"] for c in out] == ["Acme was founded in 2001"]
    dropped = trace.steps[0].dropped
    assert len(dropped) == 2
    assert all(d["reason"] == "sentence reference not a real step-1 sentence" for d in dropped)


def test_fenced_json_tolerated(monkeypatch):
    def fake_llm(prompt, **kw):
        return '```json\n{"claims": [{"claim": "Acme employs 50 people", "sentence": 2}]}\n```'
    monkeypatch.setattr(cl.providers, "llm_complete", fake_llm)
    out = cl.extract_claims([CHUNK])
    assert len(out) == 1 and out[0]["source_sentence_id"] == "c0:s2"


def test_trace_step_appended_even_on_llm_error(monkeypatch):
    def boom(prompt, **kw):
        raise RuntimeError("ollama down")
    monkeypatch.setattr(cl.providers, "llm_complete", boom)

    trace = Trace(query="q")
    try:
        cl.extract_claims([CHUNK], trace=trace)
        assert False, "should have raised"
    except RuntimeError:
        pass
    assert trace.step_names() == ["claim_extraction"]
    assert any("ERROR" in n for n in trace.steps[0].notes)
