"""Trace contract test (Layer -1).

Enforces the brief's rule: every pipeline stage must emit at least one StepTrace.
Runs fully offline — the local embed/chat calls are monkeypatched so this test
needs no Ollama, no BGE-M3, and no network.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.trace import Trace, StepTrace  # noqa: E402


# The stages Layer 2 (src/simulator.py) is contracted to emit, in order.
# The test asserts none of the declared stages is silently empty.
EXPECTED_STAGES = [
    "intent", "query_expansion", "retriever", "rerank", "graph_traversal",
    "evidence_validation", "contradiction_detection", "confidence",
    "answer_synthesis", "citation_selection",
]


def test_step_context_records_step_and_latency():
    t = Trace(query="q")
    with t.start_step("demo", a=1) as st:
        st.outputs["b"] = 2
    assert t.step_names() == ["demo"]
    assert t.steps[0].latency_ms >= 0.0
    assert t.steps[0].inputs == {"a": 1}


def test_step_context_appends_even_on_error():
    t = Trace(query="q")
    try:
        with t.start_step("boom") as st:  # noqa: F841
            raise ValueError("kaboom")
    except ValueError:
        pass
    # The StepTrace must still be recorded, with the error noted.
    assert t.step_names() == ["boom"]
    assert any("ERROR" in n for n in t.steps[0].notes)


def test_dropped_requires_reason():
    st = StepTrace(name="x")
    st.drop({"chunk_id": "c1"}, "neutral entailment")
    assert st.dropped == [{"item": {"chunk_id": "c1"}, "reason": "neutral entailment"}]


def test_every_layer2_stage_emits_a_nonempty_steptrace(monkeypatch):
    """The core contract: each declared stage produces a StepTrace, none empty."""
    from test_layer2_layer3 import _run  # shared stubbed Layer-2 runner

    trace = _run(monkeypatch)

    # Every expected stage is present, in order...
    assert trace.step_names() == EXPECTED_STAGES
    # ...and no stage is a no-op (must have produced outputs, scores, or notes).
    for step in trace.steps:
        produced = bool(step.outputs) or bool(step.scores) or bool(step.notes) or bool(step.dropped)
        assert produced, f"stage '{step.name}' emitted an empty StepTrace"

    # Sanity: the trace actually carried retrieval + answer through.
    assert trace.retrieved and trace.answer


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
