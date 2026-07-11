"""Trace object — the backbone of the traced pipeline (Layer -1 remediation).

Every Layer 0-2 node appends a StepTrace to the per-query Trace. Layer 3 is pure
introspection over this object — no new inference. A build-time test enforces that
each pipeline stage emits at least one StepTrace (see tests/test_trace_contract.py).

This module has ZERO heavy dependencies (no torch/spaCy/etc.) so it can be imported
by any stage and by tests without pulling the model stack.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class StepTrace:
    """One pipeline node's contribution to a query's Trace."""
    name: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    def note(self, msg: str) -> "StepTrace":
        self.notes.append(msg)
        return self

    def drop(self, item: Any, reason: str) -> "StepTrace":
        """Record a candidate dropped at this stage, with the reason logged at drop time.

        Layer 3 ("which chunks lost confidence?") reads these — always give a reason.
        """
        self.dropped.append({"item": item, "reason": reason})
        return self


@dataclass
class Trace:
    """One Trace per query. Threaded through Layers 0-2, introspected by Layer 3."""
    query: str = ""
    intent: Optional[dict[str, Any]] = None
    constraints: dict[str, Any] = field(default_factory=dict)
    expanded_queries: list[str] = field(default_factory=list)

    retrieved: list[dict[str, Any]] = field(default_factory=list)   # {chunk_id, bm25, embed, combined}
    reranked: list[dict[str, Any]] = field(default_factory=list)    # {chunk_id, rerank}
    graph_paths: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)    # {claim_id, chunk_id, nli_label, nli_conf}
    contradictions: list[dict[str, Any]] = field(default_factory=list)

    confidence: Optional[float] = None
    confidence_breakdown: dict[str, float] = field(default_factory=dict)

    answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)   # {sentence_span, chunk_id, page_url}
    unsupported_sentences: list[str] = field(default_factory=list)

    # ── Layer-3 rollups (derived, not inferred) ──────────────────────────────
    missing_entities: list[str] = field(default_factory=list)
    missing_relationships: list[str] = field(default_factory=list)
    weak_evidence: list[dict[str, Any]] = field(default_factory=list)
    circular_refs: list[Any] = field(default_factory=list)
    retrieval_dead_ends: list[str] = field(default_factory=list)
    trust_bottlenecks: list[dict[str, Any]] = field(default_factory=list)
    citation_probability_simulated: Optional[float] = None
    citation_probability_observed: Optional[float] = None  # only if real engine runs were logged

    # ── The ordered record of every stage that touched this query ─────────────
    steps: list[StepTrace] = field(default_factory=list)

    def start_step(self, name: str, **inputs: Any) -> "StepContext":
        """Context manager that times a node and appends its StepTrace on exit.

        Usage:
            with trace.start_step("retriever", query=q) as st:
                ...
                st.outputs["candidates"] = candidates
        """
        return StepContext(self, name, inputs)

    def add_step(self, step: StepTrace) -> StepTrace:
        self.steps.append(step)
        return step

    def step_names(self) -> list[str]:
        return [s.name for s in self.steps]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StepContext:
    """Times a stage and guarantees a StepTrace is appended even on error."""

    def __init__(self, trace: Trace, name: str, inputs: dict[str, Any]):
        self._trace = trace
        self.step = StepTrace(name=name, inputs=dict(inputs))
        self._t0 = 0.0

    def __enter__(self) -> StepTrace:
        self._t0 = time.perf_counter()
        return self.step

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.step.latency_ms = (time.perf_counter() - self._t0) * 1000.0
        if exc is not None:
            self.step.note(f"ERROR: {exc_type.__name__}: {exc}")
        self._trace.add_step(self.step)
        return False  # never suppress exceptions
