"""
Hermes data models — workflow, step, artifact, event, and job-attempt records.

All state is represented as frozen (immutable-after-creation) dataclasses
so the store has single-point-of-truth for mutations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


# ── Enums ──────────────────────────────────────────────────────────────────────

class WorkflowStatus(str, Enum):
    created = "created"
    queued = "queued"
    running = "running"
    waiting = "waiting"
    retrying = "retrying"
    cancel_requested = "cancel_requested"
    cancelled = "cancelled"
    failed = "failed"
    completed = "completed"


class StepStatus(str, Enum):
    pending = "pending"
    ready = "ready"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    skipped = "skipped"
    blocked = "blocked"


class ErrorClass(str, Enum):
    transient = "transient"       # network timeout, provider failure
    recoverable = "recoverable"   # invalid LLM output, partial crawl
    configuration = "configuration"  # bad API key, missing connection
    fatal = "fatal"               # corrupted artifact, schema mismatch


# ── Core Data Models ────────────────────────────────────────────────────────────

@dataclass
class WorkflowRun:
    """
    One execution of a full scan pipeline.

    | column | type | notes |
    |--------|------|-------|
    | id | str | 'wf_' + uuid4 short |
    | project_id | str | project this workflow belongs to |
    | scan_id | str | scan this workflow drives |
    | status | WorkflowStatus | current state |
    | current_step | str or None | name of step being executed |
    | progress_percent | float | 0.0–100.0 |
    | created_at | str | ISO-8601 |
    | started_at | str or None | ISO-8601 |
    | finished_at | str or None | ISO-8601 |
    | metadata | dict | caller-supplied tags / config |
    """
    id: str
    project_id: str
    scan_id: str
    status: WorkflowStatus = WorkflowStatus.created
    current_step: Optional[str] = None
    progress_percent: float = 0.0
    created_at: str = field(default_factory=lambda: _now())
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class WorkflowStepRun:
    """
    One step within a workflow run.

    | column | type | notes |
    |--------|------|-------|
    | id | str | 'step_' + uuid4 short |
    | workflow_id | str | FK → workflow_run.id |
    | name | str | e.g. 'crawl_pages' |
    | status | StepStatus | current state |
    | attempt_count | int | 1-based |
    | depends_on | list[str] | step names that must succeed first |
    | idempotency_key | str | stable key for retry safety |
    | started_at | str or None | ISO-8601 |
    | finished_at | str or None | ISO-8601 |
    | error | dict or None | {class, message, details} |
    | input_artifacts | list[str] | artifact IDs consumed |
    | output_artifacts | list[str] | artifact IDs produced |
    """
    id: str
    workflow_id: str
    name: str
    status: StepStatus = StepStatus.pending
    attempt_count: int = 0
    depends_on: list[str] = field(default_factory=list)
    idempotency_key: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[dict] = None
    input_artifacts: list[str] = field(default_factory=list)
    output_artifacts: list[str] = field(default_factory=list)


@dataclass
class JobAttempt:
    """
    One attempt to execute a step.

    | column | type | notes |
    |--------|------|-------|
    | id | str | 'attempt_' + uuid4 short |
    | step_id | str | FK → step_run.id |
    | attempt_number | int | 1-based |
    | status | str | 'started', 'succeeded', 'failed' |
    | started_at | str | ISO-8601 |
    | finished_at | str or None | ISO-8601 |
    | error | dict or None | {class, message, details} |
    """
    id: str
    step_id: str
    attempt_number: int
    status: str = "started"
    started_at: str = field(default_factory=lambda: _now())
    finished_at: Optional[str] = None
    error: Optional[dict] = None


@dataclass
class Artifact:
    """
    An artifact produced or consumed by a workflow step.

    | column | type | notes |
    |--------|------|-------|
    | id | str | 'art_' + uuid4 short |
    | workflow_id | str | FK → workflow_run.id |
    | step_name | str | step that created it |
    | name | str | human-readable name |
    | type | str | 'crawl_results', 'chunks', 'embeddings', 'entities', etc. |
    | storage_path | str or None | JSON/file reference or inline JSON |
    | content_hash | str or None | sha256 of serialised content |
    | size_bytes | int or None | approximate byte size |
    | created_at | str | ISO-8601 |
    """
    id: str
    workflow_id: str
    step_name: str
    name: str
    type: str
    storage_path: Optional[str] = None
    content_hash: Optional[str] = None
    size_bytes: Optional[int] = None
    created_at: str = field(default_factory=lambda: _now())


@dataclass
class ProgressEvent:
    """
    One progress event emitted during a workflow.

    | column | type | notes |
    |--------|------|-------|
    | id | str | 'evt_' + uuid4 short |
    | workflow_id | str | FK |
    | scan_id | str | for direct lookup |
    | step_name | str or None | step that emitted it |
    | type | str | event type constant |
    | message | str | human-readable |
    | progress_percent | float | 0.0–100.0 |
    | created_at | str | ISO-8601 |
    """
    id: str
    workflow_id: str
    scan_id: str
    step_name: Optional[str] = None
    type: str = "workflow.step.progress"
    message: str = ""
    progress_percent: float = 0.0
    created_at: str = field(default_factory=lambda: _now())


@dataclass
class ModelUsageEvent:
    """
    Track model calls made during a step.

    | column | type | notes |
    |--------|------|-------|
    | id | str | 'mu_' + uuid4 short |
    | workflow_id | str | FK |
    | step_name | str | step that called the model |
    | provider | str | 'openai', 'anthropic', 'deepseek', etc. |
    | model | str | model name |
    | input_tokens | int | approximate |
    | output_tokens | int | approximate |
    | latency_ms | int | wall-clock |
    | cost_usd | float | estimated |
    | created_at | str | ISO-8601 |
    """
    id: str
    workflow_id: str
    step_name: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    created_at: str = field(default_factory=lambda: _now())


# ── Event Type Constants ───────────────────────────────────────────────────────

EVENT_WORKFLOW_CREATED   = "workflow.created"
EVENT_WORKFLOW_STARTED   = "workflow.started"
EVENT_WORKFLOW_STEP_STARTED  = "workflow.step.started"
EVENT_WORKFLOW_STEP_PROGRESS = "workflow.step.progress"
EVENT_WORKFLOW_STEP_SUCCEEDED = "workflow.step.succeeded"
EVENT_WORKFLOW_STEP_FAILED    = "workflow.step.failed"
EVENT_WORKFLOW_RETRYING  = "workflow.retrying"
EVENT_WORKFLOW_CANCELLED = "workflow.cancelled"
EVENT_WORKFLOW_COMPLETED = "workflow.completed"


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
