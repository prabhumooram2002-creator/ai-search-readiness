"""
Hermes workflow engine — state machine driving step lifecycle.

Transitions:
    pending ──(deps satisfied)──> ready ──(worker picks up)──> running
    running ──(success)──> succeeded
    running ──(transient error)──> ready (retry increment)
    running ──(fatal/config error)──> failed
    running ──(recoverable, retries exhausted)──> failed
    pending/ready ──(cancel)──> skipped
"""

from __future__ import annotations
import asyncio
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional

from .models import (
    WorkflowRun, WorkflowStatus, WorkflowStepRun, StepStatus,
    ErrorClass, Artifact, JobAttempt, ModelUsageEvent,
    EVENT_WORKFLOW_STARTED, EVENT_WORKFLOW_STEP_STARTED,
    EVENT_WORKFLOW_STEP_SUCCEEDED, EVENT_WORKFLOW_STEP_FAILED,
    EVENT_WORKFLOW_RETRYING, EVENT_WORKFLOW_CANCELLED,
    EVENT_WORKFLOW_COMPLETED,
)
from .storage import HermesStore, _new_id

# Signature for a step handler function.
# Receives: (step_run, workflow_run, store, artifacts_map)
# Returns: (output_artifacts: list[Artifact], model_usage: list[ModelUsageEvent])
StepHandler = Callable[
    [WorkflowStepRun, WorkflowRun, HermesStore, dict],
    Awaitable[tuple[list[Artifact], list[ModelUsageEvent]]],
]

# ── Retry policy ────────────────────────────────────────────────────────────────

class RetryPolicy:
    """Per-step retry configuration."""
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay_s: float = 2.0,
        max_delay_s: float = 120.0,
        backoff_factor: float = 2.0,
    ):
        self.max_attempts = max_attempts
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        self.backoff_factor = backoff_factor

    def delay(self, attempt: int) -> float:
        """Exponential backoff with jitter."""
        import random
        d = min(self.base_delay_s * (self.backoff_factor ** (attempt - 1)),
                self.max_delay_s)
        return d * (0.5 + random.random() * 0.5)


DEFAULT_RETRY = RetryPolicy(max_attempts=3)

# ── Step definition ─────────────────────────────────────────────────────────────

class StepDef:
    """Declarative step definition — name, deps, handler, policy."""
    def __init__(
        self,
        name: str,
        handler: StepHandler,
        depends_on: list[str] | None = None,
        retry_policy: RetryPolicy | None = None,
        timeout_s: int = 600,
    ):
        self.name = name
        self.handler = handler
        self.depends_on = depends_on or []
        self.retry_policy = retry_policy or DEFAULT_RETRY
        self.timeout_s = timeout_s

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "depends_on": self.depends_on,
            "idempotency_key_template": "{scan_id}:{name}:v1",
        }


# ── Workflow Engine ─────────────────────────────────────────────────────────────

class WorkflowEngine:
    """
    Stateless workflow state machine.

    Usage::

        engine = WorkflowEngine(store, [step1, step2, ...])
        wf = engine.create_workflow("proj_1", "scan_1")
        engine.start_workflow(wf.id)
        steps = engine.tick(wf.id)  # returns ready-to-run steps
        # worker runs the steps, calls engine.complete_step / engine.fail_step
    """

    def __init__(self, store: HermesStore, step_defs: list[StepDef]):
        self.store = store
        self.step_defs = {sd.name: sd for sd in step_defs}
        self.step_names = [sd.name for sd in step_defs]

    # ── Orchestration ──────────────────────────────────────────────────────

    def create_workflow(
        self, project_id: str, scan_id: str,
        metadata: dict | None = None,
    ) -> WorkflowRun:
        wf = self.store.create_workflow(project_id, scan_id, metadata)
        # Auto-create step runs from definitions
        step_defs_data = [
            {
                "name": sd.name,
                "depends_on": sd.depends_on,
                "idempotency_key": f"{scan_id}:{sd.name}:v1",
            }
            for sd in self.step_defs.values()
        ]
        self.store.create_steps_from_definitions(wf.id, step_defs_data)
        return wf

    def start_workflow(self, workflow_id: str) -> WorkflowRun:
        now = datetime.now(timezone.utc).isoformat()
        wf = self.store.update_workflow(
            workflow_id,
            status=WorkflowStatus.running,
            started_at=now,
        )
        self.store.emit_event(
            workflow_id=workflow_id,
            scan_id=wf.scan_id,
            type_=EVENT_WORKFLOW_STARTED,
            message=f"Workflow {workflow_id} started",
        )
        return wf

    def get_current_step_names(self, workflow_id: str) -> list[str]:
        """Return names of steps that are currently ready to run."""
        steps = self.store.get_workflow_steps(workflow_id)
        ready = []
        for step in steps:
            if step.status not in (StepStatus.pending, StepStatus.ready):
                continue
            # Check deps
            deps_met = self._check_deps(step, steps)
            if deps_met:
                if step.status == StepStatus.pending:
                    self.store.update_step(step.id, status=StepStatus.ready)
                ready.append(step.name)
        return ready

    def mark_step_running(self, step_id: str) -> WorkflowStepRun:
        now = datetime.now(timezone.utc).isoformat()
        step = self.store.get_step(step_id)
        if not step:
            raise ValueError(f"Step {step_id} not found")
        attempt = step.attempt_count + 1
        self.store.update_step(
            step_id,
            status=StepStatus.running,
            started_at=now,
            attempt_count=attempt,
        )
        # Create attempt record
        self.store.create_attempt(step_id, attempt)
        return self.store.get_step(step_id)

    async def complete_step(
        self, step_id: str,
        artifacts: list[Artifact] | None = None,
        model_usage: list[ModelUsageEvent] | None = None,
    ) -> WorkflowStepRun:
        now = datetime.now(timezone.utc).isoformat()
        step = self.store.get_step(step_id)

        # Mark attempt succeeded
        attempts = self.store._conn().execute(
            "SELECT id FROM job_attempt WHERE step_id=? AND status='started' "
            "ORDER BY attempt_number DESC LIMIT 1",
            (step_id,),
        ).fetchone()
        if attempts:
            self.store.update_attempt(attempts["id"], status="succeeded",
                                      finished_at=now)

        artifact_ids = []
        for art in (artifacts or []):
            self.store.add_artifact(art)
            artifact_ids.append(art.id)

        for mu in (model_usage or []):
            self.store.add_model_usage(mu)

        self.store.update_step(
            step_id, status=StepStatus.succeeded,
            finished_at=now, output_artifacts=artifact_ids,
        )
        step = self.store.get_step(step_id)

        wf = self.store.get_workflow(step.workflow_id)
        self.store.emit_event(
            workflow_id=wf.id,
            scan_id=wf.scan_id,
            type_=EVENT_WORKFLOW_STEP_SUCCEEDED,
            message=f"Step '{step.name}' succeeded",
            progress_percent=self._calc_progress(wf.id),
            step_name=step.name,
        )

        # Update workflow progress
        progress = self._calc_progress(wf.id)
        wf = self.store.update_workflow(wf.id, progress_percent=progress)
        if progress >= 100.0:
            self._complete_workflow(wf.id)
        return step

    async def fail_step(
        self, step_id: str, error: dict,
    ) -> WorkflowStepRun:
        now = datetime.now(timezone.utc).isoformat()
        step = self.store.get_step(step_id)

        # Mark attempt failed
        attempts = self.store._conn().execute(
            "SELECT id FROM job_attempt WHERE step_id=? AND status='started' "
            "ORDER BY attempt_number DESC LIMIT 1",
            (step_id,),
        ).fetchone()
        if attempts:
            self.store.update_attempt(attempts["id"],
                                      status="failed", finished_at=now,
                                      error_json=json.dumps(error))

        error_class = error.get("class", "transient")
        attempt_count = step.attempt_count
        step_def = self.step_defs.get(step.name)

        # Check if we should retry
        can_retry = (
            error_class in ("transient", "recoverable")
            and step_def
            and attempt_count < step_def.retry_policy.max_attempts
        )

        if can_retry:
            # Set back to ready for retry (worker will handle delay)
            delay_s = step_def.retry_policy.delay(attempt_count + 1)
            self.store.update_step(
                step_id, status=StepStatus.ready,
                error=error,
            )
            wf = self.store.get_workflow(step.workflow_id)
            self.store.emit_event(
                workflow_id=wf.id, scan_id=wf.scan_id,
                type_=EVENT_WORKFLOW_RETRYING,
                message=(f"Step '{step.name}' failed ({error_class}), "
                         f"retry {attempt_count}/{step_def.retry_policy.max_attempts} "
                         f"in {delay_s:.0f}s — {error.get('message', '')}"),
                progress_percent=wf.progress_percent,
                step_name=step.name,
            )
            return self.store.get_step(step_id)
        else:
            # Final failure
            self.store.update_step(
                step_id, status=StepStatus.failed,
                finished_at=now, error=error,
            )
            wf = self.store.get_workflow(step.workflow_id)
            self.store.update_workflow(
                wf.id, status=WorkflowStatus.failed,
                current_step=step.name,
            )
            self.store.emit_event(
                workflow_id=wf.id, scan_id=wf.scan_id,
                type_=EVENT_WORKFLOW_STEP_FAILED,
                message=f"Step '{step.name}' failed — {error.get('message', '')}",
                progress_percent=wf.progress_percent,
                step_name=step.name,
            )
            return self.store.get_step(step_id)

    def cancel_workflow(self, workflow_id: str) -> WorkflowRun:
        wf = self.store.get_workflow(workflow_id)
        if wf and wf.status in (WorkflowStatus.running, WorkflowStatus.queued,
                                 WorkflowStatus.created, WorkflowStatus.waiting):
            # Mark all pending/ready steps as skipped
            steps = self.store.get_workflow_steps(workflow_id)
            for step in steps:
                if step.status in (StepStatus.pending, StepStatus.ready):
                    self.store.update_step(
                        step.id, status=StepStatus.skipped,
                        finished_at=datetime.now(timezone.utc).isoformat(),
                    )
            now = datetime.now(timezone.utc).isoformat()
            wf = self.store.update_workflow(
                workflow_id, status=WorkflowStatus.cancelled,
                finished_at=now,
            )
            self.store.emit_event(
                workflow_id=workflow_id, scan_id=wf.scan_id,
                type_=EVENT_WORKFLOW_CANCELLED,
                message=f"Workflow {workflow_id} cancelled",
                progress_percent=wf.progress_percent,
            )
        return wf

    def retry_failed_workflow(self, workflow_id: str) -> WorkflowRun:
        wf = self.store.get_workflow(workflow_id)
        if wf and wf.status == WorkflowStatus.failed:
            steps = self.store.get_workflow_steps(workflow_id)

            # Find failed step, reset it to pending (clears its deps blockers too)
            for step in steps:
                if step.status == StepStatus.failed:
                    self.store.update_step(
                        step.id, status=StepStatus.pending,
                        error=None, finished_at=None,
                    )
                    break

            # Reset all steps after the failed one
            found_failed = False
            for step in steps:
                if step.name == step.name and step.status == StepStatus.failed:
                    found_failed = True
                elif found_failed and step.status == StepStatus.skipped:
                    self.store.update_step(
                        step.id, status=StepStatus.pending,
                        finished_at=None,
                    )

            self.store.update_workflow(
                workflow_id, status=WorkflowStatus.running,
                current_step=None, finished_at=None,
            )
        return self.store.get_workflow(workflow_id)

    # ── Internal Helpers ───────────────────────────────────────────────────

    def _check_deps(self, step: WorkflowStepRun, all_steps: list) -> bool:
        if not step.depends_on:
            return True
        dep_names = set(step.depends_on)
        for s in all_steps:
            if s.name in dep_names:
                if s.status != StepStatus.succeeded:
                    return False
        return True

    def _calc_progress(self, workflow_id: str) -> float:
        steps = self.store.get_workflow_steps(workflow_id)
        if not steps:
            return 0.0
        completed = sum(
            1 for s in steps
            if s.status in (StepStatus.succeeded, StepStatus.skipped)
        )
        return round((completed / len(steps)) * 100, 1)

    def _complete_workflow(self, workflow_id: str) -> WorkflowRun:
        now = datetime.now(timezone.utc).isoformat()
        wf = self.store.update_workflow(
            workflow_id, status=WorkflowStatus.completed,
            finished_at=now, progress_percent=100.0,
        )
        self.store.emit_event(
            workflow_id=workflow_id, scan_id=wf.scan_id,
            type_=EVENT_WORKFLOW_COMPLETED,
            message=f"Workflow {workflow_id} completed",
            progress_percent=100.0,
        )
        return wf


# Lazy import for json in fail_step
import json
