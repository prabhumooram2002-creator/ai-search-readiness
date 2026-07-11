"""
Hermes public API — scan lifecycle, workflow management, artifact access.
"""

from __future__ import annotations
import asyncio
import json
import uuid
from pathlib import Path
from typing import Optional

from .models import (
    WorkflowRun, WorkflowStatus, WorkflowStepRun, StepStatus,
    ProgressEvent, Artifact, ModelUsageEvent,
    EVENT_WORKFLOW_CREATED, EVENT_WORKFLOW_STARTED,
)
from .storage import HermesStore, _new_id
from .engine import WorkflowEngine, StepDef, RetryPolicy
from .worker import HermesWorker

from src.core.logging import get_logger
logger = get_logger(__name__)


# ── Default step definitions ─────────────────────────────────────────────────────

DEFAULT_SCAN_STEPS = [
    StepDef(
        name="crawl_pages",
        handler=None,  # filled in below to avoid circular import
        depends_on=[],
        retry_policy=RetryPolicy(max_attempts=3, base_delay_s=5, max_delay_s=60),
        timeout_s=600,
    ),
    StepDef(
        name="normalize_content",
        handler=None,
        depends_on=["crawl_pages"],
        retry_policy=RetryPolicy(max_attempts=2),
        timeout_s=300,
    ),
    StepDef(
        name="chunk_content",
        handler=None,
        depends_on=["normalize_content"],
        retry_policy=RetryPolicy(max_attempts=2),
        timeout_s=300,
    ),
    StepDef(
        name="extract_semantics",
        handler=None,
        depends_on=["chunk_content"],
        retry_policy=RetryPolicy(max_attempts=3, base_delay_s=3),
        timeout_s=600,
    ),
    StepDef(
        name="generate_embeddings",
        handler=None,
        depends_on=["chunk_content"],
        retry_policy=RetryPolicy(max_attempts=3),
        timeout_s=600,
    ),
    StepDef(
        name="upsert_vectors",
        handler=None,
        depends_on=["generate_embeddings"],
        retry_policy=RetryPolicy(max_attempts=2),
        timeout_s=120,
    ),
    StepDef(
        name="upsert_graph",
        handler=None,
        depends_on=["extract_semantics"],
        retry_policy=RetryPolicy(max_attempts=2),
        timeout_s=120,
    ),
    StepDef(
        name="run_graphrag",
        handler=None,
        depends_on=["upsert_vectors", "upsert_graph"],
        retry_policy=RetryPolicy(max_attempts=3),
        timeout_s=600,
    ),
    StepDef(
        name="run_semantic_repair",
        handler=None,
        depends_on=["run_graphrag"],
        retry_policy=RetryPolicy(max_attempts=2),
        timeout_s=600,
    ),
    StepDef(
        name="run_reasoning_simulation",
        handler=None,
        depends_on=["run_semantic_repair"],
        retry_policy=RetryPolicy(max_attempts=2),
        timeout_s=300,
    ),
    StepDef(
        name="generate_recommendations",
        handler=None,
        depends_on=["run_reasoning_simulation"],
        retry_policy=RetryPolicy(max_attempts=2),
        timeout_s=120,
    ),
    StepDef(
        name="finalize_scan",
        handler=None,
        depends_on=["generate_recommendations"],
        retry_policy=RetryPolicy(max_attempts=3),
        timeout_s=60,
    ),
]


def _wire_handlers(steps: list[StepDef]) -> list[StepDef]:
    """Attach handler functions to step definitions (late-binding to avoid circular imports)."""
    from . import steps as step_handlers

    handler_map = {
        "crawl_pages": step_handlers.step_crawl_pages,
        "normalize_content": step_handlers.step_normalize_content,
        "chunk_content": step_handlers.step_chunk_content,
        "extract_semantics": step_handlers.step_extract_semantics,
        "generate_embeddings": step_handlers.step_generate_embeddings,
        "upsert_vectors": step_handlers.step_upsert_vectors,
        "upsert_graph": step_handlers.step_upsert_graph,
        "run_graphrag": step_handlers.step_run_graphrag,
        "run_semantic_repair": step_handlers.step_run_semantic_repair,
        "run_reasoning_simulation": step_handlers.step_run_reasoning_simulation,
        "generate_recommendations": step_handlers.step_generate_recommendations,
        "finalize_scan": step_handlers.step_finalize_scan,
    }
    for sd in steps:
        sd.handler = handler_map.get(sd.name)
        # Also update idempotency_key_template
    return steps


# ── Hermes API ───────────────────────────────────────────────────────────────────

class HermesAPI:
    """
    Public API for Hermes orchestration.

    Usage (CLI)::

        api = HermesAPI()
        wf = api.create_scan(
            project_id="my_project",
            target_url="https://example.com",
            queries=["query1", "query2"],
        )
        api.start_worker()
        # ... let worker run ...
        wf = api.get_workflow(wf.id)
        print(wf.status)
        recs = api.get_recommendations(wf.scan_id)
    """

    def __init__(self, db_path: str = "data/hermes.db"):
        self.store = HermesStore(db_path)
        step_defs = _wire_handlers(DEFAULT_SCAN_STEPS)
        self.engine = WorkflowEngine(self.store, step_defs)
        self.worker = HermesWorker(self.engine)
        self._worker_started = False

    def close(self) -> None:
        """Release resources (the underlying DB connection). Safe to call twice."""
        self.store.close()

    # ── Scan Lifecycle ─────────────────────────────────────────────────────

    def create_scan(
        self,
        project_id: str = "default",
        target_url: str = "",
        queries: list[str] | None = None,
        import_file: str | None = None,
        max_pages: int = 100,
        metadata: dict | None = None,
    ) -> dict:
        """
        Create a new scan and its workflow.

        Returns immediately with the workflow record.
        Background work starts when ``start_worker()`` is called.
        """
        scan_id = _new_id("scan_")
        wf_meta = {
            "target_url": target_url,
            "queries": queries or [],
            "import_file": import_file or "",
            "max_pages": max_pages,
            **(metadata or {}),
        }

        wf = self.engine.create_workflow(project_id, scan_id, metadata=wf_meta)

        return self._workflow_to_dict(wf)

    def get_scan_status(self, scan_id: str) -> dict | None:
        """Get the current status of a scan via its workflow."""
        wf = self.store.get_workflow_by_scan(scan_id)
        if not wf:
            return None

        steps = self.store.get_workflow_steps(wf.id)
        step_statuses = {
            s.name: {
                "status": s.status.value,
                "attempt_count": s.attempt_count,
                "error": s.error,
            }
            for s in steps
        }

        return {
            "scan_id": scan_id,
            "workflow_id": wf.id,
            "status": wf.status.value,
            "current_step": wf.current_step,
            "progress_percent": wf.progress_percent,
            "created_at": wf.created_at,
            "started_at": wf.started_at,
            "finished_at": wf.finished_at,
            "steps": step_statuses,
            "metadata": wf.metadata,
        }

    def get_workflow(self, workflow_id: str) -> dict | None:
        """Get full workflow details including steps."""
        return self._get_workflow_dict(workflow_id)

    def list_scans(
        self, project_id: str | None = None,
        limit: int = 20, offset: int = 0,
    ) -> list[dict]:
        """List recent scans / workflows."""
        wfs = self.store.list_workflows(
            project_id=project_id, limit=limit, offset=offset,
        )
        return [self._workflow_to_dict(w) for w in wfs]

    def cancel_scan(self, scan_id: str) -> dict | None:
        """Cancel an active scan."""
        wf = self.store.get_workflow_by_scan(scan_id)
        if not wf:
            return None
        wf = self.engine.cancel_workflow(wf.id)
        return self._workflow_to_dict(wf)

    def retry_scan(self, scan_id: str) -> dict | None:
        """Retry a failed scan from its failed step."""
        wf = self.store.get_workflow_by_scan(scan_id)
        if not wf:
            return None
        wf = self.engine.retry_failed_workflow(wf.id)
        return self._get_workflow_dict(wf.id)

    def start_scan(self, scan_id: str) -> dict | None:
        """Explicitly start a created/queued scan's workflow."""
        wf = self.store.get_workflow_by_scan(scan_id)
        if not wf:
            return None
        if wf.status in (WorkflowStatus.created, WorkflowStatus.queued):
            self.engine.start_workflow(wf.id)
        return self._get_workflow_dict(wf.id)

    # ── Worker Control ─────────────────────────────────────────────────────

    def start_worker(self) -> None:
        """Start the background worker if not already running."""
        if not self._worker_started:
            self.worker.start()
            self._worker_started = True

    def stop_worker(self) -> None:
        """Stop the background worker."""
        if self._worker_started:
            try:
                # Can't await from sync — create task on the running loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self.worker.stop())
                else:
                    loop.run_until_complete(self.worker.stop())
            except RuntimeError:
                pass
            self._worker_started = False

    # ── Data Access ────────────────────────────────────────────────────────

    def get_events(
        self, scan_id: str, limit: int = 100, offset: int = 0,
    ) -> list[dict]:
        """Get progress events for a scan."""
        evts = self.store.get_events(scan_id=scan_id, limit=limit, offset=offset)
        return [dict(e.__dict__) for e in evts]

    def get_artifacts(self, scan_id: str) -> list[dict]:
        """Get all artifacts created during a scan."""
        wf = self.store.get_workflow_by_scan(scan_id)
        if not wf:
            return []
        arts = self.store.get_artifacts(wf.id)
        return [dict(a.__dict__) for a in arts]

    def get_recommendations(self, scan_id: str) -> list[dict]:
        """Get the final recommendations for a scan."""
        arts = self.get_artifacts(scan_id)
        for art in arts:
            if art["type"] == "recommendation_output" and art.get("storage_path"):
                path = Path(art["storage_path"])
                if path.exists():
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
        return []

    def get_simulation_results(self, scan_id: str) -> list[dict]:
        """Get simulation results for a scan."""
        arts = self.get_artifacts(scan_id)
        for art in arts:
            if art["type"] == "simulation_output" and art.get("storage_path"):
                path = Path(art["storage_path"])
                if path.exists():
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
        return []

    def get_model_usage(self, scan_id: str) -> list[dict]:
        """Get model usage/cost data for a scan."""
        wf = self.store.get_workflow_by_scan(scan_id)
        if not wf:
            return []
        usage = self.store.get_model_usage(wf.id)
        return [dict(u.__dict__) for u in usage]

    # ── Internal ───────────────────────────────────────────────────────────

    def _get_workflow_dict(self, workflow_id: str) -> dict | None:
        wf = self.store.get_workflow(workflow_id)
        if not wf:
            return None
        steps = self.store.get_workflow_steps(workflow_id)
        return {
            **self._workflow_to_dict(wf),
            "steps": [
                {
                    "id": s.id,
                    "name": s.name,
                    "status": s.status.value,
                    "attempt_count": s.attempt_count,
                    "depends_on": s.depends_on,
                    "error": s.error,
                    "started_at": s.started_at,
                    "finished_at": s.finished_at,
                }
                for s in steps
            ],
        }

    @staticmethod
    def _workflow_to_dict(wf: WorkflowRun) -> dict:
        return {
            "workflow_id": wf.id,
            "scan_id": wf.scan_id,
            "project_id": wf.project_id,
            "status": wf.status.value,
            "current_step": wf.current_step,
            "progress_percent": wf.progress_percent,
            "created_at": wf.created_at,
            "started_at": wf.started_at,
            "finished_at": wf.finished_at,
            "metadata": wf.metadata,
        }
