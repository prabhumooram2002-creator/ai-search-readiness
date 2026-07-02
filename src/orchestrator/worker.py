"""
Hermes background worker — polls the workflow store for ready steps,
dispatches them to the appropriate step handler, and marks results.

The worker runs as an asyncio task.  It can be started alongside the
API server or as a standalone process.
"""

from __future__ import annotations
import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Optional

from .models import (
    WorkflowStatus, StepStatus, ErrorClass,
    Artifact, ModelUsageEvent,
    EVENT_WORKFLOW_STEP_STARTED, EVENT_WORKFLOW_STEP_PROGRESS,
)
from .storage import HermesStore, _new_id
from .engine import WorkflowEngine, StepDef
from src.core.logging import get_logger

logger = get_logger(__name__)


class HermesWorker:
    """
    Polls the store for workflows with steps in 'ready' state,
    dispatches their handlers, and records results.

    Usage::

        worker = HermesWorker(engine)
        worker.start()  # runs in background task
        ...
        worker.stop()
    """

    def __init__(
        self,
        engine: WorkflowEngine,
        poll_interval_s: float = 2.0,
        max_concurrent: int = 3,
    ):
        self.engine = engine
        self.store = engine.store
        self.poll_interval_s = poll_interval_s
        self.max_concurrent = max_concurrent
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def start(self) -> None:
        """Start the worker as a background asyncio task."""
        if self._task is not None and not self._task.done():
            return
        self._stop_event.clear()

        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._run_loop())
        except RuntimeError:
            # No running loop — start in a background thread
            import threading
            def _run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    new_loop.run_until_complete(self._run_loop())
                except Exception:
                    pass
            t = threading.Thread(target=_run_in_thread, daemon=True)
            t.start()

        logger.info(f"[Hermes Worker] Started (poll={self.poll_interval_s}s, "
                    f"concurrent={self.max_concurrent})")

    async def stop(self, wait: bool = True) -> None:
        """Stop the worker loop."""
        self._stop_event.set()
        if self._task and wait:
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        logger.info("[Hermes Worker] Stopped")

    async def _run_loop(self) -> None:
        """Main polling loop."""
        while not self._stop_event.is_set():
            try:
                await self._poll_once()
            except Exception as e:
                logger.error(f"[Hermes Worker] Poll error: {e}")
            # Sleep with early exit on stop
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.poll_interval_s,
                )
                break  # stop was requested
            except asyncio.TimeoutError:
                pass  # normal — poll again

    async def _poll_once(self) -> None:
        """Find all active workflows and dispatch ready steps."""
        # Get all running workflows
        workflows = self.store.list_workflows(status=WorkflowStatus.running.value)

        for wf in workflows:
            ready_step_names = self.engine.get_current_step_names(wf.id)
            if not ready_step_names:
                continue

            for step_name in ready_step_names:
                step = self.store.get_step_by_name(wf.id, step_name)
                if not step or step.status != StepStatus.ready:
                    continue

                # Check concurrency semaphore
                if self._semaphore.locked():
                    logger.debug(f"[Hermes Worker] At max concurrency, "
                                 f"deferring step '{step_name}'")
                    continue

                # Acquire semaphore and dispatch
                asyncio.create_task(self._execute_step(step))

    async def _execute_step(self, step) -> None:
        """Execute a single step, handling success/failure."""
        async with self._semaphore:
            step_def = self.engine.step_defs.get(step.name)
            if not step_def:
                logger.error(f"[Hermes Worker] No handler for step '{step.name}'")
                return

            # Mark step as running
            step = self.engine.mark_step_running(step.id)
            wf = self.store.get_workflow(step.workflow_id)
            self.store.update_workflow(wf.id, current_step=step.name)

            self.store.emit_event(
                workflow_id=wf.id, scan_id=wf.scan_id,
                type_=EVENT_WORKFLOW_STEP_STARTED,
                message=f"Step '{step.name}' started (attempt {step.attempt_count})",
                progress_percent=self.engine._calc_progress(wf.id),
                step_name=step.name,
            )

            # Gather input artifacts
            all_artifacts = self.store.get_artifacts(step.workflow_id)
            artifacts_map: dict[str, list[Artifact]] = {}
            for art in all_artifacts:
                artifacts_map.setdefault(art.step_name, []).append(art)

            # Run the handler
            try:
                handler = step_def.handler
                result = await asyncio.wait_for(
                    handler(step, wf, self.store, artifacts_map),
                    timeout=step_def.timeout_s,
                )
                artifacts, model_usage = result

                # Register output artifacts
                step = await self.engine.complete_step(
                    step.id, artifacts, model_usage,
                )
                logger.info(f"[Hermes Worker] Step '{step.name}' succeeded")

            except asyncio.TimeoutError:
                error = {
                    "class": "transient",
                    "message": f"Step '{step.name}' timed out after {step_def.timeout_s}s",
                    "details": {},
                }
                step = await self.engine.fail_step(step.id, error)
                logger.warning(f"[Hermes Worker] Step '{step.name}' timed out")

            except Exception as e:
                error_class = _classify_error(e)
                error = {
                    "class": error_class,
                    "message": str(e)[:500],
                    "details": {"type": type(e).__name__},
                }
                step = await self.engine.fail_step(step.id, error)
                logger.warning(f"[Hermes Worker] Step '{step.name}' failed: {e}")


# ── Error classification ────────────────────────────────────────────────────────

def _classify_error(exc: Exception) -> str:
    """Classify an exception into an ErrorClass string."""
    msg = str(exc).lower()
    exc_name = type(exc).__name__

    # Configuration errors
    if any(kw in msg for kw in ["api key", "invalid key", "no api", "bad config",
                                 "connection refused"]):
        return ErrorClass.configuration.value

    # Transient errors
    if any(kw in msg for kw in ["timeout", "connection reset", "temporary",
                                 "rate limit", "too many requests", "503",
                                 "502", "500", "service unavailable"]):
        return ErrorClass.transient.value

    # Fatal errors
    if any(kw in msg for kw in ["corrupted", "schema", "migration",
                                 "unsupported", "invalid state"]):
        return ErrorClass.fatal.value

    # Default: recoverable if it looks like a data/LLM issue, else transient
    if any(kw in msg for kw in ["invalid output", "parsing", "validation",
                                 "malformed", "schema validation"]):
        return ErrorClass.recoverable.value

    return ErrorClass.transient.value
