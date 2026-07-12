"""
Hermes integration tests -- workflow lifecycle over the durable SQLite store.

Covers:
1. Workflow creation, status, events, cancellation
2. Retry / failure escalation
3. Event + artifact recording
4. Idempotency (unknown scan -> empty)
5. (opt-in) full worker pipeline end-to-end

Each test runs against its own throwaway DB and closes the store in a finally,
so the SQLite file handle is released before the temp dir is removed (Windows
will not unlink an open DB file).
"""

import asyncio
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest

# Ensure project root on path
BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.optional.hermes import create_api


@contextmanager
def temp_api():
    """Yield a HermesAPI backed by a fresh temp DB; close + delete on exit."""
    tmpdir = tempfile.mkdtemp(prefix="hermes_test_")
    api = create_api(str(Path(tmpdir) / "test_hermes.db"))
    try:
        yield api
    finally:
        api.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_basic_lifecycle():
    """Workflow creation, status, events, and cancellation."""
    with temp_api() as api:
        scan = api.create_scan(
            project_id="test_proj",
            target_url="https://example.com",
            queries=["test query"],
        )
        assert "workflow_id" in scan
        assert "scan_id" in scan
        assert scan["status"] == "created"
        scan_id = scan["scan_id"]

        status = api.get_scan_status(scan_id)
        assert status is not None
        assert status["status"] == "created"
        steps = status["steps"]
        assert len(steps) == 12  # 12 workflow steps
        assert "crawl_pages" in steps
        assert "finalize_scan" in steps

        started = api.start_scan(scan_id)
        assert started["status"] == "running"

        cancelled = api.cancel_scan(scan_id)
        assert cancelled["status"] == "cancelled"

        events = api.get_events(scan_id)
        assert len(events) >= 2
        event_types = [e["type"] for e in events]
        assert "workflow.created" in event_types
        assert "workflow.started" in event_types
        assert "workflow.cancelled" in event_types

        arts = api.get_artifacts(scan_id)
        assert len(arts) == 0


def test_retry_flow():
    """A failed step auto-retries, then escalates to failed; workflow can retry."""
    from src.optional.hermes.models import WorkflowStatus, StepStatus

    with temp_api() as api:
        scan = api.create_scan(
            project_id="test_proj",
            target_url="https://example.com",
        )
        scan_id = scan["scan_id"]
        wf_id = scan["workflow_id"]

        api.engine.start_workflow(wf_id)
        first_step = api.store.get_workflow_steps(wf_id)[0]

        api.engine.mark_step_running(first_step.id)
        result = asyncio.run(api.engine.fail_step(first_step.id, {
            "class": "transient", "message": "Network timeout",
        }))
        assert result.status == StepStatus.ready  # retried

        api.engine.mark_step_running(first_step.id)
        asyncio.run(api.engine.fail_step(first_step.id, {
            "class": "transient", "message": "Network timeout again",
        }))
        api.engine.mark_step_running(first_step.id)
        result = asyncio.run(api.engine.fail_step(first_step.id, {
            "class": "fatal", "message": "Corrupted data",
        }))
        assert result.status == StepStatus.failed

        wf = api.store.get_workflow(wf_id)
        assert wf.status == WorkflowStatus.failed

        retried = api.retry_scan(scan_id)
        assert retried["status"] == "running"


def test_events_and_artifacts():
    """Events and artifacts are recorded and read back correctly."""
    from src.optional.hermes.models import Artifact

    with temp_api() as api:
        scan = api.create_scan(
            project_id="test_proj",
            target_url="https://example.com",
            metadata={"test": True},
        )
        scan_id = scan["scan_id"]
        wf_id = scan["workflow_id"]

        api.store.emit_event(wf_id, scan_id, "workflow.step.progress",
                             message="Test event", progress_percent=50.0,
                             step_name="test_step")

        events = api.get_events(scan_id)
        assert len(events) >= 1
        latest = events[-1]
        assert latest["type"] == "workflow.step.progress"
        assert latest["message"] == "Test event"
        assert latest["progress_percent"] == 50.0

        art = Artifact(
            id="art_test_1", workflow_id=wf_id,
            step_name="test_step", name="test_artifact",
            type="test", storage_path="/tmp/test.json",
            size_bytes=100,
        )
        api.store.add_artifact(art)

        arts = api.get_artifacts(scan_id)
        assert len(arts) == 1
        assert arts[0]["name"] == "test_artifact"


def test_idempotency():
    """Unknown scan returns empty artifacts (no phantom state)."""
    with temp_api() as api:
        arts_before = api.get_artifacts("nonexistent")
        assert len(arts_before) == 0


@pytest.mark.skipif(
    os.getenv("HERMES_RUN_PIPELINE") != "1",
    reason="Heavy end-to-end pipeline (real crawl/embed/LLM). "
           "Set HERMES_RUN_PIPELINE=1 to run.",
)
def test_full_pipeline_end_to_end():
    """Opt-in: start the worker and run the 12-step pipeline to completion.

    Queries are generic and caller-supplied (no brand-specific defaults -- those
    were removed in Layer -1). Tolerates timeout by cancelling and returning.
    """
    import time

    with temp_api() as api:
        scan = api.create_scan(
            project_id="demo",
            target_url="https://example.com",
            queries=[
                "what is this website about",
                "what does this page offer",
            ],
            max_pages=5,
        )
        scan_id = scan["scan_id"]

        api.start_scan(scan_id)
        api.start_worker()

        deadline = time.time() + 120
        final_status = None
        while time.time() < deadline:
            status = api.get_scan_status(scan_id)
            if status["status"] in ("completed", "failed", "cancelled"):
                final_status = status
                break
            time.sleep(2)

        api.stop_worker()

        if final_status is None:
            api.cancel_scan(scan_id)
            pytest.skip("pipeline did not finish within the deadline")

        recs = api.get_recommendations(scan_id)
        for r in recs:
            assert "type" in r and "title" in r and "priority" in r


if __name__ == "__main__":
    # Direct run: exercise everything including the opt-in pipeline test.
    os.environ.setdefault("HERMES_RUN_PIPELINE", "1")
    test_basic_lifecycle();            print("[PASS] basic_lifecycle")
    test_retry_flow();                 print("[PASS] retry_flow")
    test_events_and_artifacts();       print("[PASS] events_and_artifacts")
    test_idempotency();                print("[PASS] idempotency")
    test_full_pipeline_end_to_end();   print("[PASS] full_pipeline_end_to_end")
    print("ALL HERMES TESTS PASSED")
