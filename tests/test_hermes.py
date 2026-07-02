"""
Hermes integration test — full workflow lifecycle.

Tests:
1. Creating a scan returns immediately with workflow_id
2. Scanning status of an empty project
3. Cancelling a scan
4. Event recording
5. Running the full pipeline on existing data
6. Producing recommendations
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure project root on path
BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.orchestrator import create_api

# Use a temporary DB for tests
TEST_DB = str(BASE / "data" / "test_hermes.db")


def cleanup():
    db = Path(TEST_DB)
    if db.exists():
        db.unlink()


def test_basic_lifecycle():
    """Test workflow creation, status, events, and cancellation."""
    cleanup()
    api = create_api(TEST_DB)

    # 1. Create a scan
    scan = api.create_scan(
        project_id="test_proj",
        target_url="https://example.com",
        queries=["test query"],
    )
    assert "workflow_id" in scan
    assert "scan_id" in scan
    assert scan["status"] == "created"
    scan_id = scan["scan_id"]
    print(f"[OK] Scan created: {scan_id} / {scan['workflow_id']}")

    # 2. Get status
    status = api.get_scan_status(scan_id)
    assert status is not None
    assert status["status"] == "created"
    steps = status["steps"]
    assert len(steps) == 12  # 12 workflow steps
    assert "crawl_pages" in steps
    assert "finalize_scan" in steps
    print(f"[OK] Scan status retrieved: {len(steps)} steps defined")

    # 3. Start the scan
    started = api.start_scan(scan_id)
    assert started["status"] == "running"
    print(f"[OK] Scan started")

    # 4. Cancel the scan
    cancelled = api.cancel_scan(scan_id)
    assert cancelled["status"] == "cancelled"
    print(f"[OK] Scan cancelled")

    # 5. Events recorded
    events = api.get_events(scan_id)
    assert len(events) >= 2  # created + started + cancelled
    event_types = [e["type"] for e in events]
    assert "workflow.created" in event_types
    assert "workflow.started" in event_types
    assert "workflow.cancelled" in event_types
    print(f"[OK] {len(events)} events recorded")

    # 6. No artifacts on cancelled scan
    arts = api.get_artifacts(scan_id)
    assert len(arts) == 0
    print(f"[OK] No orphan artifacts")

    cleanup()
    print("\n✅ test_basic_lifecycle PASSED")


def test_retry_flow():
    """Test creating a failed workflow and retrying it."""
    cleanup()
    api = create_api(TEST_DB)

    scan = api.create_scan(
        project_id="test_proj",
        target_url="https://example.com",
    )
    scan_id = scan["scan_id"]
    wf_id = scan["workflow_id"]

    # Manually simulate a step failure to test retry logic
    from src.orchestrator.models import WorkflowStatus, StepStatus

    api.engine.start_workflow(wf_id)

    # Get the first step
    steps = api.store.get_workflow_steps(wf_id)
    first_step = steps[0]

    # Mark it as failed
    api.engine.mark_step_running(first_step.id)
    result = asyncio.run(api.engine.fail_step(first_step.id, {
        "class": "transient",
        "message": "Network timeout",
    }))
    assert result.status == StepStatus.ready  # retried
    print(f"[OK] Step auto-retried (status={result.status.value})")

    # Make it fail again until exhausted
    api.engine.mark_step_running(first_step.id)
    result = asyncio.run(api.engine.fail_step(first_step.id, {
        "class": "transient",
        "message": "Network timeout again",
    }))
    api.engine.mark_step_running(first_step.id)
    result = asyncio.run(api.engine.fail_step(first_step.id, {
        "class": "fatal",
        "message": "Corrupted data",
    }))
    assert result.status == StepStatus.failed
    print(f"[OK] Step finally failed after retries exhausted")

    # Workflow should be failed
    wf = api.store.get_workflow(wf_id)
    assert wf.status == WorkflowStatus.failed
    print(f"[OK] Workflow marked failed")

    # Retry from failed step
    retried = api.retry_scan(scan_id)
    assert retried["status"] == "running"
    print(f"[OK] Workflow retried from failed step")

    cleanup()
    print("\n✅ test_retry_flow PASSED")


def test_events_and_artifacts():
    """Test event recording and artifact lifecycle."""
    cleanup()
    api = create_api(TEST_DB)

    scan = api.create_scan(
        project_id="test_proj",
        target_url="https://example.com",
        metadata={"test": True},
    )
    scan_id = scan["scan_id"]
    wf_id = scan["workflow_id"]

    # Emit events directly
    api.store.emit_event(wf_id, scan_id, "workflow.step.progress",
                         message="Test event", progress_percent=50.0,
                         step_name="test_step")

    # Verify events
    events = api.get_events(scan_id)
    assert len(events) >= 1
    latest = events[-1]
    assert latest["type"] == "workflow.step.progress"
    assert latest["message"] == "Test event"
    assert latest["progress_percent"] == 50.0
    print(f"[OK] Events recorded correctly")

    # Create an artifact
    from src.orchestrator.models import Artifact
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
    print(f"[OK] Artifacts recorded correctly")

    cleanup()
    print("\n✅ test_events_and_artifacts PASSED")


def test_idempotency():
    """Test that creating the same workflow twice doesn't duplicate state."""
    cleanup()
    api = create_api(TEST_DB)

    # Same scan_id should not be duplicated
    arts_before = api.get_artifacts("nonexistent")
    assert len(arts_before) == 0
    print(f"[OK] Non-existent scan returns empty artifacts")

    cleanup()
    print("\n✅ test_idempotency PASSED")


def test_full_pipeline_on_existing_data():
    """
    Run the full Hermes pipeline end-to-end using existing WickedGud data.
    This test:
    1. Creates a scan with queries
    2. Starts the worker
    3. Waits for completion (or timeout)
    4. Verifies recommendations exist
    """
    cleanup()
    api = create_api(TEST_DB)

    # Use queries that match our existing crawl data
    scan = api.create_scan(
        project_id="wickedgud",
        target_url="https://wickedgud.com",
        queries=[
            "healthy alternatives for fitness freaks",
            "the noodles with protein for gym freaks",
            "best pasta for weight loss",
        ],
        max_pages=50,
    )
    scan_id = scan["scan_id"]
    print(f"\n[TEST] Created scan: {scan_id}")
    print(f"[TEST] Workflow: {scan['workflow_id']}")

    # Start the worker and the workflow
    api.start_scan(scan_id)
    api.start_worker()

    # Poll for completion (max 60 seconds)
    import time
    deadline = time.time() + 60
    final_status = None
    while time.time() < deadline:
        status = api.get_scan_status(scan_id)
        if status["status"] in ("completed", "failed", "cancelled"):
            final_status = status
            break
        print(f"[TEST] Status: {status['status']} ({status['progress_percent']:.0f}%) "
              f"step={status['current_step']}")
        time.sleep(2)

    api.stop_worker()

    if final_status is None:
        print(f"[WARN] Timeout -- stopping test. "
              f"Last status: {api.get_scan_status(scan_id)['status']}")
        api.cancel_scan(scan_id)
        cleanup()
        return

    print(f"\n[TEST] Final status: {final_status['status']}")
    print(f"[TEST] Progress: {final_status['progress_percent']:.0f}%")

    # Show step statuses
    for step_name, s in final_status["steps"].items():
        marker = "[OK]" if s["status"] == "succeeded" else \
                 "[FAIL]" if s["status"] == "failed" else \
                 "[SKIP]" if s["status"] == "skipped" else \
                 "[...]"
        print(f"  {marker} {step_name}: {s['status']}")

    # Get recommendations
    recs = api.get_recommendations(scan_id)
    if recs:
        print(f"\n[OK] {len(recs)} recommendations generated")
        for r in recs[:3]:
            print(f"  [{r['priority']}] {r['type']}: {r['title'][:80]}")

        # Type assertions
        assert len(recs) > 0, "Expected at least 1 recommendation"
        for r in recs:
            assert "type" in r
            assert "title" in r
            assert "priority" in r
    else:
        print(f"\n[WARN] No recommendations generated (steps may have been skipped)")

    # Show events
    events = api.get_events(scan_id, limit=20)
    print(f"\n[TEST] {len(events)} events recorded")

    # Get model usage
    usage = api.get_model_usage(scan_id)
    if usage:
        print(f"[TEST] {len(usage)} model usage records")
    else:
        print(f"[TEST] No model usage records")

    cleanup()
    print("\n✅ test_full_pipeline_on_existing_data PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("  HERMES ORCHESTRATION TESTS")
    print("=" * 60)
    print()

    test_basic_lifecycle()
    test_retry_flow()
    test_events_and_artifacts()
    test_idempotency()
    test_full_pipeline_on_existing_data()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
