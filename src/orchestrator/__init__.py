"""
Hermes — durable workflow orchestration for the AI Search Intelligence Platform.

Hermes coordinates multi-step analysis pipelines (crawl → chunk → extract →
embed → graph → repair → simulate → recommend), tracking progress durably,
retrying transient failures, and emitting events for dashboard/API consumers.

Quick start::

    from src.orchestrator import create_api

    api = create_api("data/hermes.db")
    scan = api.create_scan(
        project_id="wickedgud",
        target_url="https://wickedgud.com",
        queries=["healthy alternatives for fitness freaks"],
    )
    api.start_worker()  # background asyncio loop
    # ... let it run ...
    status = api.get_scan_status(scan["scan_id"])
    recs = api.get_recommendations(scan["scan_id"])
"""

from .api import HermesAPI
from .storage import HermesStore
from .engine import WorkflowEngine, StepDef, RetryPolicy
from .worker import HermesWorker
from .models import (
    WorkflowRun, WorkflowStatus, WorkflowStepRun, StepStatus,
    ErrorClass, ProgressEvent, Artifact, ModelUsageEvent,
)


def create_api(db_path: str = "data/hermes.db") -> HermesAPI:
    """Create a ready-to-use HermesAPI instance."""
    return HermesAPI(db_path)
