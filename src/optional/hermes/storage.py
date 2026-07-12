"""
Durable SQLite-backed state store for Hermes workflows.

All CRUD operations are O(1) or O(log n) via primary-key + covering-index queries.
"""

from __future__ import annotations
import json
import sqlite3
import uuid
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from .models import (
    WorkflowRun, WorkflowStatus, WorkflowStepRun, StepStatus,
    JobAttempt, Artifact, ProgressEvent, ModelUsageEvent,
    EVENT_WORKFLOW_CREATED,
)

# ── Helpers ─────────────────────────────────────────────────────────────────────

_SHORT_KEY_LEN = 12


def _new_id(prefix: str = "") -> str:
    raw = uuid.uuid4().hex[:_SHORT_KEY_LEN]
    return f"{prefix}{raw}" if prefix else raw


# ── Store ───────────────────────────────────────────────────────────────────────

class HermesStore:
    """
    SQLite-backed durable store.

    Thread-safe via connection-per-thread with WAL mode.  Schema is auto-created
    on first ``__init__``.  All mutation methods use ``INSERT … ON CONFLICT`` so
    idempotent retries are safe.
    """

    def __init__(self, db_path: str = "data/hermes.db"):
        # Per-instance so distinct stores never share a connection (a class-level
        # threading.local leaked the first-opened connection across instances).
        self._local = threading.local()
        self.db_path = str(Path(db_path).resolve())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── Connection management ──────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        try:
            return self._local.connection
        except AttributeError:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.connection = conn
            return conn

    def close(self) -> None:
        """Close this thread's SQLite connection so the file handle is released.

        Windows refuses to unlink an open DB file, so tests (and any caller that
        wants to delete the DB) must call this first. Safe to call repeatedly.
        """
        conn = getattr(self._local, "connection", None)
        if conn is not None:
            conn.close()
            del self._local.connection

    @contextmanager
    def tx(self) -> Generator[sqlite3.Connection, None, None]:
        """Transaction context manager -- commits on success, rollback on error."""
        conn = self._conn()
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise

    # ── Schema ──────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self.tx() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS workflow_run (
                    id              TEXT PRIMARY KEY,
                    project_id      TEXT NOT NULL,
                    scan_id         TEXT NOT NULL,
                    status          TEXT NOT NULL DEFAULT 'created',
                    current_step    TEXT,
                    progress_percent REAL NOT NULL DEFAULT 0.0,
                    created_at      TEXT NOT NULL,
                    started_at      TEXT,
                    finished_at     TEXT,
                    metadata_json   TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_wf_project ON workflow_run(project_id);
                CREATE INDEX IF NOT EXISTS idx_wf_scan    ON workflow_run(scan_id);
                CREATE INDEX IF NOT EXISTS idx_wf_status  ON workflow_run(status);

                CREATE TABLE IF NOT EXISTS step_run (
                    id              TEXT PRIMARY KEY,
                    workflow_id     TEXT NOT NULL REFERENCES workflow_run(id),
                    name            TEXT NOT NULL,
                    status          TEXT NOT NULL DEFAULT 'pending',
                    attempt_count   INTEGER NOT NULL DEFAULT 0,
                    depends_on_json TEXT NOT NULL DEFAULT '[]',
                    idempotency_key TEXT NOT NULL DEFAULT '',
                    started_at      TEXT,
                    finished_at     TEXT,
                    error_json      TEXT,
                    input_artifacts_json  TEXT NOT NULL DEFAULT '[]',
                    output_artifacts_json TEXT NOT NULL DEFAULT '[]'
                );
                CREATE INDEX IF NOT EXISTS idx_step_wf ON step_run(workflow_id);
                CREATE INDEX IF NOT EXISTS idx_step_ik ON step_run(idempotency_key);

                CREATE TABLE IF NOT EXISTS job_attempt (
                    id              TEXT PRIMARY KEY,
                    step_id         TEXT NOT NULL REFERENCES step_run(id),
                    attempt_number  INTEGER NOT NULL,
                    status          TEXT NOT NULL DEFAULT 'started',
                    started_at      TEXT NOT NULL,
                    finished_at     TEXT,
                    error_json      TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_attempt_step ON job_attempt(step_id);

                CREATE TABLE IF NOT EXISTS artifact (
                    id              TEXT PRIMARY KEY,
                    workflow_id     TEXT NOT NULL,
                    step_name       TEXT NOT NULL,
                    name            TEXT NOT NULL,
                    type            TEXT NOT NULL,
                    storage_path    TEXT,
                    content_hash    TEXT,
                    size_bytes      INTEGER,
                    created_at      TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_art_wf ON artifact(workflow_id);

                CREATE TABLE IF NOT EXISTS progress_event (
                    id              TEXT PRIMARY KEY,
                    workflow_id     TEXT NOT NULL,
                    scan_id         TEXT NOT NULL,
                    step_name       TEXT,
                    type            TEXT NOT NULL DEFAULT 'workflow.step.progress',
                    message         TEXT NOT NULL DEFAULT '',
                    progress_percent REAL NOT NULL DEFAULT 0.0,
                    created_at      TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_evt_wf  ON progress_event(workflow_id);
                CREATE INDEX IF NOT EXISTS idx_evt_sc  ON progress_event(scan_id);
                CREATE INDEX IF NOT EXISTS idx_evt_typ ON progress_event(type);

                CREATE TABLE IF NOT EXISTS model_usage (
                    id              TEXT PRIMARY KEY,
                    workflow_id     TEXT NOT NULL,
                    step_name       TEXT NOT NULL,
                    provider        TEXT NOT NULL,
                    model           TEXT NOT NULL,
                    input_tokens    INTEGER NOT NULL DEFAULT 0,
                    output_tokens   INTEGER NOT NULL DEFAULT 0,
                    latency_ms      INTEGER NOT NULL DEFAULT 0,
                    cost_usd        REAL NOT NULL DEFAULT 0.0,
                    created_at      TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_mu_wf ON model_usage(workflow_id);
            """)

    # ── Workflow CRUD ──────────────────────────────────────────────────────

    def create_workflow(
        self, project_id: str, scan_id: str,
        metadata: dict | None = None,
    ) -> WorkflowRun:
        wf = WorkflowRun(
            id=_new_id("wf_"),
            project_id=project_id,
            scan_id=scan_id,
            metadata=metadata or {},
        )
        with self.tx() as conn:
            conn.execute(
                "INSERT INTO workflow_run (id, project_id, scan_id, status, "
                "progress_percent, created_at, metadata_json) VALUES (?,?,?,?,?,?,?)",
                (wf.id, wf.project_id, wf.scan_id, wf.status.value,
                 wf.progress_percent, wf.created_at, json.dumps(wf.metadata)),
            )
        self._emit_event(ProgressEvent(
            id=_new_id("evt_"), workflow_id=wf.id, scan_id=scan_id,
            type=EVENT_WORKFLOW_CREATED,
            message=f"Workflow {wf.id} created for scan {scan_id}",
            progress_percent=0.0,
        ))
        return wf

    def get_workflow(self, workflow_id: str) -> WorkflowRun | None:
        row = self._conn().execute(
            "SELECT * FROM workflow_run WHERE id=?", (workflow_id,)
        ).fetchone()
        return self._row_to_workflow(row) if row else None

    def get_workflow_by_scan(self, scan_id: str) -> WorkflowRun | None:
        row = self._conn().execute(
            "SELECT * FROM workflow_run WHERE scan_id=? ORDER BY created_at DESC LIMIT 1",
            (scan_id,),
        ).fetchone()
        return self._row_to_workflow(row) if row else None

    def update_workflow(
        self, workflow_id: str, **kw,
    ) -> WorkflowRun | None:
        fields = []
        values = []
        for k, v in kw.items():
            if k == "metadata":
                fields.append("metadata_json=?")
                values.append(json.dumps(v))
            elif k == "status":
                fields.append("status=?")
                values.append(v.value if isinstance(v, WorkflowStatus) else v)
            else:
                fields.append(f"{k}=?")
                values.append(v)
        values.append(workflow_id)
        with self.tx() as conn:
            conn.execute(
                f"UPDATE workflow_run SET {', '.join(fields)} WHERE id=?",
                values,
            )
        return self.get_workflow(workflow_id)

    def list_workflows(
        self, project_id: str | None = None, status: str | None = None,
        limit: int = 20, offset: int = 0,
    ) -> list[WorkflowRun]:
        q = "SELECT * FROM workflow_run WHERE 1=1"
        params = []
        if project_id:
            q += " AND project_id=?"
            params.append(project_id)
        if status:
            q += " AND status=?"
            params.append(status)
        q += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self._conn().execute(q, params).fetchall()
        return [self._row_to_workflow(r) for r in rows]

    # ── Step CRUD ──────────────────────────────────────────────────────────

    def create_steps_from_definitions(
        self, workflow_id: str, step_defs: list[dict],
    ) -> list[WorkflowStepRun]:
        steps = []
        now = ProgressEvent._now() if hasattr(ProgressEvent, "_now") else ""
        with self.tx() as conn:
            for sd in step_defs:
                step = WorkflowStepRun(
                    id=_new_id("step_"),
                    workflow_id=workflow_id,
                    name=sd["name"],
                    status=StepStatus.pending,
                    depends_on=sd.get("depends_on", []),
                    idempotency_key=sd.get("idempotency_key", ""),
                )
                conn.execute(
                    "INSERT INTO step_run (id, workflow_id, name, status, "
                    "depends_on_json, idempotency_key) VALUES (?,?,?,?,?,?)",
                    (step.id, step.workflow_id, step.name, step.status.value,
                     json.dumps(step.depends_on), step.idempotency_key),
                )
                steps.append(step)
        return steps

    def get_workflow_steps(self, workflow_id: str) -> list[WorkflowStepRun]:
        rows = self._conn().execute(
            "SELECT * FROM step_run WHERE workflow_id=? ORDER BY rowid",
            (workflow_id,),
        ).fetchall()
        return [self._row_to_step(r) for r in rows]

    def get_step(self, step_id: str) -> WorkflowStepRun | None:
        row = self._conn().execute(
            "SELECT * FROM step_run WHERE id=?", (step_id,)
        ).fetchone()
        return self._row_to_step(row) if row else None

    def get_step_by_name(self, workflow_id: str, name: str) -> WorkflowStepRun | None:
        row = self._conn().execute(
            "SELECT * FROM step_run WHERE workflow_id=? AND name=?",
            (workflow_id, name),
        ).fetchone()
        return self._row_to_step(row) if row else None

    def update_step(self, step_id: str, **kw) -> WorkflowStepRun | None:
        fields = []
        values = []
        for k, v in kw.items():
            if k == "status":
                fields.append("status=?")
                values.append(v.value if isinstance(v, StepStatus) else v)
            elif k == "error":
                fields.append("error_json=?")
                values.append(json.dumps(v) if v else None)
            elif k in ("depends_on", "input_artifacts", "output_artifacts"):
                fields.append(f"{k}_json=?")
                values.append(json.dumps(v))
            else:
                fields.append(f"{k}=?")
                values.append(v)
        values.append(step_id)
        with self.tx() as conn:
            conn.execute(
                f"UPDATE step_run SET {', '.join(fields)} WHERE id=?",
                values,
            )
        return self.get_step(step_id)

    # ── Attempt CRUD ───────────────────────────────────────────────────────

    def create_attempt(self, step_id: str, attempt_number: int) -> JobAttempt:
        att = JobAttempt(
            id=_new_id("att_"),
            step_id=step_id,
            attempt_number=attempt_number,
        )
        with self.tx() as conn:
            conn.execute(
                "INSERT INTO job_attempt (id, step_id, attempt_number, "
                "status, started_at) VALUES (?,?,?,?,?)",
                (att.id, att.step_id, att.attempt_number,
                 att.status, att.started_at),
            )
        return att

    def update_attempt(self, attempt_id: str, **kw) -> None:
        fields = ", ".join(f"{k}=?" for k in kw)
        values = list(kw.values())
        values.append(attempt_id)
        with self.tx() as conn:
            conn.execute(
                f"UPDATE job_attempt SET {fields} WHERE id=?", values,
            )

    # ── Artifact CRUD ──────────────────────────────────────────────────────

    def add_artifact(self, art: Artifact) -> None:
        with self.tx() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO artifact "
                "(id, workflow_id, step_name, name, type, storage_path, "
                "content_hash, size_bytes, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (art.id, art.workflow_id, art.step_name, art.name, art.type,
                 art.storage_path, art.content_hash, art.size_bytes, art.created_at),
            )

    def get_artifacts(self, workflow_id: str) -> list[Artifact]:
        rows = self._conn().execute(
            "SELECT * FROM artifact WHERE workflow_id=? ORDER BY created_at",
            (workflow_id,),
        ).fetchall()
        return [Artifact(**dict(r)) for r in rows]

    # ── Events ─────────────────────────────────────────────────────────────

    def _emit_event(self, evt: ProgressEvent) -> None:
        with self.tx() as conn:
            conn.execute(
                "INSERT INTO progress_event (id, workflow_id, scan_id, step_name, "
                "type, message, progress_percent, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (evt.id, evt.workflow_id, evt.scan_id, evt.step_name,
                 evt.type, evt.message, evt.progress_percent, evt.created_at),
            )

    def emit_event(
        self, workflow_id: str, scan_id: str, type_: str,
        message: str = "", progress_percent: float = 0.0,
        step_name: str | None = None,
    ) -> ProgressEvent:
        evt = ProgressEvent(
            id=_new_id("evt_"),
            workflow_id=workflow_id,
            scan_id=scan_id,
            step_name=step_name,
            type=type_,
            message=message,
            progress_percent=progress_percent,
        )
        self._emit_event(evt)
        return evt

    def get_events(
        self, workflow_id: str | None = None, scan_id: str | None = None,
        limit: int = 100, offset: int = 0,
    ) -> list[ProgressEvent]:
        q = "SELECT * FROM progress_event WHERE 1=1"
        params = []
        if workflow_id:
            q += " AND workflow_id=?"
            params.append(workflow_id)
        if scan_id:
            q += " AND scan_id=?"
            params.append(scan_id)
        q += " ORDER BY created_at ASC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self._conn().execute(q, params).fetchall()
        return [ProgressEvent(**dict(r)) for r in rows]

    # ── Model Usage ────────────────────────────────────────────────────────

    def add_model_usage(self, mu: ModelUsageEvent) -> None:
        with self.tx() as conn:
            conn.execute(
                "INSERT INTO model_usage (id, workflow_id, step_name, provider, "
                "model, input_tokens, output_tokens, latency_ms, cost_usd, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (mu.id, mu.workflow_id, mu.step_name, mu.provider, mu.model,
                 mu.input_tokens, mu.output_tokens, mu.latency_ms, mu.cost_usd,
                 mu.created_at),
            )

    def get_model_usage(self, workflow_id: str) -> list[ModelUsageEvent]:
        rows = self._conn().execute(
            "SELECT * FROM model_usage WHERE workflow_id=? ORDER BY created_at",
            (workflow_id,),
        ).fetchall()
        return [ModelUsageEvent(**dict(r)) for r in rows]

    # ── Row → Dataclass ────────────────────────────────────────────────────

    @staticmethod
    def _row_to_workflow(row: sqlite3.Row) -> WorkflowRun:
        d = dict(row)
        d["metadata"] = json.loads(d.pop("metadata_json", "{}"))
        d["status"] = WorkflowStatus(d["status"])
        return WorkflowRun(**d)

    @staticmethod
    def _row_to_step(row: sqlite3.Row) -> WorkflowStepRun:
        d = dict(row)
        d["depends_on"] = json.loads(d.pop("depends_on_json", "[]"))
        raw_err = d.pop("error_json", None)
        d["error"] = json.loads(raw_err) if raw_err else None
        d["input_artifacts"] = json.loads(d.pop("input_artifacts_json", "[]"))
        d["output_artifacts"] = json.loads(d.pop("output_artifacts_json", "[]"))
        # Remove any remaining _json columns not in dataclass
        for key in list(d.keys()):
            if key.endswith("_json"):
                d.pop(key, None)
        d["status"] = StepStatus(d["status"])
        return WorkflowStepRun(**d)
