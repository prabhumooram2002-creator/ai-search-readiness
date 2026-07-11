"""PHASE 2 — Incremental crawl & data freshness (v3 brief).

Four-level skip chain, each level preventing the more expensive one below:

  1. Conditional GET  — stored ETag/Last-Modified -> If-None-Match /
                        If-Modified-Since; 304 skips the page entirely.
                        Sitemap ``lastmod`` prioritizes fetch order.
  2. Page hash        — hash of extracted MARKDOWN (never raw HTML — ads and
                        timestamps churn it); unchanged -> skip processing.
  3. Chunk-level diff — changed page -> re-chunk -> per-chunk content hash;
                        only changed/new chunks are reprocessed; deleted
                        chunks are tombstoned.
  4. Surgical update  — everything traces to chunk_id: delete that chunk's
                        graph edges/claims + Chroma vector, insert
                        replacements. NO full KG rebuild. Topic clustering
                        re-runs only when >15% of chunks changed.

State lives in SQLite (``data/incremental.db``): pages (validators, hashes,
adaptive revisit), chunks (hash + content + tombstones), crawl_runs (metrics).
"""
from __future__ import annotations

import hashlib
import json as _json
import sqlite3
import time
from pathlib import Path
from typing import Callable, Optional

from .core.config import BASE_DIR
from .core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_DB = str(BASE_DIR / "data" / "incremental.db")
TOPIC_RERUN_RATIO = 0.15          # >15% chunks changed -> re-cluster topics
MIN_REVISIT, MAX_REVISIT = 1, 16  # adaptive revisit interval (in runs)


def _h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:24]


class IncrementalState:
    """SQLite-backed crawl state. One instance per audited site DB."""

    def __init__(self, db_path: Optional[str] = None):
        path = db_path or DEFAULT_DB
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS pages(
                url TEXT PRIMARY KEY, etag TEXT, last_modified TEXT,
                markdown_hash TEXT, access_gaps_json TEXT DEFAULT '[]',
                internal_links_json TEXT DEFAULT '[]',
                title TEXT DEFAULT '', change_count INTEGER DEFAULT 0,
                revisit_interval INTEGER DEFAULT 1,
                runs_until_revisit INTEGER DEFAULT 0,
                last_seen_run INTEGER, last_changed_run INTEGER);
            CREATE TABLE IF NOT EXISTS chunks(
                chunk_id TEXT PRIMARY KEY, url TEXT, content_hash TEXT,
                content TEXT, status TEXT DEFAULT 'active',
                last_seen_run INTEGER);
            CREATE TABLE IF NOT EXISTS crawl_runs(
                run_id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT,
                pages_total INTEGER, pages_304 INTEGER,
                pages_hash_unchanged INTEGER, pages_changed INTEGER,
                chunks_added INTEGER, chunks_tombstoned INTEGER,
                duration_s REAL);
        """)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ── pages ───────────────────────────────────────────────────────────────
    def has_baseline(self) -> bool:
        return self.conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0] > 0

    def get_page(self, url: str):
        return self.conn.execute("SELECT * FROM pages WHERE url=?", (url,)).fetchone()

    def all_pages(self) -> list:
        return self.conn.execute("SELECT * FROM pages").fetchall()

    def upsert_page(self, url: str, *, etag=None, last_modified=None,
                    markdown_hash=None, access_gaps=None, internal_links=None,
                    title=None, changed: bool = False, run_id: int = 0):
        row = self.get_page(url)
        if row is None:
            self.conn.execute(
                "INSERT INTO pages(url, etag, last_modified, markdown_hash, "
                "access_gaps_json, internal_links_json, title, change_count, "
                "revisit_interval, runs_until_revisit, last_seen_run, "
                "last_changed_run) VALUES (?,?,?,?,?,?,?,1,1,0,?,?)",
                (url, etag, last_modified, markdown_hash,
                 _json.dumps(access_gaps or []), _json.dumps(internal_links or []),
                 title or "", run_id, run_id))
        else:
            # adaptive revisit: changed -> halve interval; stable -> double it
            interval = row["revisit_interval"]
            interval = max(MIN_REVISIT, interval // 2) if changed \
                else min(MAX_REVISIT, interval * 2)
            self.conn.execute(
                "UPDATE pages SET etag=COALESCE(?,etag), "
                "last_modified=COALESCE(?,last_modified), "
                "markdown_hash=COALESCE(?,markdown_hash), "
                "access_gaps_json=COALESCE(?,access_gaps_json), "
                "internal_links_json=COALESCE(?,internal_links_json), "
                "title=COALESCE(?,title), "
                "change_count=change_count+?, revisit_interval=?, "
                "runs_until_revisit=?, last_seen_run=?, "
                "last_changed_run=CASE WHEN ? THEN ? ELSE last_changed_run END "
                "WHERE url=?",
                (etag, last_modified, markdown_hash,
                 _json.dumps(access_gaps) if access_gaps is not None else None,
                 _json.dumps(internal_links) if internal_links is not None else None,
                 title, 1 if changed else 0, interval, interval, run_id,
                 changed, run_id, url))
        self.conn.commit()

    # ── chunks ──────────────────────────────────────────────────────────────
    def chunks_for_url(self, url: str) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT chunk_id, content_hash FROM chunks "
            "WHERE url=? AND status='active'", (url,)).fetchall()
        return {r["chunk_id"]: r["content_hash"] for r in rows}

    def active_chunks(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT chunk_id, url, content FROM chunks WHERE status='active'"
        ).fetchall()
        return [{"chunk_id": r["chunk_id"], "url": r["url"],
                 "content": r["content"]} for r in rows]

    def upsert_chunk(self, chunk_id: str, url: str, content: str, run_id: int):
        self.conn.execute(
            "INSERT INTO chunks(chunk_id, url, content_hash, content, status, "
            "last_seen_run) VALUES (?,?,?,?,'active',?) "
            "ON CONFLICT(chunk_id) DO UPDATE SET content_hash=excluded.content_hash, "
            "content=excluded.content, status='active', last_seen_run=excluded.last_seen_run",
            (chunk_id, url, _h(content), content, run_id))
        self.conn.commit()

    def tombstone_chunk(self, chunk_id: str, run_id: int):
        self.conn.execute("UPDATE chunks SET status='tombstone', last_seen_run=? "
                          "WHERE chunk_id=?", (run_id, chunk_id))
        self.conn.commit()

    # ── runs ────────────────────────────────────────────────────────────────
    def record_run(self, **kw) -> int:
        cur = self.conn.execute(
            "INSERT INTO crawl_runs(ts, kind, pages_total, pages_304, "
            "pages_hash_unchanged, pages_changed, chunks_added, "
            "chunks_tombstoned, duration_s) VALUES (?,?,?,?,?,?,?,?,?)",
            (time.strftime("%Y-%m-%dT%H:%M:%S"), kw.get("kind", "incremental"),
             kw.get("pages_total", 0), kw.get("pages_304", 0),
             kw.get("pages_hash_unchanged", 0), kw.get("pages_changed", 0),
             kw.get("chunks_added", 0), kw.get("chunks_tombstoned", 0),
             kw.get("duration_s", 0.0)))
        self.conn.commit()
        return cur.lastrowid

    def next_run_id(self) -> int:
        row = self.conn.execute("SELECT MAX(run_id) FROM crawl_runs").fetchone()
        return (row[0] or 0) + 1


# ─────────────────────────────────────────────────────────────────────────────
# Level 1 — conditional GET probe
# ─────────────────────────────────────────────────────────────────────────────
def probe_conditional(url: str, etag: Optional[str],
                      last_modified: Optional[str]) -> dict:
    """Cheap HTTP probe with validators. 304 -> page skipped entirely."""
    import httpx
    headers = {}
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified
    try:
        resp = httpx.get(url, headers=headers, timeout=20.0,
                         follow_redirects=True)
        return {"status": resp.status_code,
                "etag": resp.headers.get("ETag"),
                "last_modified": resp.headers.get("Last-Modified")}
    except Exception as e:
        logger.warning(f"[incremental] probe failed {url}: {e}")
        return {"status": -1, "etag": None, "last_modified": None}


# ─────────────────────────────────────────────────────────────────────────────
# Levels 2-4 — the incremental run
# ─────────────────────────────────────────────────────────────────────────────
def baseline_capture(state: IncrementalState, crawl_results: list,
                     chunk_objs: list, l0_pages: list[dict] | None = None) -> int:
    """After a FULL build: store validators, markdown hashes, and chunks so
    the next run can be incremental."""
    run_id = state.next_run_id()
    l0_by_url = {p["url"]: p for p in (l0_pages or [])}
    for r in crawl_results:
        if not getattr(r, "success", False):
            continue
        probe = probe_conditional(r.url, None, None)   # capture validators
        l0p = l0_by_url.get(r.url, {})
        state.upsert_page(
            r.url, etag=probe["etag"], last_modified=probe["last_modified"],
            markdown_hash=_h(r.content), title=r.title,
            access_gaps=l0p.get("access_gaps"),
            internal_links=l0p.get("internal_links"), run_id=run_id)
    for c in chunk_objs:
        state.upsert_chunk(c.chunk_id, c.url, c.content, run_id)
    state.record_run(kind="full", pages_total=len(crawl_results),
                     pages_changed=len(crawl_results),
                     chunks_added=len(chunk_objs))
    logger.info(f"[incremental] baseline captured: {len(chunk_objs)} chunks, "
                f"run_id={run_id}")
    return run_id


def incremental_update(
    site_url: str,
    state: IncrementalState,
    kg,
    vector_store,
    process_new_chunks: Callable[[list], None],
    render_pages: Optional[Callable[[list[str]], list]] = None,
    trace=None,
) -> dict:
    """One incremental run. Returns the crawl_runs stats dict.

    ``process_new_chunks(chunk_objs)`` is the caller's pipeline for
    changed/new chunks (embed -> Chroma -> GLiNER -> GLiREL -> claims -> NLI
    -> KG upserts). ``render_pages(urls)`` renders changed pages (defaults to
    the crawl4ai crawler, one page each).
    """
    from .layer0 import fetch_sitemap
    from .chunk.chunking import chunk_page

    t0 = time.time()
    run_id = state.next_run_id()
    step_cm = trace.start_step("incremental_update", site=site_url,
                               run_id=run_id) if trace else None
    st = step_cm.__enter__() if step_cm else None
    try:
        if render_pages is None:
            from .crawl import crawl_site_sync
            from .core import config as _cfg

            def render_pages(urls):
                out = []
                for u in urls:
                    _cfg.CFG.setdefault("crawl", {})["max_pages"] = 1
                    out.extend([r for r in crawl_site_sync(u) if r.success])
                return out

        # sitemap lastmod prioritizes fetch order; new sitemap URLs join in
        sitemap = fetch_sitemap(site_url)
        lastmod = {s["url"]: s["lastmod"] or "" for s in sitemap}
        known = {r["url"]: r for r in state.all_pages()}
        candidates = sorted(known, key=lambda u: lastmod.get(u, ""), reverse=True)
        new_urls = [u for u in lastmod if u not in known]

        pages_304 = pages_unchanged = pages_changed = 0
        chunks_added = chunks_tombstoned = 0
        to_render: list[str] = list(new_urls)

        # Level 1 — conditional GET on known pages
        for url in candidates:
            row = known[url]
            probe = probe_conditional(url, row["etag"], row["last_modified"])
            if probe["status"] == 304:
                pages_304 += 1
                state.upsert_page(url, changed=False, run_id=run_id)
                continue
            to_render.append(url)

        # Levels 2-3 — render survivors, hash markdown, diff chunks
        rendered = render_pages(to_render) if to_render else []
        chunks_to_process = []
        for r in rendered:
            row = state.get_page(r.url)
            new_hash = _h(r.content)
            if row is not None and row["markdown_hash"] == new_hash:
                pages_unchanged += 1          # headers lied; markdown identical
                state.upsert_page(r.url, changed=False, run_id=run_id)
                continue
            pages_changed += 1
            probe = probe_conditional(r.url, None, None)
            state.upsert_page(r.url, etag=probe["etag"],
                              last_modified=probe["last_modified"],
                              markdown_hash=new_hash, title=r.title,
                              changed=True, run_id=run_id)
            old = state.chunks_for_url(r.url)
            new_chunks = chunk_page(r)
            new_ids = set()
            for c in new_chunks:
                new_ids.add(c.chunk_id)
                if old.get(c.chunk_id) == _h(c.content):
                    continue                   # identical chunk — untouched
                if c.chunk_id in old:          # changed -> surgical replace
                    kg.delete_chunk_data(c.chunk_id)
                    vector_store.delete_ids([c.chunk_id])
                chunks_to_process.append(c)
                state.upsert_chunk(c.chunk_id, c.url, c.content, run_id)
            # Level 4 — tombstone deleted chunks + surgical graph delete
            for gone in set(old) - new_ids:
                state.tombstone_chunk(gone, run_id)
                kg.delete_chunk_data(gone)
                vector_store.delete_ids([gone])
                chunks_tombstoned += 1

        chunks_added = len(chunks_to_process)
        if chunks_to_process:
            process_new_chunks(chunks_to_process)

        total_active = len(state.active_chunks())
        changed_ratio = (chunks_added + chunks_tombstoned) / total_active \
            if total_active else 0.0
        rerun_topics = changed_ratio > TOPIC_RERUN_RATIO

        duration = time.time() - t0
        stats = {"run_id": run_id, "kind": "incremental",
                 "pages_total": len(candidates) + len(new_urls),
                 "pages_304": pages_304,
                 "pages_hash_unchanged": pages_unchanged,
                 "pages_changed": pages_changed,
                 "chunks_added": chunks_added,
                 "chunks_tombstoned": chunks_tombstoned,
                 "changed_ratio": round(changed_ratio, 4),
                 "rerun_topics": rerun_topics,
                 "duration_s": round(duration, 2)}
        state.record_run(**{k: v for k, v in stats.items()
                            if k not in ("run_id", "changed_ratio", "rerun_topics")})
        logger.info(f"[incremental] {stats}")
        if st is not None:
            st.outputs.update(stats)
            st.scores["pages_304"] = float(pages_304)          # measured
            st.scores["chunks_added"] = float(chunks_added)    # measured
        return stats
    except BaseException as exc:
        if step_cm is not None:
            step_cm.__exit__(type(exc), exc, exc.__traceback__)
            step_cm = None
        raise
    finally:
        if step_cm is not None:
            step_cm.__exit__(None, None, None)
