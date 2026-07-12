"""Phase 2 — incremental crawl (src/incremental.py). Offline: HTTP probe,
sitemap, and rendering are stubbed; state is real SQLite in tmp; KG/VS are
fakes that record surgical deletes. Live local-server proof is in BUILD_LOG."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import pytest
from src import incremental as inc
from src.crawl.crawler import CrawlResult
from src.trace import Trace


@pytest.fixture()
def state(tmp_path):
    s = inc.IncrementalState(str(tmp_path / "inc.db"))
    yield s
    s.close()


class FakeKG:
    def __init__(self):
        self.deleted = []
    def delete_chunk_data(self, cid):
        self.deleted.append(cid)


class FakeVS:
    def __init__(self):
        self.deleted = []
    def delete_ids(self, ids):
        self.deleted.extend(ids)


def _page(url, content, title="T"):
    return CrawlResult(url=url, title=title, content=content, html="", success=True)


def _seed_baseline(state, pages):
    run = state.next_run_id()
    from src.chunk.chunking import chunk_page
    for p in pages:
        state.upsert_page(p.url, etag='"e1"', last_modified="Mon, 01 Jan 2026",
                          markdown_hash=inc._h(p.content), run_id=run)
        for c in chunk_page(p):
            state.upsert_chunk(c.chunk_id, c.url, c.content, run)
    state.record_run(kind="full", pages_total=len(pages))


LONG_A = "Alpha page paragraph one with plenty of words to form a chunk of meaningful size. " * 3
LONG_B = "Bravo page has a different long paragraph that also forms a full chunk on its own. " * 3


def test_304_skips_everything(state, monkeypatch):
    pages = [_page("http://t/a", LONG_A), _page("http://t/b", LONG_B)]
    _seed_baseline(state, pages)
    monkeypatch.setattr(inc, "probe_conditional",
                        lambda url, e, lm: {"status": 304, "etag": e, "last_modified": lm})
    monkeypatch.setattr("src.layer0.fetch_sitemap", lambda u: [])
    rendered = []
    stats = inc.incremental_update(
        "http://t/", state, FakeKG(), FakeVS(),
        process_new_chunks=lambda ch: rendered.append(ch),
        render_pages=lambda urls: (_ for _ in ()).throw(AssertionError("rendered!"))
        if urls else [])
    assert stats["pages_304"] == 2
    assert stats["pages_changed"] == 0 and stats["chunks_added"] == 0
    assert rendered == []


def test_markdown_hash_catches_lying_headers(state, monkeypatch):
    pages = [_page("http://t/a", LONG_A)]
    _seed_baseline(state, pages)
    # probe says 200 (no 304 support), but content is identical
    monkeypatch.setattr(inc, "probe_conditional",
                        lambda url, e, lm: {"status": 200, "etag": None, "last_modified": None})
    monkeypatch.setattr("src.layer0.fetch_sitemap", lambda u: [])
    stats = inc.incremental_update(
        "http://t/", state, FakeKG(), FakeVS(),
        process_new_chunks=lambda ch: (_ for _ in ()).throw(AssertionError("processed!")),
        render_pages=lambda urls: [_page("http://t/a", LONG_A)])
    assert stats["pages_hash_unchanged"] == 1
    assert stats["pages_changed"] == 0


def test_chunk_diff_processes_only_changed_and_tombstones_deleted(state, monkeypatch):
    # Baseline: page had TWO chunks (seeded with the chunker's own id scheme —
    # the diff logic is under test here, not the chunker's split points).
    from src.chunk.chunking import _make_chunk_id
    url = "http://t/a"
    run = state.next_run_id()
    state.upsert_page(url, etag='"e1"', last_modified="Mon, 01 Jan 2026",
                      markdown_hash=inc._h("old page"), run_id=run)
    state.upsert_chunk(_make_chunk_id(url, 0), url, LONG_A, run)
    state.upsert_chunk(_make_chunk_id(url, 1), url, LONG_B, run)
    state.record_run(kind="full", pages_total=1)
    kg, vs = FakeKG(), FakeVS()
    monkeypatch.setattr(inc, "probe_conditional",
                        lambda url, e, lm: {"status": 200, "etag": '"e2"',
                                            "last_modified": "later"})
    monkeypatch.setattr("src.layer0.fetch_sitemap", lambda u: [])
    processed = []
    # page now has only the first paragraph, and it's EDITED
    edited = LONG_A.replace("plenty", "loads")
    trace = Trace(query="q")
    stats = inc.incremental_update(
        "http://t/", state, kg, vs,
        process_new_chunks=lambda ch: processed.extend(ch),
        render_pages=lambda urls: [_page("http://t/a", edited)], trace=trace)
    assert stats["pages_changed"] == 1
    assert stats["chunks_added"] == 1              # exactly the edited chunk
    assert stats["chunks_tombstoned"] == 1         # the removed 2nd paragraph
    assert len(processed) == 1 and "loads" in processed[0].content
    # surgical deletes hit both stores (changed chunk replaced + deleted chunk)
    assert len(kg.deleted) == 2 and set(kg.deleted) == set(vs.deleted)
    # trace records measured counts
    assert trace.steps[0].outputs["chunks_added"] == 1


def test_topics_rerun_ratio(state, monkeypatch):
    pages = [_page("http://t/a", LONG_A)]
    _seed_baseline(state, pages)
    monkeypatch.setattr(inc, "probe_conditional",
                        lambda url, e, lm: {"status": 200, "etag": None,
                                            "last_modified": None})
    monkeypatch.setattr("src.layer0.fetch_sitemap", lambda u: [])
    stats = inc.incremental_update(
        "http://t/", state, FakeKG(), FakeVS(),
        process_new_chunks=lambda ch: None,
        render_pages=lambda urls: [_page("http://t/a", LONG_B)])  # full rewrite
    assert stats["changed_ratio"] > inc.TOPIC_RERUN_RATIO
    assert stats["rerun_topics"] is True


def test_adaptive_revisit_interval(state):
    run = state.next_run_id()
    state.upsert_page("http://t/a", markdown_hash="h", run_id=run)
    state.upsert_page("http://t/a", changed=False, run_id=run)   # stable -> x2
    assert state.get_page("http://t/a")["revisit_interval"] == 2
    state.upsert_page("http://t/a", changed=False, run_id=run)
    assert state.get_page("http://t/a")["revisit_interval"] == 4
    state.upsert_page("http://t/a", changed=True, run_id=run)    # changed -> /2
    assert state.get_page("http://t/a")["revisit_interval"] == 2
    assert state.get_page("http://t/a")["change_count"] == 2     # insert + change


def test_crawl_runs_recorded(state, monkeypatch):
    pages = [_page("http://t/a", LONG_A)]
    _seed_baseline(state, pages)
    monkeypatch.setattr(inc, "probe_conditional",
                        lambda url, e, lm: {"status": 304, "etag": e, "last_modified": lm})
    monkeypatch.setattr("src.layer0.fetch_sitemap", lambda u: [])
    inc.incremental_update("http://t/", state, FakeKG(), FakeVS(),
                           process_new_chunks=lambda ch: None,
                           render_pages=lambda urls: [])
    rows = state.conn.execute("SELECT kind, pages_304 FROM crawl_runs").fetchall()
    assert [r["kind"] for r in rows] == ["full", "incremental"]
    assert rows[1]["pages_304"] == 1
