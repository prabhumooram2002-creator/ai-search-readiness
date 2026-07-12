"""Layer 1 step 7 — topic clustering (src/topics.py).

Offline tests: embed_texts and llm_complete are stubbed; HDBSCAN runs for real
on synthetic, well-separated vectors. Covers cluster->topic mapping, the
generic-label reject/retry rule, the noise re-run rule (via a stubbed
_cluster), the too-few-chunks guard, and Trace emission. Live run on
docs.python.org is in BUILD_LOG.md.
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src import topics as tp
from src.trace import Trace


def _chunks(n, prefix="c"):
    return [{"content": f"text {i}", "chunk_id": f"{prefix}{i}"} for i in range(n)]


def _blob_embeddings():
    """Two tight, well-separated 4D blobs of 4 points each (8 chunks)."""
    a = [[1.0, 0.0, 0.0, 0.0] for _ in range(4)]
    b = [[0.0, 0.0, 0.0, 1.0] for _ in range(4)]
    # tiny jitter so HDBSCAN sees distinct points
    for i, v in enumerate(a):
        v[1] = 0.001 * i
    for i, v in enumerate(b):
        v[2] = 0.001 * i
    return a + b


def test_two_blobs_two_labeled_topics(monkeypatch):
    monkeypatch.setattr(tp.providers, "embed_texts", lambda texts: _blob_embeddings())
    calls = []

    def fake_llm(prompt, **kw):
        calls.append(prompt)
        return f'{{"label": "Specific Topic {len(calls)}"}}'
    monkeypatch.setattr(tp.providers, "llm_complete", fake_llm)

    trace = Trace(query="q")
    topics = tp.cluster_topics(_chunks(8), min_cluster_size=2, trace=trace)

    assert len(topics) == 2
    sizes = sorted(len(t["chunk_ids"]) for t in topics)
    assert sizes == [4, 4]
    # every chunk id accounted for exactly once across topics (no noise here)
    all_ids = [cid for t in topics for cid in t["chunk_ids"]]
    assert sorted(all_ids) == sorted(c["chunk_id"] for c in _chunks(8))
    assert all(t["label"].startswith("Specific Topic") for t in topics)
    st = trace.steps[0]
    assert st.scores["n_topics"] == 2.0
    assert st.scores["noise_ratio"] == 0.0


def test_generic_label_rejected_and_retried(monkeypatch):
    monkeypatch.setattr(tp.providers, "embed_texts", lambda texts: _blob_embeddings())
    answers = iter(['{"label": "General"}', '{"label": "Python Data Structures"}',
                    '{"label": "Misc"}', '{"label": "Misc"}'])
    monkeypatch.setattr(tp.providers, "llm_complete", lambda p, **kw: next(answers))

    trace = Trace(query="q")
    topics = tp.cluster_topics(_chunks(8), min_cluster_size=2, trace=trace)
    labels = sorted(t["label"] for t in topics)
    # cluster 1: "General" rejected -> retry gave a specific label
    assert "Python Data Structures" in labels
    # cluster 2: still generic after retry -> kept but flagged in notes
    assert "Misc" in labels
    assert any("STILL-GENERIC" in n for n in trace.steps[0].notes)
    assert any("rejected -> retry" in n for n in trace.steps[0].notes)


def test_noise_over_15pct_reruns_with_smaller_mcs(monkeypatch):
    monkeypatch.setattr(tp.providers, "embed_texts",
                        lambda texts: [[float(i)] * 4 for i in range(len(texts))])
    monkeypatch.setattr(tp.providers, "llm_complete",
                        lambda p, **kw: '{"label": "Whatever Topic"}')
    # attempt 1 (mcs=4): 50% noise; attempt 2 (mcs=3): all clustered
    batches = iter([
        [0, 0, 0, 0, -1, -1, -1, -1],
        [0, 0, 0, 0, 1, 1, 1, 1],
    ])
    seen_mcs = []

    def fake_cluster(embeddings, mcs):
        seen_mcs.append(mcs)
        return next(batches)
    monkeypatch.setattr(tp, "_cluster", fake_cluster)

    trace = Trace(query="q")
    topics = tp.cluster_topics(_chunks(8), min_cluster_size=4, trace=trace)
    assert seen_mcs == [4, 2]                      # re-ran with halved mcs
    assert len(topics) == 2
    assert trace.steps[0].scores["noise_ratio"] == 0.0


def test_too_few_chunks_skips_cleanly(monkeypatch):
    called = {"embed": False}
    monkeypatch.setattr(tp.providers, "embed_texts",
                        lambda texts: called.__setitem__("embed", True) or [])
    trace = Trace(query="q")
    topics = tp.cluster_topics(_chunks(2), trace=trace)
    assert topics == []
    assert called["embed"] is False               # no wasted embedding calls
    assert any("too few to cluster" in n for n in trace.steps[0].notes)


def test_trace_step_appended_even_on_error(monkeypatch):
    def boom(texts):
        raise RuntimeError("embedder down")
    monkeypatch.setattr(tp.providers, "embed_texts", boom)

    trace = Trace(query="q")
    try:
        tp.cluster_topics(_chunks(8), trace=trace)
        assert False, "should have raised"
    except RuntimeError:
        pass
    assert trace.step_names() == ["topic_clustering"]
    assert any("ERROR" in n for n in trace.steps[0].notes)
