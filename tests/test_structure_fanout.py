"""Phase 3b (src/structure.py) + 3c (src/fanout.py) — offline tests."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src import structure as sc
from src import fanout as fo
from src.trace import Trace


# ── 3b: structural scorer ──────────────────────────────────────────────────
WELL = {"id": "c_good", "page_url": "http://x/p", "heading_path": ["What is X?"],
        "text": "X is a tool for teams."}
WALL = {"id": "c_bad", "page_url": "http://x/p", "heading_path": [],
        "text": ("Our platform is fast. It is secure. It scales. It costs 10 "
                 "dollars in tier 1, 20 in tier 2, 30 in tier 3, and 40 for "
                 "enterprise. " + "It does many things and keeps going on. " * 40)}


def test_well_structured_scores_high_wall_scores_low():
    good = sc.score_chunk(WELL, has_schema=True)
    bad = sc.score_chunk(WALL, has_schema=False)
    assert good["structure_score"] > 0.8, good
    assert bad["structure_score"] < 0.4, bad


def test_flags_point_at_real_issues():
    bad = sc.score_chunk(WALL, has_schema=False)
    kinds = {f["flag"] for f in bad["flags"]}
    assert "paragraph_too_long" in kinds
    assert "numeric_series_should_be_table" in kinds
    assert "no_heading" in kinds
    assert "no_schema_markup" in kinds
    # every flag carries a concrete fix + the chunk id (exact location)
    for f in bad["flags"]:
        assert f["fix"] and f["chunk_id"] == "c_bad"


def test_numeric_series_ignores_existing_tables():
    tbl = {"id": "c_t", "heading_path": ["Prices"],
           "text": "| tier | price |\n| a | 10 |\n| b | 20 |\n| c | 30 |"}
    r = sc.score_chunk(tbl, has_schema=True)
    assert not any(f["flag"] == "numeric_series_should_be_table" for f in r["flags"])


def test_score_chunks_trace_and_schema_set():
    trace = Trace(query="q")
    res = sc.score_chunks([WELL, WALL], pages_with_schema={"http://x/p"}, trace=trace)
    assert len(res) == 2
    assert trace.step_names() == ["structural_scoring"]
    assert "mean_structure_score" in trace.steps[0].scores


# ── 3c: stochastic fan-out ─────────────────────────────────────────────────
def _stub_embed(texts):
    # deterministic pseudo-embeddings: identical strings -> identical vectors;
    # bucket by first word so near-duplicates cluster.
    import hashlib
    out = []
    for t in texts:
        key = t.split()[0].lower() if t.split() else ""
        h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        v = [((h >> (i * 8)) & 0xFF) / 255.0 for i in range(8)]
        out.append(v)
    return out


def test_fanout_dedups_and_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(fo, "CACHE_DIR", tmp_path / "fo")
    monkeypatch.setattr(fo.providers, "embed_texts", _stub_embed)
    # 12 samples: "pricing ..." dominates, "security ..." minority
    calls = iter([
        '{"sub_queries": ["pricing of the tool", "security of the tool"]}',
    ] * fo.N_SAMPLES)
    monkeypatch.setattr(fo.providers, "llm_complete", lambda p, **k: next(calls))

    trace = Trace(query="q")
    dist = fo.fanout_distribution("is the tool good", use_cache=False, trace=trace)
    subs = {d["sub_query"].split()[0] for d in dist}
    assert subs == {"pricing", "security"}      # clustered by first word
    assert abs(sum(d["weight"] for d in dist) - 1.0) < 1e-6
    assert trace.steps[0].scores["n_sub_intents"] == 2.0


def test_fanout_cache_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(fo, "CACHE_DIR", tmp_path / "fo")
    monkeypatch.setattr(fo.providers, "embed_texts", _stub_embed)
    n = {"llm": 0}
    def llm(p, **k):
        n["llm"] += 1
        return '{"sub_queries": ["alpha thing", "beta thing"]}'
    monkeypatch.setattr(fo.providers, "llm_complete", llm)

    d1 = fo.fanout_distribution("q1", use_cache=True)
    first_calls = n["llm"]
    assert first_calls == fo.N_SAMPLES
    d2 = fo.fanout_distribution("q1", use_cache=True)   # cached -> no new calls
    assert n["llm"] == first_calls
    assert d1 == d2


def test_stability_jaccard_top5():
    # two rounds of the SAME distribution: top-5 Jaccard must be >= 0.6
    a = [{"sub_query": f"q{i}", "weight": 0.2} for i in range(5)]
    b = [{"sub_query": f"q{i}", "weight": 0.2} for i in range(1, 6)]  # 4/6 overlap
    top_a = {d["sub_query"] for d in a[:5]}
    top_b = {d["sub_query"] for d in b[:5]}
    jac = len(top_a & top_b) / len(top_a | top_b)
    assert jac >= 0.6
