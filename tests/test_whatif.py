"""Phase 7 — pre-publish what-if scoring (src/whatif.py)."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src import whatif as wf
from src.trace import Trace


def _embed(texts):
    # keyword one-hot: pricing / security / refund dims; else zero
    out = []
    for t in texts:
        tl = t.lower()
        out.append([1.0 if "pricing" in tl or "cost" in tl or "price" in tl else 0.0,
                    1.0 if "security" in tl or "compliance" in tl else 0.0,
                    1.0 if "refund" in tl or "return" in tl else 0.0])
    return out


def test_draft_raises_coverage_of_a_dead_end_query(monkeypatch, tmp_path):
    monkeypatch.setattr(wf.providers, "embed_texts", _embed)
    # fan-out has a refund sub-intent that the base corpus does NOT cover
    monkeypatch.setattr("src.fanout.fanout_distribution",
                        lambda q, use_cache=True, **k: [
                            {"sub_query": "product pricing and cost", "weight": 0.5},
                            {"sub_query": "refund and return policy", "weight": 0.5}])
    # semantic_chunks -> one chunk from the draft (stub to avoid MiniLM)
    monkeypatch.setattr("src.chunking2.semantic_chunks",
                        lambda text, url="", **k: [
                            {"id": "d0", "text": text, "page_url": url,
                             "heading_path": ["Refunds"], "char_start": 0,
                             "char_end": len(text), "token_estimate": 20}])

    base = [{"chunk_id": "c1", "content": "our pricing and cost plans", "url": "u1"}]
    base_emb = _embed([base[0]["content"]])

    draft = tmp_path / "draft.md"
    draft.write_text("# Refunds\n\nOur refund and return policy explained in full.\n",
                     encoding="utf-8")

    trace = Trace(query="q")
    res = wf.whatif(str(draft), ["tell me about the product"], base, base_emb,
                    trace=trace)
    pq = res["per_query"][0]
    assert pq["coverage_before"] == 0.5           # only pricing covered
    assert pq["coverage_after"] == 1.0            # draft adds refund coverage
    assert pq["coverage_delta"] == 0.5
    assert pq["newly_covered"] == 1
    assert trace.step_names() == ["whatif"]


def test_overlay_never_persists(monkeypatch, tmp_path):
    monkeypatch.setattr(wf.providers, "embed_texts", _embed)
    monkeypatch.setattr("src.fanout.fanout_distribution",
                        lambda q, use_cache=True, **k: [
                            {"sub_query": "pricing", "weight": 1.0}])
    monkeypatch.setattr("src.chunking2.semantic_chunks",
                        lambda text, url="", **k: [
                            {"id": "d0", "text": text, "page_url": url,
                             "heading_path": [], "char_start": 0,
                             "char_end": len(text), "token_estimate": 5}])

    # a fake persistent store (file + dir) that must stay byte-identical
    store_file = tmp_path / "kg.kuzu"
    store_file.write_bytes(b"persistent-graph-bytes")
    store_dir = tmp_path / "chromadb"
    store_dir.mkdir()
    (store_dir / "data.bin").write_bytes(b"persistent-vectors")

    draft = tmp_path / "d.md"
    draft.write_text("# X\n\nSome pricing content here for the draft.\n", encoding="utf-8")

    res = wf.whatif(str(draft), ["q"], [{"chunk_id": "c1", "content": "pricing", "url": "u"}],
                    _embed(["pricing"]),
                    persist_paths=[str(store_file), str(store_dir)])
    assert res["stores_unchanged"] is True
    assert res["store_hashes_before"] == res["store_hashes_after"]
    # and the files are literally untouched
    assert store_file.read_bytes() == b"persistent-graph-bytes"
    assert (store_dir / "data.bin").read_bytes() == b"persistent-vectors"


def test_draft_structure_flags_surface(monkeypatch, tmp_path):
    monkeypatch.setattr(wf.providers, "embed_texts", _embed)
    monkeypatch.setattr("src.fanout.fanout_distribution",
                        lambda q, use_cache=True, **k: [{"sub_query": "pricing", "weight": 1.0}])
    wall = "It is fast. It is cheap. Costs 10, 20, 30, 40, 50. " + "It does things. " * 40
    monkeypatch.setattr("src.chunking2.semantic_chunks",
                        lambda text, url="", **k: [
                            {"id": "d0", "text": wall, "page_url": url,
                             "heading_path": [], "char_start": 0, "char_end": 1,
                             "token_estimate": 300}])
    draft = tmp_path / "d.md"
    draft.write_text("wall of text", encoding="utf-8")
    res = wf.whatif(str(draft), ["q"], [{"chunk_id": "c", "content": "pricing", "url": "u"}],
                    _embed(["pricing"]))
    flags = {f["flag"] for f in res["structure_flags"]}
    assert "paragraph_too_long" in flags and "numeric_series_should_be_table" in flags
    assert res["mean_structure_score"] < 0.5
