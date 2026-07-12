"""PHASE 5 — Snapshots & regression alerts (v3 brief).

After every run, persist a ``snapshot`` of the metrics that matter (per-page
invisibility, per-chunk structure_score, per-query cluster coverage +
confidence, site signals). ``diff`` reports ONLY regressions and improvements,
each traced to a cause, ranked by severity = magnitude x query-weight x page
PageRank. Non-zero exit on a regression above threshold -> cron/CI usable.
"""
from __future__ import annotations

import json as _json
import time
from pathlib import Path
from typing import Optional

from .core.config import BASE_DIR
from .core.logging import get_logger

logger = get_logger(__name__)

SNAP_DIR = BASE_DIR / "data" / "snapshots"
REGRESSION_THRESHOLD = 0.1     # severity above this -> non-zero exit


def build_snapshot(url: str, invisibility: dict, structure_scores: list[dict],
                   page_signals: dict, traces: list) -> dict:
    """Assemble a snapshot from a run's artifacts."""
    pages = {}
    pr = {wp["url"]: wp.get("pagerank", 0.0)
          for wp in (invisibility or {}).get("worst_pages", [])}
    for u, sig in (page_signals or {}).items():
        pages[u] = {
            "invisibility": next((wp["mean_invisible_ratio"]
                                  for wp in (invisibility or {}).get("worst_pages", [])
                                  if wp["url"] == u), 0.0),
            "pagerank": pr.get(u, 0.0),
            "signals": {k: sig.get(k) for k in
                        ("trust", "citation", "source_authority", "temporal_freshness")},
        }
    chunks = {s["chunk_id"]: s["structure_score"] for s in (structure_scores or [])}
    queries = {}
    for t in traces:
        retr = next((s for s in t.steps if s.name == "retriever"), None)
        cov = (retr.outputs.get("cluster_coverage", {}) if retr else {})
        queries[t.query] = {
            "confidence": t.confidence,
            "confidence_breakdown": t.confidence_breakdown,
            "weighted_coverage": cov.get("weighted_coverage"),
            "sub_intents_covered": cov.get("sub_intents_covered"),
            "citation_chunks": sorted({c["chunk_id"] for c in t.reranked}),
            "winning_pages": sorted({c["page_url"] for c in t.citations
                                     if c.get("page_url")}),
        }
    return {"run_id": None, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "url": url, "pages": pages, "chunks": chunks, "queries": queries}


def save_snapshot(snap: dict) -> int:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    run_id = _next_id()
    snap["run_id"] = run_id
    (SNAP_DIR / f"run_{run_id:04d}.json").write_text(
        _json.dumps(snap, indent=2, default=str), encoding="utf-8")
    logger.info(f"[snapshot] saved run_{run_id:04d}")
    return run_id


def _next_id() -> int:
    return len(list(SNAP_DIR.glob("run_*.json"))) + 1 if SNAP_DIR.exists() else 1


def list_snapshots() -> list[int]:
    if not SNAP_DIR.exists():
        return []
    return sorted(int(p.stem.split("_")[1]) for p in SNAP_DIR.glob("run_*.json"))


def load_snapshot(run_id: int) -> dict:
    return _json.loads((SNAP_DIR / f"run_{run_id:04d}.json").read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# diff
# ─────────────────────────────────────────────────────────────────────────────
def diff_snapshots(a: dict, b: dict) -> dict:
    """Regressions + improvements from snapshot ``a`` -> ``b``, each with a
    cause and severity. Severity = magnitude x query_weight x page_pagerank."""
    regressions, improvements = [], []

    # ── pages: invisibility up = regression; signal drops ──────────────────
    for url, pb in b["pages"].items():
        pa = a["pages"].get(url)
        if pa is None:
            continue
        pr = pb.get("pagerank") or 0.01
        d_inv = pb["invisibility"] - pa["invisibility"]
        if abs(d_inv) > 1e-6:
            item = {"kind": "page_invisibility", "url": url,
                    "from": pa["invisibility"], "to": pb["invisibility"],
                    "magnitude": round(abs(d_inv), 4),
                    "severity": round(abs(d_inv) * (pr / 0.01 if pr else 1), 4),
                    "cause": _invisibility_cause(d_inv)}
            (regressions if d_inv > 0 else improvements).append(item)

    # ── queries: coverage / confidence drops, traced to deleted chunks ─────
    deleted_chunks = set(a["chunks"]) - set(b["chunks"])
    for q, qb in b["queries"].items():
        qa = a["queries"].get(q)
        if qa is None:
            continue
        weight = _query_weight(b, q)
        for metric in ("weighted_coverage", "confidence"):
            va, vb = qa.get(metric), qb.get(metric)
            if va is None or vb is None:
                continue
            d = vb - va
            if abs(d) < 1e-6:
                continue
            cause = "unchanged retrieval"
            if d < 0:
                lost = set(qa.get("citation_chunks", [])) & deleted_chunks
                if lost:
                    cause = f"covering chunk(s) deleted: {sorted(lost)}"
                elif set(qa.get("winning_pages", [])) - set(qb.get("winning_pages", [])):
                    cause = "winning page(s) no longer cited"
            item = {"kind": f"query_{metric}", "query": q,
                    "from": va, "to": vb, "magnitude": round(abs(d), 4),
                    "weight": weight,
                    "severity": round(abs(d) * weight, 4), "cause": cause}
            (regressions if d < 0 else improvements).append(item)

    # ── chunks: structure_score drops ──────────────────────────────────────
    for cid, sb in b["chunks"].items():
        sa = a["chunks"].get(cid)
        if sa is None or abs(sb - sa) < 1e-6:
            continue
        d = sb - sa
        item = {"kind": "chunk_structure", "chunk_id": cid,
                "from": sa, "to": sb, "magnitude": round(abs(d), 4),
                "severity": round(abs(d), 4),
                "cause": "structure changed on re-crawl"}
        (regressions if d < 0 else improvements).append(item)

    regressions.sort(key=lambda x: -x["severity"])
    improvements.sort(key=lambda x: -x["severity"])
    worst = regressions[0]["severity"] if regressions else 0.0
    return {"from_run": a.get("run_id"), "to_run": b.get("run_id"),
            "regressions": regressions, "improvements": improvements,
            "worst_severity": worst,
            "ci_status": "fail" if worst > REGRESSION_THRESHOLD else "pass"}


def _invisibility_cause(delta: float) -> str:
    return ("content became invisible to AI bots (likely new JS-rendered template)"
            if delta > 0 else "content became visible to AI bots")


def _query_weight(snap: dict, q: str) -> float:
    return round(snap["queries"][q].get("weight", 1.0) or 1.0, 4)


def render_diff(d: dict) -> str:
    L = [f"# Regression report: run {d['from_run']} -> {d['to_run']}",
         f"CI status: **{d['ci_status'].upper()}** "
         f"(worst severity {d['worst_severity']}, threshold {REGRESSION_THRESHOLD})", ""]
    if d["regressions"]:
        L.append("## Regressions (most severe first)")
        for r in d["regressions"]:
            tgt = r.get("url") or r.get("query") or r.get("chunk_id")
            L.append(f"- [{r['severity']}] {r['kind']} `{tgt}`: "
                     f"{r['from']} -> {r['to']} — {r['cause']}")
    else:
        L.append("## No regressions")
    if d["improvements"]:
        L.append("\n## Improvements")
        for r in d["improvements"][:10]:
            tgt = r.get("url") or r.get("query") or r.get("chunk_id")
            L.append(f"- [{r['severity']}] {r['kind']} `{tgt}`: "
                     f"{r['from']} -> {r['to']} — {r['cause']}")
    return "\n".join(L)
