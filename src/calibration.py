"""PHASE 6 — Manual calibration panel (v3 brief). $0, no API probes.

Simulated citation probability is untrusted until checked against reality. With
no API budget, the user runs the top-weighted queries in the free ChatGPT,
Perplexity, and Gemini web UIs and pastes each answer + its cited URLs into
``calibration/<run>/<query_id>.md``. ``calibrate`` parses those, extracts cited
domains, and computes — per engine and overall — whether the simulator's top-5
chunks' pages appear in the real citations (agreement rate). The result is
stored as ``citation_probability_observed`` and reported STRICTLY separately
from the simulated number (the v2 simulated-vs-observed rule stands).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from .core.config import BASE_DIR
from .core.logging import get_logger

logger = get_logger(__name__)

CALIB_DIR = BASE_DIR / "calibration"
ENGINES = ["ChatGPT", "Perplexity", "Gemini"]
AGREEMENT_FLOOR = 0.5          # < this after 2 panels -> tune retriever weights
_URL = re.compile(r"https?://[^\s)>\]]+")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "q"


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().lstrip("www.")


# ─────────────────────────────────────────────────────────────────────────────
# Template generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_templates(run_id: int, queries: list[dict],
                       calib_dir: Optional[Path] = None) -> Path:
    """Write one paste-template per query for the manual panel. ``queries``:
    [{"query", "weight"?}] (ideally the top-20 by weight)."""
    root = (calib_dir or CALIB_DIR) / f"run_{run_id:04d}"
    root.mkdir(parents=True, exist_ok=True)
    for i, q in enumerate(sorted(queries, key=lambda x: -x.get("weight", 0))[:20], 1):
        qid = f"q{i:03d}"
        body = [f"# Calibration — {q['query']}", f"query_id: {qid}",
                f"weight: {q.get('weight', 0)}", ""]
        for eng in ENGINES:
            body += [f"## {eng}", "<paste the answer text here>", "",
                     "### Cited URLs", "- ", ""]
        (root / f"{qid}.md").write_text("\n".join(body), encoding="utf-8")
    logger.info(f"[calibration] wrote {min(len(queries),20)} templates -> {root}")
    return root


# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_calibration_file(path: str | Path) -> dict:
    """Parse one filled template -> {query_id, query, weight,
    engines: {engine: {answer, cited_urls, cited_domains}}}."""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    query = re.search(r"^#\s+Calibration\s*[—-]\s*(.+)$", text, re.M)
    qid = re.search(r"^query_id:\s*(\S+)", text, re.M)
    weight = re.search(r"^weight:\s*([\d.]+)", text, re.M)

    engines: dict[str, dict] = {}
    # split on level-2 engine headings
    parts = re.split(r"^##\s+(" + "|".join(ENGINES) + r")\s*$", text, flags=re.M)
    # parts = [preamble, engine1, body1, engine2, body2, ...]
    for i in range(1, len(parts), 2):
        eng, body = parts[i], parts[i + 1]
        # cited URLs live under the '### Cited URLs' subsection
        cited_section = re.split(r"^###\s+Cited URLs\s*$", body, flags=re.M)
        answer = cited_section[0].strip()
        urls = _URL.findall(cited_section[1]) if len(cited_section) > 1 else _URL.findall(body)
        # ignore the empty placeholder
        if answer.startswith("<paste"):
            answer = ""
        urls = [u for u in urls if not u.endswith(">")]
        engines[eng] = {"answer": answer, "cited_urls": urls,
                        "cited_domains": sorted({_domain(u) for u in urls})}
    return {"query_id": qid.group(1) if qid else None,
            "query": query.group(1).strip() if query else "",
            "weight": float(weight.group(1)) if weight else 0.0,
            "engines": engines}


def load_run(run_id: int, calib_dir: Optional[Path] = None) -> list[dict]:
    root = (calib_dir or CALIB_DIR) / f"run_{run_id:04d}"
    if not root.exists():
        return []
    return [parse_calibration_file(p) for p in sorted(root.glob("q*.md"))]


# ─────────────────────────────────────────────────────────────────────────────
# Agreement
# ─────────────────────────────────────────────────────────────────────────────
def compute_agreement(calibration: list[dict],
                      winning_pages_by_query: dict[str, list[str]]) -> dict:
    """Did the simulator's winning pages appear in real citations?

    winning_pages_by_query: {query_text: [urls the simulator cited]}.
    Returns per-engine + overall agreement rates and per-query observed
    citation (1.0 if any winning-page domain was cited by that engine, else 0).
    """
    per_engine: dict[str, list[float]] = {e: [] for e in ENGINES}
    per_query = []
    for rec in calibration:
        wins = winning_pages_by_query.get(rec["query"], [])
        win_domains = {_domain(u) for u in wins if u}
        row = {"query": rec["query"], "query_id": rec["query_id"],
               "weight": rec["weight"], "engines": {}}
        for eng, data in rec["engines"].items():
            if not data["cited_urls"] and not data["answer"]:
                continue                       # engine not filled in -> skip
            hit = 1.0 if win_domains & set(data["cited_domains"]) else 0.0
            per_engine[eng].append(hit)
            row["engines"][eng] = {"observed_citation": hit,
                                   "cited_domains": data["cited_domains"],
                                   "win_domains": sorted(win_domains)}
        if row["engines"]:
            per_query.append(row)

    engine_rates = {e: round(sum(v) / len(v), 4) if v else None
                    for e, v in per_engine.items()}
    all_hits = [h for v in per_engine.values() for h in v]
    overall = round(sum(all_hits) / len(all_hits), 4) if all_hits else None
    return {
        "label": "observed",   # real panel data, never merged with simulated
        "per_engine_agreement": engine_rates,
        "overall_agreement": overall,
        "n_queries": len(per_query),
        "per_query": per_query,
        "below_floor": overall is not None and overall < AGREEMENT_FLOOR,
        "note": ("overall agreement < 50% -> tune retriever BM25/embed split & "
                 "rerank cutoff against this observed set (log every change)"
                 if (overall is not None and overall < AGREEMENT_FLOOR)
                 else "agreement acceptable"),
    }
