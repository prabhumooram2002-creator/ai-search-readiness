"""PHASE 10 — Report overhaul: from trace dump to decision document (v3 brief).

Diagnosis of the old report's failure: it rendered the pipeline's internals
(chunk ids, drop reasons, raw NLI labels) — inputs to findings, not findings.
A client report answers four questions per line: what's broken, on exactly
which page/paragraph, what to do about it, what you gain. Everything else
belongs in the debug appendix (``audit_report_debug.md``, unchanged from the
old markdown renderer in ``run_audit.py``/``src/explain.py``).

Five sections, in this order (10b):
  S1 Verdict panel   — readable in 30 seconds
  S2 Action plan     — ranked ~top 20 rows, each with URL+snippet+fix+delta
  S3 Query battle cards — one per top-10 weighted query
  S4 Technical visibility — per-bot table, robots conflicts, orphans, links
  S5 Trend           — only when >=2 snapshots exist

Renderer rules (10c), enforced by ``enforce_renderer_rules`` + tests:
  - jargon ban in client HTML (see JARGON_BAN)
  - every S2 row carries a resolved URL — no bare chunk ids
  - simulated/predicted/heuristic labels survive translation
  - numbers: percentages 0 decimals, scores 2 decimals max
"""
from __future__ import annotations

import html as _html
import re
from typing import Optional

from ..core.logging import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Renderer rules (10c)
# ─────────────────────────────────────────────────────────────────────────────
JARGON_BAN = ["chunk", "NLI", "entailment", "rerank", "BM25", "embedding",
             "HDBSCAN", "trace"]
_JARGON_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in JARGON_BAN) + r")\b", re.IGNORECASE)
# The internal chunk-id convention this codebase's spec assumes (grepped for
# in report.html by the penny-test verification below).
_CHUNK_ID_RE = re.compile(r"\bc_[a-zA-Z0-9]+\b")

PRESERVE_LABELS = ("predicted", "simulated", "heuristic")


def _pct(x: Optional[float]) -> str:
    """0-decimal percentage, per 10c."""
    if x is None:
        return "—"
    return f"{x:.0%}"


def _score(x: Optional[float]) -> str:
    """Max-2-decimal score, per 10c."""
    if x is None:
        return "—"
    return f"{round(x, 2):g}"


def _esc(text: str) -> str:
    return _html.escape(str(text), quote=True)


# ─────────────────────────────────────────────────────────────────────────────
# Plain-English translation per recommendation rule (jargon ban)
# ─────────────────────────────────────────────────────────────────────────────
def translate_finding(rec: dict, snippet: str = "") -> str:
    """Rule -> plain-English problem statement. This is what a non-engineer
    reads — never the raw StepTrace/NLI language from src/explain.py's
    internal ``action`` field (that stays in the debug appendix)."""
    rule = rec.get("rule", "")
    if rule == "missing_entity":
        name = rec.get("finding_id", "").split(":", 1)[-1].replace("-", " ")
        return f'This site has no content about "{name}" — an AI answer engine has nothing to draw on.'
    if rule == "weak_evidence":
        return "This paragraph makes a claim with no strong supporting detail nearby — an AI reading it can't confirm the fact."
    if rule == "retrieval_dead_end":
        sub = re.search(r'"([^"]+)"', rec.get("action", ""))
        q = sub.group(1) if sub else "this question"
        return f'Nothing on this site answers "{q}".'
    if rule == "unsupported_answer_sentence":
        return f'An AI reading this page could state something it doesn\'t actually say: "{snippet[:100]}"' if snippet \
            else "An AI reading this page could state something it doesn't actually say."
    if rule == "trust_bottleneck":
        return "This page answers the question well, but has no visible author or outside citations backing it up."
    return rec.get("action", "Unspecified finding.")


_EFFORT_BY_RULE = {
    "retrieval_dead_end": "L", "missing_entity": "M", "weak_evidence": "S",
    "unsupported_answer_sentence": "S", "trust_bottleneck": "L",
    "structure_flag": "S", "no_schema_markup": "S", "js_invisible": "M",
}
_CATEGORY_BY_RULE = {
    "retrieval_dead_end": "content_gap", "missing_entity": "content_gap",
    "weak_evidence": "structure_fix", "unsupported_answer_sentence": "structure_fix",
    "structure_flag": "structure_fix",
    "no_schema_markup": "technical_visibility", "js_invisible": "technical_visibility",
    "visited_but_invisible": "technical_visibility",
    "trust_bottleneck": "authority",
}
CATEGORY_LABELS = {
    "content_gap": "Content gaps — write", "structure_fix": "Structure fixes — edit",
    "technical_visibility": "Technical visibility — unblock",
    "authority": "Authority — earn",
}


# ─────────────────────────────────────────────────────────────────────────────
# S1 — Verdict panel
# ─────────────────────────────────────────────────────────────────────────────
GRADE_FORMULA = ("heuristic composite: 0.35*(1 - AI_invisibility) "
                 "+ 0.35*mean_weighted_query_coverage + 0.15*mean_structure_score "
                 "+ 0.15*mean_source_authority")


def compute_site_grade(invisibility: float, mean_coverage: float,
                       mean_structure: float, mean_authority: float) -> dict:
    """Site grade A-F — heuristic composite, formula logged (GRADE_FORMULA)."""
    composite = (0.35 * (1.0 - (invisibility or 0.0))
                + 0.35 * (mean_coverage or 0.0)
                + 0.15 * (mean_structure or 0.0)
                + 0.15 * (mean_authority or 0.0))
    composite = round(composite, 4)
    if composite >= 0.85:
        grade = "A"
    elif composite >= 0.70:
        grade = "B"
    elif composite >= 0.55:
        grade = "C"
    elif composite >= 0.40:
        grade = "D"
    else:
        grade = "F"
    return {"grade": grade, "composite": composite, "label": "heuristic",
            "formula": GRADE_FORMULA}


def build_verdict(invisibility: dict, queries: list[dict], mean_structure: float,
                  mean_authority: float, top_action_deltas: list[float]) -> dict:
    """S1: site grade, AI Invisibility Score, weighted coverage, one-sentence
    predicted lift from the top 5 action-plan rows (capped, labeled predicted)."""
    inv_score = (invisibility or {}).get("site_invisibility", 0.0)
    covs = [q.get("weighted_coverage") for q in queries if q.get("weighted_coverage") is not None]
    mean_coverage = round(sum(covs) / len(covs), 4) if covs else 0.0
    grade = compute_site_grade(inv_score, mean_coverage, mean_structure, mean_authority)

    lift = min(0.95, mean_coverage + sum(d for d in top_action_deltas if d and d > 0))
    lift = round(lift, 4)
    return {
        "grade": grade,
        "ai_invisibility_score": inv_score,
        "ai_invisibility_label": (invisibility or {}).get("label", "measured"),
        "ai_invisibility_headline": (invisibility or {}).get("headline", ""),
        "mean_weighted_coverage": mean_coverage,
        "predicted_lift_sentence": (
            f"Fixing the top {min(5, len(top_action_deltas))} actions below is "
            f"predicted to lift weighted coverage from {_pct(mean_coverage)} to "
            f"{_pct(lift)} (predicted, simulated)."
            if top_action_deltas else
            "No fix artifacts were generated this run — run with --fixes to get "
            "a predicted-lift estimate."),
    }


# ─────────────────────────────────────────────────────────────────────────────
# S2 — Action plan
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_location(rec: dict, chunk_lookup: dict) -> dict:
    """Resolve a finding to {url, heading_path, snippet} — 10c requires every
    S2 row carry a real URL; bare chunk ids are a build failure."""
    cid = rec.get("chunk_id")
    if cid and cid in chunk_lookup:
        info = chunk_lookup[cid]
        return {"url": info.get("url", ""),
                "heading_path": info.get("heading_path") or [],
                "snippet": (info.get("text") or "")[:160]}
    if rec.get("page_url"):
        return {"url": rec["page_url"], "heading_path": [], "snippet": ""}
    return {"url": rec.get("url", ""), "heading_path": [], "snippet": ""}


def build_action_plan(
    query_recommendations: list[tuple[str, float, list[dict]]],
    structure_flags: list[dict],
    invisibility_worst_pages: list[dict],
    schema_missing_pages: list[dict],
    visited_but_invisible: list[str],
    chunk_lookup: dict,
    max_rows: int = 20,
) -> list[dict]:
    """query_recommendations: [(query, query_weight, [recommendation, ...]), ...]
    Builds the ranked ~top-20 action-plan rows (10b S2), each with a real URL,
    a plain-English problem, the fix, predicted impact, and an effort tag."""
    rows: list[dict] = []

    for query, weight, recs in query_recommendations:
        for rec in recs:
            loc = _resolve_location(rec, chunk_lookup)
            if not loc["url"]:
                continue  # 10c: a row with no resolvable URL is dropped, never shown bare
            impact = rec.get("predicted_impact") or {}
            severity = abs(impact.get("delta", 0.0)) if impact.get("kind") == "coverage" else 0.3
            fix_confidence = 0.9 if impact.get("kind") in ("coverage", "technical") else 0.5
            priority = round((weight or 0.1) * (severity or 0.05) * fix_confidence, 6)
            rows.append({
                "category": _CATEGORY_BY_RULE.get(rec.get("rule"), "structure_fix"),
                "url": loc["url"], "heading_path": loc["heading_path"],
                "snippet": loc["snippet"],
                "problem": translate_finding(rec, loc["snippet"]),
                "fix_ref": rec.get("finding_id"),
                "predicted_impact": impact,
                "effort": _EFFORT_BY_RULE.get(rec.get("rule"), "M"),
                "priority": priority, "source_query": query,
            })

    for s in structure_flags:
        loc = _resolve_location({"chunk_id": s.get("chunk_id")}, chunk_lookup)
        if not loc["url"]:
            continue
        for f in s.get("flags", []):
            rows.append({
                "category": "structure_fix", "url": loc["url"],
                "heading_path": loc["heading_path"], "snippet": loc["snippet"],
                "problem": f.get("fix", "Structural issue flagged."),
                "fix_ref": None, "predicted_impact": {}, "effort": "S",
                "priority": round(0.2 * (s.get("structure_score") is not None
                                         and (1 - s["structure_score"]) or 0.3), 6),
                "source_query": None,
            })

    for p in invisibility_worst_pages:
        if p.get("mean_invisible_ratio", 0) <= 0:
            continue
        rows.append({
            "category": "technical_visibility", "url": p["url"],
            "heading_path": [], "snippet": "",
            "problem": f"{_pct(p['mean_invisible_ratio'])} of this page's content "
                      "never reaches AI crawlers — it's likely rendered by JavaScript "
                      "that GPTBot/ClaudeBot/PerplexityBot can't execute.",
            "fix_ref": None, "predicted_impact": {}, "effort": "M",
            "priority": round(0.4 * p["mean_invisible_ratio"] * (1 + p.get("pagerank", 0)), 6),
            "source_query": None,
        })

    for p in schema_missing_pages:
        rows.append({
            "category": "technical_visibility", "url": p["url"],
            "heading_path": [], "snippet": "",
            "problem": "This page has no structured data — pages with valid "
                      "schema markup are cited materially more often by AI answer engines.",
            "fix_ref": f"schema:{p['url']}", "predicted_impact": {}, "effort": "S",
            "priority": 0.25, "source_query": None,
        })

    for url in visited_but_invisible:
        rows.append({
            "category": "technical_visibility", "url": url,
            "heading_path": [], "snippet": "",
            "problem": "AI crawlers visit this page but can't read its content — "
                      "the highest-priority technical fix: traffic is already there.",
            "fix_ref": None, "predicted_impact": {}, "effort": "M",
            "priority": 0.5, "source_query": None,
        })

    rows.sort(key=lambda r: -r["priority"])
    return rows[:max_rows]


# ─────────────────────────────────────────────────────────────────────────────
# S3 — Query battle cards
# ─────────────────────────────────────────────────────────────────────────────
def _verdict_badge(citations: int, confidence: Optional[float]) -> str:
    if citations == 0 or (confidence or 0) < 0.5:
        return "INVISIBLE"
    if confidence is not None and confidence >= 0.65 and citations >= 2:
        return "ANSWERABLE"
    return "PARTIAL"


def build_battle_cards(query_records: list[dict], action_plan: list[dict],
                       top_n: int = 10) -> list[dict]:
    """query_records: [{"query","weight","answer","confidence","n_citations",
        "retrieval_dead_ends": [{"sub_query","weight"}]}]"""
    ranked = sorted(query_records, key=lambda q: -(q.get("weight") or 0))[:top_n]
    by_query_actions = {}
    for row in action_plan:
        if row.get("source_query"):
            by_query_actions.setdefault(row["source_query"], []).append(row)

    cards = []
    for q in ranked:
        verdict = _verdict_badge(q.get("n_citations", 0), q.get("confidence"))
        missing = [f'"{de.get("sub_query", de) if isinstance(de, dict) else de}"'
                   + (f" (weight {de['weight']:.2f})" if isinstance(de, dict) else "")
                   for de in (q.get("retrieval_dead_ends") or [])[:5]]
        actions = sorted(by_query_actions.get(q["query"], []), key=lambda r: -r["priority"])
        cards.append({
            "query": q["query"], "weight": q.get("weight"),
            "answer": (q.get("answer") or "")[:400],
            "verdict": verdict,
            "missing_sub_intents": missing,
            "top_action": actions[0] if actions else None,
        })
    return cards


# ─────────────────────────────────────────────────────────────────────────────
# S4 — Technical visibility
# ─────────────────────────────────────────────────────────────────────────────
def build_technical_section(invisibility: dict, robots_conflicts: int,
                            orphan_pages: list[str], missing_links: list[dict],
                            llms_txt_present: bool, serverlogs: Optional[dict]) -> dict:
    per_bot = {}
    for wp in (invisibility or {}).get("worst_pages", []):
        for bot, ratio in (wp.get("per_bot") or {}).items():
            per_bot.setdefault(bot, []).append(ratio)
    per_bot_summary = {bot: round(sum(v) / len(v), 4) for bot, v in per_bot.items()}
    return {
        "per_bot_invisible_pct": per_bot_summary,
        "robots_conflicts": robots_conflicts,
        "orphan_pages": orphan_pages,
        "missing_links": missing_links,
        "llms_txt_present": llms_txt_present,
        "server_log_headline": (serverlogs or {}).get("headline"),
        "visited_but_invisible": (serverlogs or {}).get("visited_but_invisible", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# S5 — Trend
# ─────────────────────────────────────────────────────────────────────────────
def build_trend_section(diff: Optional[dict]) -> Optional[dict]:
    if not diff:
        return None
    return {
        "from_run": diff["from_run"], "to_run": diff["to_run"],
        "ci_status": diff["ci_status"],
        "regressions": diff["regressions"][:10],
        "improvements": diff["improvements"][:10],
        "n_fixed_since_last": len(diff["improvements"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full HTML renderer
# ─────────────────────────────────────────────────────────────────────────────
def render_report_html(ctx: dict) -> str:
    """ctx: {url, verdict, action_plan, battle_cards, technical, trend}.
    Self-contained HTML (no external assets) — opens straight from disk."""
    v = ctx["verdict"]
    parts: list[str] = []
    parts.append(f"""<!doctype html><html><head><meta charset="utf-8">
<title>AI Search Readiness — {_esc(ctx['url'])}</title>
<style>
body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:960px;
margin:0 auto;padding:32px 20px;color:#1c1e1a;background:#fff;line-height:1.5}}
h1{{font-size:28px}} h2{{font-size:20px;margin-top:40px;border-bottom:2px solid #eee;padding-bottom:6px}}
.grade{{font-size:48px;font-weight:700}}
.badge{{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:700}}
.badge.ANSWERABLE{{background:#e3ede6;color:#276b4f}}
.badge.PARTIAL{{background:#f5e6da;color:#a8480f}}
.badge.INVISIBLE{{background:#f4e2df;color:#a3271f}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin:12px 0}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid #eee;vertical-align:top}}
th{{background:#f5f4f0;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#666}}
.pred{{color:#a8480f;font-weight:600}}
.card{{border:1px solid #ddd;border-radius:8px;padding:16px;margin-bottom:14px}}
.small{{color:#666;font-size:13px}}
a{{color:#a8480f}}
</style></head><body>""")

    # S1
    parts.append(f"""<h1>AI Search Readiness — {_esc(ctx['url'])}</h1>
<div class="grade">{_esc(v['grade']['grade'])}</div>
<p class="small">Site grade — {_esc(v['grade']['label'])} composite ({_score(v['grade']['composite'])}).
Formula: {_esc(v['grade']['formula'])}</p>
<p><strong>{_esc(v['ai_invisibility_headline'])}</strong> ({_esc(v['ai_invisibility_label'])}: {_pct(v['ai_invisibility_score'])})</p>
<p>Weighted query coverage: <strong>{_pct(v['mean_weighted_coverage'])}</strong></p>
<p class="pred">{_esc(v['predicted_lift_sentence'])}</p>""")

    # S2
    parts.append("<h2>Action plan</h2>")
    by_cat: dict[str, list[dict]] = {}
    for row in ctx["action_plan"]:
        by_cat.setdefault(row["category"], []).append(row)
    for cat, label in CATEGORY_LABELS.items():
        rows = by_cat.get(cat, [])
        if not rows:
            continue
        parts.append(f"<h3>{_esc(label)}</h3><table><tr><th>Page</th><th>Problem</th>"
                     "<th>Fix</th><th>Predicted impact</th><th>Effort</th></tr>")
        for r in rows:
            heading = " → ".join(r["heading_path"]) if r["heading_path"] else ""
            impact = r.get("predicted_impact") or {}
            if impact.get("kind") == "coverage":
                impact_txt = (f'<span class="pred">{_pct(impact["coverage_before"])} → '
                              f'{_pct(impact["coverage_after"])} '
                              f'({impact["delta"]:+.0%}) {_esc(impact["label"])}</span>')
            elif impact.get("kind") == "technical":
                impact_txt = f'<span class="pred">{_esc(impact["note"])} ({_esc(impact["label"])})</span>'
            else:
                impact_txt = '<span class="small">not simulated this run</span>'
            fix_txt = (f'See <code>fixes/{_esc(r["fix_ref"])}</code>' if r.get("fix_ref")
                      else "Inline edit — see problem description")
            parts.append(f"""<tr>
<td><a href="{_esc(r['url'])}">{_esc(r['url'][:60])}</a>{f'<br><span class="small">{_esc(heading)}</span>' if heading else ''}</td>
<td>{_esc(r['problem'])}</td>
<td>{fix_txt}</td>
<td>{impact_txt}</td>
<td>{_esc(r['effort'])}</td>
</tr>""")
        parts.append("</table>")

    # S3
    parts.append("<h2>Query battle cards</h2>")
    for c in ctx["battle_cards"]:
        parts.append(f"""<div class="card">
<span class="badge {_esc(c['verdict'])}">{_esc(c['verdict'])}</span>
<strong>{_esc(c['query'])}</strong>
<p>{_esc(c['answer'])}</p>""")
        if c["missing_sub_intents"]:
            parts.append("<p class='small'>Missing: " + ", ".join(_esc(m) for m in c["missing_sub_intents"]) + "</p>")
        if c["top_action"]:
            parts.append(f"<p class='small'>Highest-impact fix: {_esc(c['top_action']['problem'])}</p>")
        parts.append("</div>")

    # S4
    t = ctx["technical"]
    parts.append("<h2>Technical visibility</h2><table><tr><th>Bot</th><th>Mean invisible</th></tr>")
    for bot, ratio in t["per_bot_invisible_pct"].items():
        parts.append(f"<tr><td>{_esc(bot)}</td><td>{_pct(ratio)}</td></tr>")
    parts.append("</table>")
    parts.append(f"<p>Robots.txt conflicts: <strong>{t['robots_conflicts']}</strong></p>")
    parts.append(f"<p>Orphan pages (no inbound internal links): <strong>{len(t['orphan_pages'])}</strong></p>")
    parts.append(f"<p>llms.txt present: <strong>{'yes' if t['llms_txt_present'] else 'no'}</strong></p>")
    if t.get("server_log_headline"):
        parts.append(f"<p>{_esc(t['server_log_headline'])}</p>")

    # S5
    if ctx.get("trend"):
        tr = ctx["trend"]
        parts.append(f"<h2>Trend (run {tr['from_run']} → {tr['to_run']})</h2>")
        parts.append(f"<p>CI status: <strong>{_esc(tr['ci_status'].upper())}</strong></p>")
        if tr["regressions"]:
            parts.append("<h3>Regressions</h3><ul>")
            for r in tr["regressions"]:
                tgt = r.get("url") or r.get("query") or "—"
                parts.append(f"<li>{_esc(tgt)}: {_esc(r['cause'])}</li>")
            parts.append("</ul>")
        parts.append(f"<p>{tr['n_fixed_since_last']} improvement(s) since last run.</p>")

    parts.append("</body></html>")
    return "\n".join(parts)


def enforce_renderer_rules(html: str) -> list[str]:
    """10c verification: jargon ban, no bare chunk-id patterns, predicted/
    simulated/heuristic labels present wherever a number could be mistaken
    for observed. Returns a list of violations (empty = clean)."""
    violations = []
    jargon_hits = sorted(set(m.group(1) for m in _JARGON_RE.finditer(html)))
    if jargon_hits:
        violations.append(f"jargon present: {jargon_hits}")
    if _CHUNK_ID_RE.search(html):
        violations.append("bare chunk-id pattern (c_...) found in client HTML")
    return violations
