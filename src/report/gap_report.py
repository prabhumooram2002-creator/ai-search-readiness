"""Gap report generator — Phase 8."""
import json, datetime
from pathlib import Path
from ..core.config import BASE_DIR
from ..core.logging import get_logger

logger = get_logger(__name__)

def generate_gap_report(gap_results=None) -> str:
    if gap_results is None:
        cache = BASE_DIR / "data" / "gap_batch_results.json"
        if cache.exists():
            with open(cache, encoding="utf-8") as f:
                raw = json.load(f)
            gap_results = [_dict_to_gap(g) for g in raw]

    if not gap_results:
        return str(BASE_DIR / "data" / "reports" / "nogaps.html")

    # Split into actionable (Tier 1/2) and disconnected (Tier 3)
    active = [g for g in gap_results if getattr(g, 'root_cause', '') != 'disconnected' and g.opportunity_score > 0]
    disconnected = [g for g in gap_results if getattr(g, 'root_cause', '') == 'disconnected' or g.opportunity_score <= 0]
    # If no connectivity_tier set (pre-v4 data), all go to active
    if not disconnected and all(getattr(g, 'connectivity_tier', 0) == 0 for g in gap_results):
        # Can't distinguish — fall back to old behavior (all active)
        has_tier_data = False
    else:
        has_tier_data = True
        # Re-split properly using connectivity_tier
        active = [g for g in gap_results if getattr(g, 'connectivity_tier', 3) <= 2]
        disconnected = [g for g in gap_results if getattr(g, 'connectivity_tier', 3) >= 3]

    actionable = active or gap_results  # fallback to all results if no tier data

    actionable.sort(key=lambda g: g.opportunity_score, reverse=True)
    disconnected.sort(key=lambda g: g.connectivity_score, reverse=True)

    total = len(gap_results)
    total_active = len(actionable)
    total_disconnected = len(disconnected)
    covered = sum(1 for g in actionable if g.coverage and g.coverage.verdict == "YES")
    partial = sum(1 for g in actionable if g.coverage and g.coverage.verdict == "PARTIALLY")
    gaps = total_active - covered - partial
    avg_score = sum(g.opportunity_score for g in actionable) / max(total_active, 1)
    high_priority = sum(1 for g in actionable if g.recommendation and g.recommendation.get("priority") == "high")

    # Phase 7/8 counts
    rc_counts = {}
    action_counts = {}
    weak_edge_count = 0
    weak_node_count = 0
    for g in gap_results:
        rc_counts[g.root_cause] = rc_counts.get(g.root_cause, 0) + 1
        if g.recommendation:
            a = g.recommendation.get("action", "unknown")
            action_counts[a] = action_counts.get(a, 0) + 1
        if g.weak_edge:
            weak_edge_count += 1
        if g.weak_node:
            weak_node_count += 1

    # ── Build table rows ────────────────────────────────────────────────
    row_parts = []
    dc_row_parts = []
    for i, g in enumerate(actionable[:200], 1):
        q_short = _esc(g.query.original[:55])
        score = g.opportunity_score
        rc = _esc(g.root_cause)
        verdict = g.coverage.verdict if g.coverage else "NO"
        verdict_class = verdict.lower()

        # Score color
        if score >= 0.65:
            score_color = "#4ade80"
        elif score >= 0.45:
            score_color = "#fbbf24"
        else:
            score_color = "#f87171"

        # Priority badge
        priority = (g.recommendation or {}).get("priority", "low")
        pri_class = priority
        pri_label = priority.upper()
        pri_color = {"high": "#f87171", "medium": "#fbbf24", "low": "#64748b"}.get(priority, "#64748b")

        # Repairs text (full, with line breaks preserved as spaces)
        rec = g.recommendation or {}
        repairs_text = rec.get("repairs", "")
        repairs_esc = _esc(repairs_text.replace(" | ", " ").replace("  ", " ")) if repairs_text else ""

        row_parts.append(
            "<tr>"
            "<td>" + str(i) + "</td>"
            "<td class=qc title=\"" + _esc(g.query.original) + "\">" + q_short + "</td>"
            "<td style=color:" + score_color + ";font-weight:700>" + ("%.3f" % score) + "</td>"
            "<td><span class=\"b b-" + verdict_class + "\">" + verdict + "</span></td>"
            "<td><span class=\"b\" style=background:" + pri_color + ";color:#0f1117>" + pri_label + "</span></td>"
            "<td><span class=brc>" + rc + "</span></td>"
            "<td class=rcell title=\"" + _esc(repairs_text) + "\">" + repairs_esc + "</td>"
            "</tr>"
        )
    rows = "\n".join(row_parts)

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    ts_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rc_labels = json.dumps(list(rc_counts.keys()))
    rc_values = json.dumps(list(rc_counts.values()))
    act_labels = json.dumps(list(action_counts.keys()))
    act_values = json.dumps(list(action_counts.values()))

    # Sort action_counts descending for the table
    top_actions = sorted(action_counts.items(), key=lambda x: -x[1])[:8]

    # Priority breakdown
    high_cnt = sum(1 for g in gap_results if (g.recommendation or {}).get("priority") == "high")
    med_cnt  = sum(1 for g in gap_results if (g.recommendation or {}).get("priority") == "medium")
    low_cnt  = len(gap_results) - high_cnt - med_cnt

    html = (
        "<!DOCTYPE html>\n<html lang=en>\n<head>\n"
        "<meta charset=UTF-8>\n<meta name=viewport content='width=device-width,initial-scale=1'>\n"
        "<title>Semantic Relationship Gap Report - WickedGud</title>\n"
        "<script src=https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js></script>\n"
        "<style>\n"
        "* {margin:0;padding:0;box-sizing:border-box}\n"
        "body {font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;"
        "background:#0f1117;color:#e2e8f0;min-height:100vh}\n"
        ".c {max-width:1600px;margin:0 auto;padding:2rem}\n"
        "h1 {color:#f8fafc;font-size:1.8rem;margin-bottom:0.25rem}\n"
        "h2 {color:#94a3b8;font-size:0.95rem;font-weight:400;margin-bottom:2rem}\n"
        "h3 {color:#e2e8f0;font-size:1rem;margin-bottom:1rem;"
        "border-bottom:1px solid #1e293b;padding-bottom:0.5rem}\n"
        ".card {background:#1a1d27;border:1px solid #2d3348;border-radius:12px;"
        "padding:1.5rem;margin-bottom:1.5rem;overflow:hidden}\n"
        ".g {display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-bottom:1.5rem}\n"
        ".s {text-align:center;padding:1rem;background:#1a1d27;border-radius:10px;border:1px solid #2d3348}\n"
        ".sv {font-size:2rem;font-weight:700}\n"
        ".sl {font-size:0.8rem;color:#94a3b8;margin-top:0.25rem}\n"
        "table {width:100%;border-collapse:collapse;font-size:0.78rem}\n"
        "th {text-align:left;color:#64748b;font-weight:500;padding:0.75rem 1rem;"
        "background:#111827;position:sticky;top:0;white-space:nowrap}\n"
        "td {padding:0.55rem 1rem;border-bottom:1px solid #1e293b;vertical-align:top}\n"
        "tr:hover {background:#1e293b}\n"
        ".qc {max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}\n"
        ".rcell {max-width:360px;color:#94a3b8;font-size:0.73rem;line-height:1.4}\n"
        ".b {display:inline-block;padding:0.2rem 0.5rem;border-radius:9999px;font-size:0.65rem;font-weight:700;letter-spacing:0.03em}\n"
        ".b-yes {background:#052e16;color:#4ade80}\n"
        ".b-partially {background:#1c1400;color:#fbbf24}\n"
        ".b-no {background:#1c0a0a;color:#f87171}\n"
        ".brc {background:#1e293b;color:#94a3b8;padding:0.15rem 0.4rem;border-radius:4px;font-size:0.68rem}\n"
        ".cw {display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-top:1rem}\n"
        ".ch {display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}\n"
        ".cc {position:relative;height:260px}\n"
        ".cw2 {display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-top:1rem}\n"
        ".tw {overflow-x:auto;max-height:65vh;overflow-y:auto}\n"
        ".ft {text-align:center;color:#475569;font-size:0.75rem;margin-top:2rem}\n"
        ".bn {background:linear-gradient(135deg,#0f1117 0%,#1a1d27 100%);"
        "border:1px solid #2d3348;border-radius:12px;padding:1.5rem 2rem;"
        "margin-bottom:2rem;display:flex;justify-content:space-between;align-items:center}\n"
        ".bn h1 {margin:0}\n"
        ".actbl {width:100%;border-collapse:collapse;font-size:0.8rem}\n"
        ".actbl td {padding:0.4rem 0.75rem;border-bottom:1px solid #1e293b}\n"
        ".actbl tr:hover {background:#1e293b}\n"
        ".pbar {display:flex;height:6px;background:#1e293b;border-radius:3px;overflow:hidden;margin-top:0.5rem}\n"
        ".pbar-h {background:#f87171;height:100%}\n"
        ".pbar-m {background:#fbbf24;height:100%}\n"
        ".pbar-l {background:#334155;height:100%}\n"
        ".ph {font-size:0.72rem;color:#94a3b8;margin-bottom:0.35rem}\n"
        "</style>\n</head>\n<body>\n<div class=c>\n"
        "<div class=bn>\n"
        "<div><h1>Semantic Relationship Gap Report</h1>\n"
        "<h2>WickedGud | https://wickedgud.com | " + str(total) + " queries | Generated " + ts + "</h2></div>\n"
        "</div>\n"
        "<div class=g>\n"
        "<div class=s><div class=sv style=color:#f87171>" + str(gaps) + "</div><div class=sl>Actionable Gaps (Tier 1/2)</div></div>\n"
        "<div class=s><div class=sv style=color:#fbbf24>" + str(partial) + "</div><div class=sl>Weak Matches</div></div>\n"
        "<div class=s><div class=sv style=color:#64748b>" + str(total_disconnected) + "</div><div class=sl>Disconnected (Tier 3)</div></div>\n"
        "<div class=s><div class=sv style=color:#4ade80>" + str(covered) + "</div><div class=sl>Full Coverage</div></div>\n"
        "<div class=s><div class=sv style=color:#38bdf8>" + ("%.3f" % avg_score) + "</div><div class=sl>Avg Opportunity</div></div>\n"
        "</div>\n"
        # Priority bar
        "<div class=card>\n"
        "<div class=ph>PRIORITY DISTRIBUTION — High / Medium / Low</div>\n"
        "<div class=pbar>\n"
        "<div class=pbar-h style=width:" + ("%.0f" % (high_cnt / max(total, 1) * 100)) + "%></div>\n"
        "<div class=pbar-m style=width:" + ("%.0f" % (med_cnt  / max(total, 1) * 100)) + "%></div>\n"
        "<div class=pbar-l></div>\n"
        "</div>\n"
        "<div style=display:flex;gap:1.5rem;margin-top:0.5rem;font-size:0.75rem>\n"
        "<span style=color:#f87171>&#9679; High: " + str(high_cnt) + "</span>\n"
        "<span style=color:#fbbf24>&#9679; Medium: " + str(med_cnt) + "</span>\n"
        "<span style=color:#64748b>&#9679; Low: " + str(low_cnt) + "</span>\n"
        "</div>\n"
        "</div>\n"
        "<div class=card>\n"
        "<h3>Gap Distribution by Root Cause + Action Type</h3>\n"
        "<div class=cw><div class=cc><canvas id=rc></canvas></div><div class=cc><canvas id=ac></canvas></div></div>\n"
        "</div>\n"
        "<div class=card>\n"
        "<h3>Ranked Gaps by Opportunity Score (Top 200)</h3>\n"
        "<div class=tw>\n"
        "<table>\n<thead>\n"
        "<tr><th>#</th><th>Query</th><th>Op<br>Score</th><th>Verdict</th>"
        "<th>Priority</th><th>Root Cause</th><th>Repairs Statement</th></tr>\n"
        "</thead>\n<tbody>\n" + rows + "\n</tbody>\n</table>\n"
        "</div>\n</div>\n"
        "<div class=card>\n"
        "<h3>Top Actions by Count</h3>\n"
        "<table class=actbl>\n"
        "<tr><th>Rank</th><th>Action Type</th><th>Count</th><th>%</th></tr>\n"
    )
    for rank_i, (act, cnt) in enumerate(top_actions, 1):
        pct = "%.0f" % (cnt / total * 100)
        html += (
            "<tr><td>" + str(rank_i) + "</td><td>" + _esc(act) + "</td><td>" + str(cnt) + "</td>"
            "<td>" + pct + "%</td></tr>\n"
        )
    html += (
        "</table>\n"
        "</div>\n"
    )

    # ── Table B: Disconnected / Out-of-Scope Concepts ───────────────────
    if disconnected:
        dc_rows_built = []
        for i, g in enumerate(disconnected[:100], 1):
            q_esc = _esc(g.query.original[:60])
            score = g.connectivity_score
            nearest = _esc(g.nearest_concept_name[:50]) if g.nearest_concept_name else "(none)"
            # Score color
            dc_color = "#64748b"
            dc_rows_built.append(
                "<tr>"
                "<td>" + str(i) + "</td>"
                "<td class=qc title=\"" + _esc(g.query.original) + "\">" + q_esc + "</td>"
                "<td style=color:" + dc_color + ";font-weight:700>" + ("%.4f" % score) + "</td>"
                "<td>" + nearest + "</td>"
                "<td style=color:#64748b;font-size:0.73rem>No semantic connection found — entirely different topic/category</td>"
                "</tr>"
            )
        dc_rows = "\n".join(dc_rows_built)
        html += (
            "<div class=card>\n"
            "<h3>Table B — Disconnected / Out-of-Scope Concepts (Tier 3) — " + str(len(disconnected)) + "</h3>\n"
            "<div class=tw>\n"
            "<table>\n<thead>\n"
            "<tr><th>#</th><th>Query</th><th>Connectivity<br>Score</th><th>Nearest Concept</th><th>Note</th></tr>\n"
            "</thead>\n<tbody>\n" + dc_rows + "\n</tbody>\n</table>\n"
            "</div>\n</div>\n"
        )

    html += (
        "<div class=ft>AI Search Readiness Platform | Semantic Relationship Gap Report | " + str(total) + " queries | Phase 7/8 enriched</div>\n"
        "</div>\n"
        "<script>\n"
        "new Chart(document.getElementById('rc'),{type:'doughnut',"
        "data:{labels:" + rc_labels + ",datasets:[{data:" + rc_values + ","
        "backgroundColor:['#f87171','#fbbf24','#38bdf8','#a78bfa','#34d399','#f472b6','#60a5fa','#facc15','#4ade80','#e879f9','#2dd4bf'],borderWidth:0}]},"
        "options:{responsive:true,maintainAspectRatio:false,"
        "plugins:{legend:{position:'right',labels:{color:'#94a3b8',font:{size:11}}}}}});\n"
        "new Chart(document.getElementById('ac'),{type:'bar',"
        "data:{labels:" + act_labels + ",datasets:[{label:'Count',data:" + act_values + ",backgroundColor:'#38bdf8',borderRadius:4}]},"
        "options:{responsive:true,maintainAspectRatio:false,indexAxis:'y',"
        "plugins:{legend:{display:false}},"
        "scales:{x:{ticks:{color:'#64748b'},grid:{color:'#1e293b'}},"
        "y:{ticks:{color:'#94a3b8',font:{size:10}},grid:{display:false}}}}});\n"
        "</script>\n</body>\n</html>"
    )

    out_dir = BASE_DIR / "data" / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / ("gap_report_" + ts_file + ".html")
    out_path.write_text(html, encoding="utf-8")
    logger.info("Gap report saved: " + str(out_path))
    return str(out_path)


def _esc(s):
    import html
    if not s:
        return ""
    return html.escape(s)


def _dict_to_gap(d):
    from ..input_layer.batch_pipeline import GapResult
    from ..input_layer import NormalizedQuery
    from ..retrieve.hybrid import PathResult
    from ..score.coverage import CoverageScore

    q = d.get("query", {})
    nq = NormalizedQuery(
        original=q.get("original", ""), canonical=q.get("canonical", ""),
        intent=q.get("intent", "informational"),
        detected_entities=q.get("detected_entities", []),
        source=q.get("source", ""), line_number=q.get("line_number", 0),
    )
    pr = d.get("path_result", {})
    path_result = PathResult(
        source=pr.get("source", ""), target=pr.get("target", ""),
        full_path=pr.get("full_path", []), hops=pr.get("hops", 0),
        break_point=pr.get("break_point"), missing_rel=pr.get("missing_rel"),
        suggested_fix=pr.get("suggested_fix"),
        entities_found=pr.get("entities_found", []),
    )
    cov = d.get("coverage")
    coverage = None
    if cov:
        coverage = CoverageScore(
            entity_coverage=cov.get("entity_coverage", 0),
            relationship_coverage=cov.get("relationship_coverage", 0),
            semantic_coverage=cov.get("semantic_coverage", 0),
            evidence_strength=cov.get("evidence_strength", 0),
            overall=cov.get("overall", 0), verdict=cov.get("verdict", "NO"),
        )
    return GapResult(
        query=nq,
        path_result=path_result,
        coverage=coverage,
        root_cause=d.get("root_cause", "unknown"),
        weak_edge=d.get("weak_edge"),
        weak_node=d.get("weak_node"),
        alternative_path=d.get("alternative_path", []),
        hop_count=d.get("hop_count", 0),
        opportunity_score=d.get("opportunity_score", 0.0),
        recommendation=d.get("recommendation"),
    )