"""Renders intelligence.json into a single self-contained intelligence.html
— sidebar nav (19 sections), client-side search (plain JS substring filter
over row text, no external library), works offline from disk."""
from __future__ import annotations

import json as _json
import re as _re

from .core_utils import esc, pct, score2

_ID_COL_RE = _re.compile(r"(^id$|_id$)", _re.IGNORECASE)

SECTION_TITLES = [
    ("section_1_identity", "1. Website Identity"),
    ("section_2_entities", "2. Complete Entity Graph"),
    ("section_3_relationships", "3. Relationship Graph"),
    ("section_4_topics", "4. Semantic Topics"),
    ("section_5_chunks", "5. Semantic Chunks"),
    ("section_6_keywords", "6. Keyword Intelligence"),
    ("section_7_backlinks", "7. Backlink Intelligence"),
    ("section_8_crawl", "8. Crawl Intelligence"),
    ("section_9_crawlability", "9. AI Crawlability"),
    ("section_10_kg_quality", "10. Knowledge Graph Quality"),
    ("section_11_12_claims", "11/12. Claim & Evidence Intelligence"),
    ("section_13_query_simulation", "13. Query Simulation"),
    ("section_14_reasoning_paths", "14. Reasoning Path"),
    ("section_15_competitor", "15. Competitor Knowledge Graph"),
    ("section_16_content", "16. Content Intelligence"),
    ("section_17_trust", "17. Trust Intelligence"),
    ("section_18_citation_readiness", "18. Citation Readiness"),
    ("section_19_recommendations", "19. Recommendation Engine"),
]

_STYLE = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;color:#1c1e1a;background:#fff}
.layout{display:flex;min-height:100vh}
nav{width:260px;flex-shrink:0;background:#f5f4f0;border-right:1px solid #ddd;padding:16px;
   position:sticky;top:0;height:100vh;overflow-y:auto}
nav a{display:block;padding:6px 8px;color:#1c1e1a;text-decoration:none;font-size:13px;border-radius:6px}
nav a:hover{background:#eee}
main{flex:1;padding:24px 32px;max-width:1000px}
h2{border-bottom:2px solid #eee;padding-bottom:6px;margin-top:40px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #eee;vertical-align:top}
th{background:#f5f4f0;font-size:11px;text-transform:uppercase;color:#666}
.searchbox{width:100%;box-sizing:border-box;padding:10px 14px;font-size:14px;
   border:1px solid #ddd;border-radius:8px;margin-bottom:16px;position:sticky;top:0;background:#fff}
.heur{color:#a8480f} .note{color:#666;font-size:12px}
"""

_SEARCH_JS = """
const box = document.getElementById('searchbox');
box.addEventListener('input', () => {
  const q = box.value.trim().toLowerCase();
  document.querySelectorAll('main tr').forEach(tr => {
    tr.style.display = !q || tr.textContent.toLowerCase().includes(q) ? '' : 'none';
  });
});
"""


def _kv_table(rows: list[tuple[str, str]]) -> str:
    return "<table>" + "".join(
        f"<tr><td>{esc(k)}</td><td>{v}</td></tr>" for k, v in rows) + "</table>"


def _resolve_source(item: dict, id_cols: list[str], ref_index: dict) -> str:
    """Turns a row's id field(s) into a link to the page URL + a text
    snippet, per the brief's hard rule that a bare chunk/entity id is a
    build failure. Falls back to an explicit, honest note (never a raw id)
    when no page can be resolved -- e.g. a fully orphaned entity."""
    url = item.get("url") or item.get("page_url")
    snippet = item.get("snippet")
    # chunk-level ids resolve to the actual source text; prefer them over
    # entity ids, which only resolve to *a* page the entity happens to
    # appear on.
    ordered_cols = sorted(id_cols, key=lambda c: 0 if "chunk" in c.lower() else 1)
    for col in ordered_cols:
        raw = item.get(col)
        if raw is None:
            continue
        ref = ref_index.get(raw)
        if ref:
            url = url or ref.get("url")
            snippet = snippet or ref.get("snippet")
    if url:
        label = esc(snippet) if snippet else "view source"
        return f'<a href="{esc(url)}" target="_blank" rel="noopener">{label}</a>'
    return "<span class='note'>heuristic: no linked page (orphan reference)</span>"


def _sanitize_nested(value, ref_index: dict):
    """A cell value that is itself a list/dict (e.g. section 13's per-query
    'retrieved'/'reranked' arrays) still gets naively stringified for
    display -- walk it and swap any id-like key's raw value for its
    resolved page URL, so a nested chunk_id can't leak into the report as
    a bare id either."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if _ID_COL_RE.search(k):
                ref = ref_index.get(v)
                out[k] = (ref.get("url") if ref else None) or "unresolved"
            else:
                out[k] = _sanitize_nested(v, ref_index)
        return out
    if isinstance(value, list):
        return [_sanitize_nested(v, ref_index) for v in value]
    return value


def _list_table(items: list[dict], columns: list[str], ref_index: dict | None = None) -> str:
    ref_index = ref_index or {}
    if not items:
        return "<p class='note'>No rows.</p>"
    id_cols = [c for c in columns if _ID_COL_RE.search(c)]
    display_cols = [c for c in columns if c not in id_cols][:6]
    head = "".join(f"<th>{esc(c)}</th>" for c in display_cols)
    if id_cols:
        head += "<th>source</th>"
    body = ""
    for item in items:
        cells = ""
        for c in display_cols:
            v = item.get(c, "")
            if isinstance(v, (list, dict)):
                v = _sanitize_nested(v, ref_index)
            cells += f"<td>{esc(v)[:200]}</td>"
        if id_cols:
            cells += f"<td>{_resolve_source(item, id_cols, ref_index)}</td>"
        body += f"<tr>{cells}</tr>"
    return f"<table><tr>{head}</tr>{body}</table>"


def render_section(key: str, data, ref_index: dict | None = None) -> str:
    """Generic fallback renderer: pretty-prints whatever a section returned.
    Section-specific layouts can be added incrementally without changing
    the overall assembly — every section is guaranteed a working row even
    before a bespoke renderer exists for it."""
    ref_index = ref_index or {}
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            cols = list(data[0].keys())
            return _list_table(data, cols, ref_index)
        return "<pre>" + esc(_json.dumps(data, indent=2, default=str)[:5000]) + "</pre>"
    if isinstance(data, dict):
        rows = []
        for k, v in data.items():
            if isinstance(v, (list, dict)):
                continue
            if _ID_COL_RE.search(k):
                rows.append((k, _resolve_source({k: v}, [k], ref_index)))
            else:
                rows.append((k, esc(v)))
        extra = "".join(
            f"<h4>{esc(k)}</h4>" + render_section(k, v, ref_index)
            for k, v in data.items() if isinstance(v, (list, dict)))
        return _kv_table(rows) + extra
    return f"<p>{esc(data)}</p>"


def render_intelligence_html(payload: dict) -> str:
    ref_index = payload.get("_ref_index") or {}
    nav = "".join(f'<a href="#{key}">{esc(title)}</a>' for key, title in SECTION_TITLES)
    body = ""
    for key, title in SECTION_TITLES:
        body += f'<h2 id="{key}">{esc(title)}</h2>' + render_section(key, payload.get(key), ref_index)

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Website Intelligence — {esc(payload.get('url', ''))}</title>
<style>{_STYLE}</style></head><body>
<div class="layout">
<nav><strong>{esc(payload.get('url', ''))}</strong><hr>{nav}</nav>
<main>
<input class="searchbox" id="searchbox" type="text" placeholder="Search all sections…">
<h1>Website Intelligence Report</h1>
{body}
</main>
</div>
<script>{_SEARCH_JS}</script>
</body></html>"""
