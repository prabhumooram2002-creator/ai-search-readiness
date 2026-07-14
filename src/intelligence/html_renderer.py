"""Renders intelligence.json into a single self-contained intelligence.html
— sidebar nav (19 sections), client-side search (plain JS substring filter
over row text, no external library), works offline from disk."""
from __future__ import annotations

import json as _json

from .core_utils import esc, pct, score2

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


def _list_table(items: list[dict], columns: list[str]) -> str:
    if not items:
        return "<p class='note'>No rows.</p>"
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body = ""
    for item in items:
        cells = "".join(f"<td>{esc(item.get(c, ''))[:200]}</td>" for c in columns)
        body += f"<tr>{cells}</tr>"
    return f"<table><tr>{head}</tr>{body}</table>"


def render_section(key: str, data) -> str:
    """Generic fallback renderer: pretty-prints whatever a section returned.
    Section-specific layouts can be added incrementally without changing
    the overall assembly — every section is guaranteed a working row even
    before a bespoke renderer exists for it."""
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            cols = list(data[0].keys())[:6]
            return _list_table(data, cols)
        return "<pre>" + esc(_json.dumps(data, indent=2, default=str)[:5000]) + "</pre>"
    if isinstance(data, dict):
        rows = []
        for k, v in data.items():
            if isinstance(v, (list, dict)):
                continue
            rows.append((k, esc(v)))
        extra = "".join(
            f"<h4>{esc(k)}</h4>" + render_section(k, v)
            for k, v in data.items() if isinstance(v, (list, dict)))
        return _kv_table(rows) + extra
    return f"<p>{esc(data)}</p>"


def render_intelligence_html(payload: dict) -> str:
    nav = "".join(f'<a href="#{key}">{esc(title)}</a>' for key, title in SECTION_TITLES)
    body = ""
    for key, title in SECTION_TITLES:
        body += f'<h2 id="{key}">{esc(title)}</h2>' + render_section(key, payload.get(key))

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
