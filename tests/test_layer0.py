"""Phase 1 — Layer 0 completion + AI Invisibility Score (src/layer0.py).

Offline: HTTP is stubbed; pymupdf generates its own test PDF. Live-site proof
(JS-heavy vs static + sitemap capture) is in BUILD_LOG.md.
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src import layer0 as L0


URLSET = """<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://a.com/p1</loc><lastmod>2026-01-01</lastmod></url>
  <url><loc>https://a.com/p2</loc></url>
</urlset>"""

SITEMAPINDEX = """<?xml version="1.0"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://a.com/nested.xml</loc></sitemap>
</sitemapindex>"""

HTML = """
<html><head>
<script type="application/ld+json">{"@type": "Organization", "sameAs": ["x"]}</script>
</head><body>
<h1>Main Title</h1><h2>Sub</h2>
<a href="/inner">Inner page link</a>
<a href="https://other.com/x">External ref</a>
<a href="/doc.pdf">The PDF</a>
<script>var hidden = "js only";</script>
<p>This paragraph is fully visible to every crawler that fetches the page.</p>
</body></html>"""


def test_sitemap_urlset_and_nested_index(monkeypatch):
    urls = L0.parse_sitemap_xml(URLSET)
    assert urls == [{"url": "https://a.com/p1", "lastmod": "2026-01-01"},
                    {"url": "https://a.com/p2", "lastmod": None}]
    # index recurses into nested sitemaps via fetch_sitemap
    monkeypatch.setattr(L0, "fetch_sitemap",
                        lambda u, _depth=0: [{"url": "https://a.com/deep", "lastmod": None}])
    nested = L0.parse_sitemap_xml(SITEMAPINDEX)
    assert nested == [{"url": "https://a.com/deep", "lastmod": None}]


def test_enrich_from_html():
    e = L0.enrich_from_html(HTML, "https://a.com/page")
    assert e["schema_jsonld"][0]["@type"] == "Organization"
    assert {"level": 1, "text": "Main Title"} in e["headings"]
    assert any(l["dest_url"] == "https://a.com/inner" and
               l["anchor_text"] == "Inner page link" for l in e["internal_links"])
    assert any(l["dest_url"] == "https://other.com/x" for l in e["external_links"])
    assert e["pdf_links"] == ["https://a.com/doc.pdf"]


def test_diff_blocks_finds_js_only_content():
    rendered = ("Static sentence that appears in the raw HTML body of the page.\n"
                "Dynamic quote injected by JavaScript after the page has loaded up.")
    bot_html = "<html><body><p>Static sentence that appears in the raw HTML " \
               "body of the page.</p></body></html>"
    d = L0.diff_blocks(rendered, bot_html)
    assert len(d["missing_content"]) == 1
    assert "Dynamic quote" in d["missing_content"][0]
    assert 0.0 < d["invisible_ratio"] < 1.0

    # static page: nothing missing
    full = L0.diff_blocks(
        "Static sentence that appears in the raw HTML body of the page.", bot_html)
    assert full["missing_content"] == [] and full["invisible_ratio"] == 0.0


def test_bot_refetch_covers_all_four_bots(monkeypatch):
    class FakeResp:
        status_code = 200
        text = "<html><body>Visible content only, nothing rendered by scripts here.</body></html>"

    import httpx
    seen_uas = []
    def fake_get(url, headers=None, timeout=None, follow_redirects=None):
        seen_uas.append(headers["User-Agent"])
        return FakeResp()
    monkeypatch.setattr(httpx, "get", fake_get)

    gaps, statuses = L0.bot_refetch(
        "https://a.com/p", "Visible content only, nothing rendered by scripts here.")
    assert [g["bot_name"] for g in gaps] == \
        ["GPTBot", "ClaudeBot", "PerplexityBot", "Google-Extended"]
    assert all(g["missing_content"] == [] for g in gaps)   # present even if empty
    assert set(statuses.values()) == {200}
    assert len(seen_uas) == 4 and all("Bot" in ua or "Extended" in ua for ua in seen_uas)


def test_robots_conflicts():
    class FakeRP:
        def can_fetch(self, bot, url):
            return bot != "GPTBot"   # GPTBot declared disallowed
    conflicts = L0.robots_conflicts(FakeRP(), "https://a.com/p",
                                    {"GPTBot": 200, "ClaudeBot": 403,
                                     "PerplexityBot": 200, "Google-Extended": 200})
    kinds = {(c["bot_name"], c["conflict"]) for c in conflicts}
    assert ("GPTBot", "fetchable_despite_disallow") in kinds   # cloaking flag
    assert ("ClaudeBot", "blocked_despite_allow") in kinds
    assert len(conflicts) == 2


def test_invisibility_score_weighting_and_label():
    pages = [
        {"url": "https://a.com/hub",
         "access_gaps": [{"bot_name": b, "invisible_ratio": 0.8,
                          "missing_content": ["hidden block"]} for b in L0.BOT_UAS],
         "internal_links": [{"dest_url": "https://a.com/leaf"}]},
        {"url": "https://a.com/leaf",
         "access_gaps": [{"bot_name": b, "invisible_ratio": 0.0,
                          "missing_content": []} for b in L0.BOT_UAS],
         "internal_links": [{"dest_url": "https://a.com/hub"}]},
    ]
    plain = L0.invisibility_score(pages)
    assert plain["label"] == "measured"
    assert 0.0 < plain["site_invisibility"] < 0.8
    assert plain["worst_pages"][0]["url"] == "https://a.com/hub"
    assert plain["worst_pages"][0]["worst_missing_blocks"] == ["hidden block"]

    # winner weighting: making the invisible page a retrieval winner raises the score
    boosted = L0.invisibility_score(pages, winner_urls={"https://a.com/hub"})
    assert boosted["site_invisibility"] > plain["site_invisibility"]


def test_pdf_extraction_roundtrip(monkeypatch, tmp_path):
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "PDF body text for extraction test.")
    pdf_bytes = doc.tobytes()
    doc.close()

    class FakeResp:
        content = pdf_bytes
        def raise_for_status(self):
            pass
    import httpx
    monkeypatch.setattr(httpx, "get",
                        lambda url, timeout=None, follow_redirects=None: FakeResp())
    out = L0.extract_pdfs(["https://a.com/doc.pdf"])
    assert len(out) == 1
    assert "PDF body text" in out[0]["extracted_text"]


def test_linked_block_with_href_is_not_false_missing():
    # regression: markdown flattens hrefs into text; hrefs are never visible
    # bot text, and this false-flagged every linked block (live-proof bug)
    rendered = "Quotes to Scrape is a site full of famous sayings https://quotes.toscrape.com/"
    bot_html = "<html><body><a href='/'>Quotes to Scrape</a> is a site full of famous sayings</body></html>"
    d = L0.diff_blocks(rendered, bot_html)
    assert d["missing_content"] == []
    assert d["invisible_ratio"] == 0.0
