"""LAYER 0 — Intelligent crawler completion + AI Invisibility Score (v3 Phase 1).

Adds on top of the base crawl4ai BFS crawl:
- sitemap-first seeding (nested indexes, ``lastmod`` captured)
- per-page enrichment: JSON-LD blocks, headings with level, internal/external
  links with anchor text, linked-PDF extraction (pymupdf)
- **4-bot UA re-fetch** (GPTBot, ClaudeBot, PerplexityBot, Google-Extended):
  plain HTTP fetch per bot (no JS), text-block diff vs the browser-rendered
  content -> ``access_gaps[{bot_name, missing_content[]}]`` per page
- robots.txt declared-vs-actual cross-check -> ``robots_conflict`` (cloaking)
- **AI Invisibility Score** — a MEASURED number (real diff, not a heuristic):
  per page ``invisible_ratio(bot) = chars(missing)/chars(rendered)``; site
  level weighted by internal PageRank and retrieval-winning pages.
"""
from __future__ import annotations

import io
import json as _json
import re
import xml.etree.ElementTree as ET
from typing import Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

from .core.logging import get_logger

logger = get_logger(__name__)

# Real bot tokens (robots.txt) + representative UA strings (2026).
BOT_UAS = {
    "GPTBot": "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko); compatible; "
              "GPTBot/1.2; +https://openai.com/gptbot",
    "ClaudeBot": "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko; compatible; "
                 "ClaudeBot/1.0; +claudebot@anthropic.com)",
    "PerplexityBot": "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko; "
                     "compatible; PerplexityBot/1.0; "
                     "+https://perplexity.ai/perplexitybot)",
    "Google-Extended": "Mozilla/5.0 (compatible; Google-Extended/1.0; "
                       "+http://www.google.com/bot.html)",
}

MIN_BLOCK_CHARS = 40          # blocks shorter than this are nav/noise
MAX_SITEMAP_URLS = 500
MAX_PDFS_PER_SITE = 5
HTTP_TIMEOUT = 20.0


# ─────────────────────────────────────────────────────────────────────────────
# Sitemap-first seeding
# ─────────────────────────────────────────────────────────────────────────────
def fetch_sitemap(root_url: str, _depth: int = 0) -> list[dict]:
    """Fetch sitemap.xml (+ nested indexes, depth<=2). [{url, lastmod}]."""
    import httpx
    base = f"{urlparse(root_url).scheme}://{urlparse(root_url).netloc}"
    sm_url = root_url if root_url.endswith(".xml") else urljoin(base, "/sitemap.xml")
    try:
        resp = httpx.get(sm_url, timeout=HTTP_TIMEOUT, follow_redirects=True)
        if resp.status_code != 200:
            logger.info(f"[layer0] no sitemap at {sm_url} ({resp.status_code})")
            return []
        return parse_sitemap_xml(resp.text, _depth)
    except Exception as e:
        logger.info(f"[layer0] sitemap fetch failed ({e})")
        return []


def parse_sitemap_xml(xml_text: str, _depth: int = 0) -> list[dict]:
    """Parse a urlset or sitemapindex document (namespace-tolerant)."""
    out: list[dict] = []
    try:
        root = ET.fromstring(xml_text.strip())
    except ET.ParseError as e:
        logger.warning(f"[layer0] sitemap XML parse error: {e}")
        return []
    strip = lambda tag: tag.rsplit("}", 1)[-1]
    if strip(root.tag) == "sitemapindex" and _depth < 2:
        for sm in root:
            loc = next((c.text for c in sm if strip(c.tag) == "loc"), None)
            if loc:
                out.extend(fetch_sitemap(loc.strip(), _depth + 1))
    elif strip(root.tag) == "urlset":
        for u in root:
            loc = next((c.text for c in u if strip(c.tag) == "loc"), None)
            lastmod = next((c.text for c in u if strip(c.tag) == "lastmod"), None)
            if loc:
                out.append({"url": loc.strip(),
                            "lastmod": lastmod.strip() if lastmod else None})
    return out[:MAX_SITEMAP_URLS]


# ─────────────────────────────────────────────────────────────────────────────
# Per-page enrichment (JSON-LD, headings, links w/ anchors, PDFs)
# ─────────────────────────────────────────────────────────────────────────────
def enrich_from_html(html: str, base_url: str) -> dict:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html or "", "html.parser")
    host = urlparse(base_url).netloc

    schema_jsonld = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            schema_jsonld.append(_json.loads(tag.string or ""))
        except (ValueError, TypeError):
            pass

    headings = [{"level": int(h.name[1]), "text": h.get_text(" ", strip=True)}
                for h in soup.find_all(re.compile(r"^h[1-6]$"))]

    internal, external, pdfs = [], [], []
    for a in soup.find_all("a", href=True):
        dest = urljoin(base_url, a["href"].strip())
        anchor = a.get_text(" ", strip=True)[:200]
        rec = {"anchor_text": anchor, "dest_url": dest}
        if dest.lower().split("?")[0].endswith(".pdf"):
            pdfs.append(dest)
        elif urlparse(dest).netloc == host:
            internal.append(rec)
        elif dest.startswith("http"):
            external.append(rec)
    return {"schema_jsonld": schema_jsonld, "headings": headings,
            "internal_links": internal, "external_links": external,
            "pdf_links": list(dict.fromkeys(pdfs))}


def extract_pdfs(pdf_urls: list[str], cap: int = MAX_PDFS_PER_SITE) -> list[dict]:
    """Download + extract linked PDFs (text + tables) via pymupdf."""
    import httpx
    out = []
    for url in pdf_urls[:cap]:
        try:
            import fitz  # pymupdf
            resp = httpx.get(url, timeout=HTTP_TIMEOUT, follow_redirects=True)
            resp.raise_for_status()
            doc = fitz.open(stream=io.BytesIO(resp.content), filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
            tables = []
            for page in doc:
                for t in (page.find_tables().tables or []):
                    tables.append(t.extract())
            doc.close()
            out.append({"url": url, "extracted_text": text, "tables": tables})
        except Exception as e:
            logger.warning(f"[layer0] PDF extraction failed for {url}: {e}")
            out.append({"url": url, "extracted_text": "", "tables": [],
                        "error": str(e)})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4-bot re-fetch + access-gap diff
# ─────────────────────────────────────────────────────────────────────────────
def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower())


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", _normalize(text)).strip()


def text_blocks(rendered_text: str) -> list[str]:
    """Meaningful content blocks from browser-rendered text/markdown.

    URLs are stripped: markdown flattens link HREFS into the text, but hrefs
    are attributes — never part of a bot's visible text — and comparing them
    marked every linked block "missing" (false 100% invisibility on a fully
    static page in the first live proof)."""
    blocks = []
    for raw in re.split(r"\n{1,}", rendered_text or ""):
        b = re.sub(r"https?://\S+|www\.\S+", " ", raw)  # hrefs aren't visible text
        b = re.sub(r"[#*_`>\[\]()|-]+", " ", b)          # strip md syntax
        b = re.sub(r"\s+", " ", b).strip()
        if len(b) >= MIN_BLOCK_CHARS:
            blocks.append(b)
    return blocks


def visible_text_from_html(html: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)


SHINGLE_WORDS = 4
SHINGLE_HIT_FLOOR = 0.5   # block counts as present if >=50% of its shingles match


def _block_present(block: str, bot_text_squashed: str) -> bool:
    """Shingle containment: robust to whitespace reflow / minor punctuation
    drift between markdown and raw-HTML text (whole-block substring was too
    brittle)."""
    words = _squash(block).split()
    if len(words) < SHINGLE_WORDS:
        return " ".join(words) in bot_text_squashed
    shingles = [" ".join(words[i:i + SHINGLE_WORDS])
                for i in range(len(words) - SHINGLE_WORDS + 1)]
    hits = sum(1 for s in shingles if s in bot_text_squashed)
    return hits / len(shingles) >= SHINGLE_HIT_FLOOR


def diff_blocks(rendered_text: str, bot_html: str) -> dict:
    """Blocks present after browser rendering but ABSENT in the bot's fetch."""
    blocks = text_blocks(rendered_text)
    bot_text = _squash(visible_text_from_html(bot_html))
    missing = [b for b in blocks if not _block_present(b, bot_text)]
    total = sum(len(b) for b in blocks)
    miss = sum(len(b) for b in missing)
    return {"missing_content": missing,
            "invisible_ratio": round(miss / total, 4) if total else 0.0,
            "total_chars": total, "missing_chars": miss}


def bot_refetch(url: str, rendered_text: str,
                bots: Optional[dict] = None) -> tuple[list[dict], dict]:
    """Re-fetch a page as each AI bot (plain HTTP, NO JS — matching real bots).

    Returns (access_gaps, statuses): access_gaps has an entry for EVERY bot
    (even if empty — that's the verification contract)."""
    import httpx
    bots = bots or BOT_UAS
    gaps, statuses = [], {}
    for bot, ua in bots.items():
        try:
            resp = httpx.get(url, headers={"User-Agent": ua},
                             timeout=HTTP_TIMEOUT, follow_redirects=True)
            statuses[bot] = resp.status_code
            d = diff_blocks(rendered_text, resp.text if resp.status_code == 200 else "")
        except Exception as e:
            statuses[bot] = -1
            d = {"missing_content": text_blocks(rendered_text),
                 "invisible_ratio": 1.0, "total_chars": 0, "missing_chars": 0}
            logger.warning(f"[layer0] {bot} fetch failed for {url}: {e}")
        gaps.append({"bot_name": bot, **d})
    return gaps, statuses


# ─────────────────────────────────────────────────────────────────────────────
# robots.txt declared-vs-actual cross-check
# ─────────────────────────────────────────────────────────────────────────────
def load_robots(root_url: str) -> Optional[RobotFileParser]:
    p = urlparse(root_url)
    rp = RobotFileParser(f"{p.scheme}://{p.netloc}/robots.txt")
    try:
        rp.read()
        return rp
    except Exception as e:
        logger.info(f"[layer0] robots.txt unreadable: {e}")
        return None


def robots_conflicts(rp: Optional[RobotFileParser], url: str,
                     statuses: dict) -> list[dict]:
    """Declared rule vs actual bot status. 200-despite-Disallow or
    403-despite-Allow = cloaking flag."""
    if rp is None:
        return []
    out = []
    for bot, status in statuses.items():
        declared_ok = rp.can_fetch(bot, url)
        if not declared_ok and status == 200:
            out.append({"bot_name": bot, "conflict": "fetchable_despite_disallow",
                        "status": status})
        elif declared_ok and status == 403:
            out.append({"bot_name": bot, "conflict": "blocked_despite_allow",
                        "status": status})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# AI Invisibility Score (MEASURED — real diff, not a heuristic)
# ─────────────────────────────────────────────────────────────────────────────
def invisibility_score(pages: list[dict],
                       winner_urls: Optional[set] = None) -> dict:
    """pages: [{"url", "access_gaps": [...], "internal_links": [...]}].

    Site score = Σ weight(page) * mean_bot_invisible_ratio(page) / Σ weight,
    weight = internal PageRank × 2 if the page holds retrieval-winning chunks.
    """
    import networkx as nx
    g = nx.DiGraph()
    for p in pages:
        g.add_node(p["url"])
        for l in p.get("internal_links", []):
            dest = l["dest_url"] if isinstance(l, dict) else l
            if any(dest == q["url"] for q in pages):
                g.add_edge(p["url"], dest)
    pr = nx.pagerank(g) if g.number_of_nodes() else {}

    winner_urls = winner_urls or set()
    per_page, total_w, weighted = [], 0.0, 0.0
    for p in pages:
        ratios = {gap["bot_name"]: gap["invisible_ratio"]
                  for gap in p.get("access_gaps", [])}
        mean_ratio = sum(ratios.values()) / len(ratios) if ratios else 0.0
        w = pr.get(p["url"], 1.0 / max(1, len(pages)))
        is_winner = p["url"] in winner_urls
        if is_winner:
            w *= 2.0
        total_w += w
        weighted += w * mean_ratio
        worst_gap = max(p.get("access_gaps", []),
                        key=lambda gap: gap["invisible_ratio"], default=None)
        per_page.append({
            "url": p["url"], "mean_invisible_ratio": round(mean_ratio, 4),
            "per_bot": ratios, "pagerank": round(w, 5),
            "retrieval_winner": is_winner,
            "worst_missing_blocks": (worst_gap or {}).get("missing_content", [])[:5],
        })
    site = round(weighted / total_w, 4) if total_w else 0.0
    per_page.sort(key=lambda r: -r["mean_invisible_ratio"])
    return {
        "label": "measured",   # real diff, not a heuristic
        "site_invisibility": site,
        "headline": (f"{site:.0%} of your answer-capable content never reaches "
                     f"GPTBot/ClaudeBot/PerplexityBot"),
        "weighting": "internal PageRank x2 for retrieval-winning pages"
                     + ("" if winner_urls else
                        " (no query winners supplied -> PageRank only)"),
        "worst_pages": per_page[:10],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run_layer0(url: str, crawl_results: list, trace=None,
               sitemap: Optional[list[dict]] = None) -> dict:
    """Enrich an existing crawl with Layer-0 data. Returns per-page records +
    sitemap stats. ``crawl_results`` = src.crawl CrawlResult list. Pass the
    ``sitemap`` already fetched for seeding to avoid a second fetch."""
    step_cm = trace.start_step("layer0", url=url,
                               n_pages=len(crawl_results)) if trace else None
    st = step_cm.__enter__() if step_cm else None
    try:
        if sitemap is None:
            sitemap = fetch_sitemap(url)
        crawled_urls = {r.url.rstrip("/") for r in crawl_results if r.success}
        captured = sum(1 for s in sitemap if s["url"].rstrip("/") in crawled_urls)
        sitemap_stats = {"sitemap_urls": len(sitemap), "captured": captured,
                         "capture_rate": round(captured / len(sitemap), 4)
                         if sitemap else None,
                         "lastmod_present": sum(1 for s in sitemap if s["lastmod"])}

        rp = load_robots(url)
        pages, all_pdf_links = [], []
        for r in crawl_results:
            if not r.success:
                continue
            enrich = enrich_from_html(r.html, r.url)
            gaps, statuses = bot_refetch(r.url, r.content)
            conflicts = robots_conflicts(rp, r.url, statuses)
            all_pdf_links.extend(enrich["pdf_links"])
            pages.append({
                "url": r.url, "title": r.title, "content": r.content,
                "html": r.html, **enrich,
                "access_gaps": gaps, "bot_statuses": statuses,
                "robots_conflict": conflicts,
            })
        pdfs = extract_pdfs(list(dict.fromkeys(all_pdf_links)))

        if st is not None:
            st.outputs["sitemap"] = sitemap_stats
            st.outputs["n_pages"] = len(pages)
            st.outputs["n_pdfs"] = len(pdfs)
            st.scores["pages_with_any_gap"] = float(sum(
                1 for p in pages
                if any(g["missing_content"] for g in p["access_gaps"])))
            conflicts_n = sum(len(p["robots_conflict"]) for p in pages)
            st.scores["robots_conflicts"] = float(conflicts_n)
        return {"pages": pages, "sitemap": sitemap, "sitemap_stats": sitemap_stats,
                "pdfs": pdfs}
    except BaseException as exc:
        if step_cm is not None:
            step_cm.__exit__(type(exc), exc, exc.__traceback__)
            step_cm = None
        raise
    finally:
        if step_cm is not None:
            step_cm.__exit__(None, None, None)
