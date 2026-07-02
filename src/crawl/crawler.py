"""Crawl4AI wrapper — respects robots.txt, caps pages/depth, auto-detects JS pages."""
import asyncio
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse, urljoin
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from ..core.config import crawl
from ..core.logging import get_logger

logger = get_logger(__name__)

# ── URL normalization ─────────────────────────────────────────────────
def normalize_url(url: str) -> str:
    """Normalize URL for dedup: strip trailing slash, lowercase host, drop tracking params."""
    if not url:
        return url
    try:
        parsed = urlparse(url)
        # Lowercase host
        netloc = parsed.netloc.lower()
        # Strip trailing slash from path
        path = parsed.path.rstrip("/") or "/"
        # Drop common tracking params
        query = parsed.query
        if query:
            keep = []
            for param in query.split("&"):
                key = param.split("=")[0].lower()
                if key not in ("utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
                               "fbclid", "gclid", "ref", "source", "variant"):
                    keep.append(param)
            query = "&".join(keep)
        # Rebuild
        normalized = f"{parsed.scheme}://{netloc}{path}"
        if query:
            normalized += f"?{query}"
        if parsed.fragment:
            normalized += f"#{parsed.fragment}"
        return normalized
    except Exception:
        return url


def resolve_url(base: str, href: str) -> Optional[str]:
    """Resolve a potentially relative URL to absolute, return None if external."""
    if not href or href.startswith("#") or href.startswith("javascript:"):
        return None
    # Clean the href: strip whitespace and cut at common corruption markers
    href = href.strip().split()[0].split("%20")[0]
    if not href:
        return None
    href_lower = href.lower()

    # Skip common non-page paths
    skip_patterns = ('/cdn/', '/checkout/', '/cdn-cgi/', '.well-known/')
    for pat in skip_patterns:
        if pat in href_lower:
            return None
    # Also skip links to non-HTML resources (images, CSS, JS, fonts, etc.)
    import re
    if re.search(r'\.(jpg|jpeg|png|gif|svg|webp|ico|css|js|json|xml|woff2?|ttf|eot|pdf|zip|mp[34]|atom|oembed)(\?|#|$)', href_lower):
        return None
    try:
        # urljoin fails on malformed URLs — check if it's already a valid URL
        if href.startswith("http://") or href.startswith("https://"):
            absolute = href
        else:
            absolute = urljoin(base, href)
        # Only keep same-domain links
        base_domain = urlparse(base).netloc.lower().replace("www.", "")
        abs_domain = urlparse(absolute).netloc.lower().replace("www.", "")
        if abs_domain != base_domain and not abs_domain.endswith("." + base_domain):
            return None
        return normalize_url(absolute)
    except Exception:
        return None

@dataclass
class CrawlResult:
    url: str
    title: str
    content: str          # cleaned text (markdown)
    html: str             # raw HTML
    internal_links: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    success: bool = False
    error: str = ""


async def _crawl_page(
    crawler: AsyncWebCrawler,
    url: str,
    run_cfg: CrawlerRunConfig,
) -> CrawlResult:
    """Crawl a single URL using crawl4ai 0.9.x API (arun method)."""
    try:
        # arun() returns CrawlResultContainer directly
        result = await crawler.arun(url, config=run_cfg)

        # Extract internal links
        internal_links = []
        if result.links:
            for link in result.links.get("internal", []):
                href = link.get("href", "")
                resolved = resolve_url(url, href)
                if resolved:
                    internal_links.append(resolved)
        
        # Also extract links from raw HTML (crawl4ai sometimes misses blog post links)
        if result.html:
            import re
            html_links = re.findall(r'href=["\']((?:/|https?://)[^"\']+)["\']', result.html)
            for href in html_links:
                resolved = resolve_url(url, href)
                if resolved and resolved not in internal_links:
                    internal_links.append(resolved)

        # Use markdown as content (cleaner than raw HTML)
        md = result.markdown
        if hasattr(md, "raw_markdown"):
            content = md.raw_markdown or ""
        elif isinstance(md, str):
            content = md or ""
        else:
            content = str(md) if md else ""

        # Title from metadata
        title = ""
        if result.metadata:
            title = result.metadata.get("title", "") or ""

        return CrawlResult(
            url=result.url or url,
            title=title,
            content=content,
            html=result.html or "",
            internal_links=internal_links,
            metadata=result.metadata or {},
            success=result.success,
        )
    except Exception as exc:
        logger.warning(f"Crawl failed for {url}: {exc}")
        return CrawlResult(url=url, title="", content="", html="", success=False, error=str(exc))


async def crawl_site(start_url: str) -> list[CrawlResult]:
    """
    Crawl a site starting from start_url.

    Depth + page capping handled in Python loop.
    Uses Playwright (chromium) for JS-rendered sites.
    """
    cfg = crawl()
    max_pages = cfg.get("max_pages", 200)
    max_depth = cfg.get("max_depth", 3)
    respect_robots = cfg.get("respect_robots_txt", True)
    delay_ms = cfg.get("delay_ms", 500)

    # Browser config — Playwright for JS-rendered sites
    browser_cfg = BrowserConfig(
        headless=True,
        extra_args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
        ],
    )

    # Run config — robots.txt respected via check_robots_txt
    run_cfg = CrawlerRunConfig(
        # Only internal links
        exclude_external_links=True,
        # Robots.txt
        check_robots_txt=respect_robots,
        # Polite delay between requests
        mean_delay=delay_ms / 1000,
        max_range=0.1,
    )

    logger.info(f"Starting crawl: {start_url} (max_pages={max_pages}, max_depth={max_depth})")

    all_results: list[CrawlResult] = []
    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(normalize_url(start_url), 0)]

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        while queue and len(all_results) < max_pages:
            url, depth = queue.pop(0)

            if url in visited:
                continue
            visited.add(url)

            if depth > max_depth:
                continue

            logger.info(f"Crawling [{len(all_results)+1}/{max_pages}] depth={depth}: {url}")

            result = await _crawl_page(crawler, url, run_cfg)
            all_results.append(result)

            # Add internal links to queue
            if result.success and result.internal_links:
                for link in result.internal_links:
                    link = normalize_url(link)
                    if link not in visited and len(all_results) < max_pages:
                        queue.append((link, depth + 1))

            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000)

    success_count = sum(1 for r in all_results if r.success)
    logger.info(f"Crawl complete: {success_count}/{len(all_results)} pages successful")

    return all_results


def crawl_site_sync(start_url: str) -> list[CrawlResult]:
    """Synchronous wrapper — handles nested event loops."""
    import concurrent.futures
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(crawl_site(start_url))
    else:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, crawl_site(start_url))
            return future.result()