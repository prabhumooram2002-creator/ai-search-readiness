"""PHASE 9 — Server log ingestion (v3 brief). The wedge no competitor has.

Parse Common/Combined access logs, filter AI-bot user agents, and join hits to
Page nodes -> a crawl-demand map: which pages bots ACTUALLY visit vs which pages
hold the retrieval-winning chunks. Cross-referenced with Layer-0 access_gaps,
"bot visited AND content invisible" is the highest-priority fix.

Malformed lines are counted and skipped — never fatal.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from .core.logging import get_logger

logger = get_logger(__name__)

# AI crawler + AI-search fetcher UA tokens (2026).
BOT_UA_TOKENS = {
    "GPTBot": "gptbot",
    "OAI-SearchBot": "oai-searchbot",
    "ChatGPT-User": "chatgpt-user",
    "ClaudeBot": "claudebot",
    "Claude-SearchBot": "claude-searchbot",
    "PerplexityBot": "perplexitybot",
    "Perplexity-User": "perplexity-user",
    "Google-Extended": "google-extended",
    "Meta-ExternalAgent": "meta-externalagent",
    "Bytespider": "bytespider",
}

# Combined Log Format: host - - [time] "METHOD path proto" status size "ref" "ua"
_COMBINED = re.compile(
    r'(?P<host>\S+)\s+\S+\s+\S+\s+\[(?P<time>[^\]]+)\]\s+'
    r'"(?P<method>[A-Z]+)\s+(?P<path>\S+)\s+[^"]*"\s+'
    r'(?P<status>\d{3})\s+(?P<size>\S+)'
    r'(?:\s+"(?P<referer>[^"]*)"\s+"(?P<ua>[^"]*)")?')


def parse_log_line(line: str) -> Optional[dict]:
    """Parse one Combined/Common log line. Returns None if malformed."""
    m = _COMBINED.match(line.strip())
    if not m:
        return None
    return {"host": m.group("host"), "path": m.group("path"),
            "status": int(m.group("status")), "ua": m.group("ua") or ""}


def classify_bot(ua: str) -> Optional[str]:
    ua_l = ua.lower()
    for name, token in BOT_UA_TOKENS.items():
        if token in ua_l:
            return name
    return None


def _norm_path(path: str) -> str:
    return urlparse(path).path.rstrip("/") or "/"


def import_logs(
    log_paths: list[str],
    site_pages: list[dict],
    winning_pages: Optional[list[str]] = None,
    access_gaps_by_url: Optional[dict[str, float]] = None,
    trace=None,
) -> dict:
    """Parse logs, count AI-bot hits per page, build the crawl-demand map.

    site_pages: [{"url"}] known Page nodes. winning_pages: urls holding
    retrieval-winning chunks. access_gaps_by_url: url -> mean invisible ratio.
    """
    step_cm = trace.start_step("server_logs", n_files=len(log_paths)) if trace else None
    st = step_cm.__enter__() if step_cm else None
    try:
        # map site page url-path -> full url
        path_to_url = {_norm_path(urlparse(p["url"]).path): p["url"] for p in site_pages}
        winning = set(winning_pages or [])
        gaps = access_gaps_by_url or {}

        per_bot: dict[str, int] = {}
        per_page_bot: dict[str, dict[str, int]] = {}
        total = malformed = matched = 0

        for lp in log_paths:
            for raw in Path(lp).read_text(encoding="utf-8", errors="replace").splitlines():
                if not raw.strip():
                    continue
                total += 1
                rec = parse_log_line(raw)
                if rec is None:
                    malformed += 1
                    continue
                bot = classify_bot(rec["ua"])
                if not bot:
                    continue
                matched += 1
                per_bot[bot] = per_bot.get(bot, 0) + 1
                url = path_to_url.get(_norm_path(rec["path"]))
                if url:
                    per_page_bot.setdefault(url, {})[bot] = \
                        per_page_bot.setdefault(url, {}).get(bot, 0) + 1

        visited = set(per_page_bot)
        all_urls = {p["url"] for p in site_pages}
        never_visited = sorted(all_urls - visited)
        # winning pages the bots never touched — the killer finding
        winners_never_visited = sorted(winning & set(never_visited))
        # bot visited AND content invisible -> highest priority
        visited_but_invisible = sorted(
            u for u in visited if gaps.get(u, 0.0) > 0.0)

        result = {
            "lines_total": total, "lines_malformed": malformed,
            "bot_hits": matched, "per_bot": dict(sorted(
                per_bot.items(), key=lambda kv: -kv[1])),
            "per_page": {u: dict(b) for u, b in per_page_bot.items()},
            "never_visited": never_visited,
            "winning_pages_never_visited": winners_never_visited,
            "visited_but_invisible": visited_but_invisible,
            "headline": (f"AI bots hit {len(visited)} of {len(all_urls)} pages; "
                         f"{len(winners_never_visited)} page(s) that answer your "
                         f"top queries were never visited"),
        }
        logger.info(f"[serverlogs] {total} lines, {malformed} malformed, "
                    f"{matched} AI-bot hits across {len(visited)} pages; "
                    f"per_bot={result['per_bot']}")
        if st is not None:
            st.outputs.update({k: result[k] for k in
                               ("per_bot", "never_visited",
                                "winning_pages_never_visited", "visited_but_invisible")})
            st.scores["bot_hits"] = float(matched)              # measured
            st.scores["malformed"] = float(malformed)
        return result
    except BaseException as exc:
        if step_cm is not None:
            step_cm.__exit__(type(exc), exc, exc.__traceback__)
            step_cm = None
        raise
    finally:
        if step_cm is not None:
            step_cm.__exit__(None, None, None)
