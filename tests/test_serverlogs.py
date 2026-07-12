"""Phase 9 — server log ingestion (src/serverlogs.py)."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src import serverlogs as sl
from src.trace import Trace


GPTBOT_UA = ('"Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko); '
             'compatible; GPTBot/1.2; +https://openai.com/gptbot"')
CLAUDEBOT_UA = '"Mozilla/5.0 (compatible; ClaudeBot/1.0; +claudebot@anthropic.com)"'
HUMAN_UA = '"Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120"'


def _line(path, ua, status=200):
    return (f'66.249.66.1 - - [12/Jul/2026:10:00:00 +0000] '
            f'"GET {path} HTTP/1.1" {status} 1234 "-" {ua}')


LOG = "\n".join([
    _line("/pricing", GPTBOT_UA),
    _line("/pricing", GPTBOT_UA),
    _line("/about", CLAUDEBOT_UA),
    _line("/blog", HUMAN_UA),            # human -> ignored
    "this is a malformed line !!!",       # malformed -> counted, skipped
    _line("/pricing", CLAUDEBOT_UA),
])

PAGES = [{"url": "https://a.com/pricing"}, {"url": "https://a.com/about"},
         {"url": "https://a.com/features"}]   # /features never in logs


def test_parse_and_classify():
    rec = sl.parse_log_line(_line("/x", GPTBOT_UA))
    assert rec["path"] == "/x" and rec["status"] == 200
    assert sl.classify_bot(rec["ua"]) == "GPTBot"
    assert sl.classify_bot("Mozilla/5.0 Chrome") is None
    assert sl.parse_log_line("garbage") is None


def test_per_bot_counts_and_malformed(tmp_path):
    f = tmp_path / "access.log"
    f.write_text(LOG, encoding="utf-8")
    trace = Trace(query="q")
    r = sl.import_logs([str(f)], PAGES, trace=trace)
    assert r["per_bot"] == {"GPTBot": 2, "ClaudeBot": 2}   # human excluded
    assert r["lines_malformed"] == 1                        # counted, not crashed
    assert r["bot_hits"] == 4
    assert r["per_page"]["https://a.com/pricing"] == {"GPTBot": 2, "ClaudeBot": 1}
    assert trace.steps[0].scores["bot_hits"] == 4.0


def test_never_visited_includes_kg_pages_absent_from_logs(tmp_path):
    f = tmp_path / "access.log"
    f.write_text(LOG, encoding="utf-8")
    r = sl.import_logs([str(f)], PAGES)
    # /features is a Page node but never appears in logs
    assert "https://a.com/features" in r["never_visited"]
    assert "https://a.com/pricing" not in r["never_visited"]


def test_winning_page_never_visited_is_the_killer_finding(tmp_path):
    f = tmp_path / "access.log"
    f.write_text(LOG, encoding="utf-8")
    r = sl.import_logs([str(f)], PAGES,
                       winning_pages=["https://a.com/features"])
    assert r["winning_pages_never_visited"] == ["https://a.com/features"]
    assert "never visited" in r["headline"]


def test_visited_but_invisible_cross_ref(tmp_path):
    f = tmp_path / "access.log"
    f.write_text(LOG, encoding="utf-8")
    r = sl.import_logs([str(f)], PAGES,
                       access_gaps_by_url={"https://a.com/pricing": 0.8})
    # bots visited /pricing AND its content is invisible -> highest priority
    assert "https://a.com/pricing" in r["visited_but_invisible"]
    assert "https://a.com/about" not in r["visited_but_invisible"]
