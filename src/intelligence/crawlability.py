"""Section 9 — AI Crawlability [EXISTS]"""
from __future__ import annotations


def crawlability_report(invisibility: dict, l0_pages: list[dict],
                        struct_by_chunk: dict, fixes_manifest: list[dict] | None,
                        structure_score_floor: float = 0.6) -> dict:
    per_bot_table = []
    for wp in (invisibility or {}).get("worst_pages", []):
        for bot, ratio in (wp.get("per_bot") or {}).items():
            per_bot_table.append({"url": wp["url"], "bot": bot,
                                  "invisible_ratio": round(ratio, 4)})

    robots_issues = sum(len(p.get("robots_conflict", [])) for p in l0_pages)

    llms_txt = None
    for m in (fixes_manifest or []):
        if m.get("type") == "llms_txt":
            llms_txt = {"present": True, "valid": m.get("valid", True), "path": m["path"]}
    if llms_txt is None:
        llms_txt = {"present": False, "valid": False,
                   "note": "not generated this run — pass --fixes"}

    extractable = sum(1 for s in struct_by_chunk.values()
                      if (s.get("structure_score") or 0) >= structure_score_floor)
    total_chunks = len(struct_by_chunk) or 1

    return {
        "per_bot_table": per_bot_table,
        "robots_conflicts": robots_issues,
        "llms_txt": llms_txt,
        "extractable_answer_blocks": extractable,
        "extractable_answer_block_rate": round(extractable / total_chunks, 4),
        "structure_score_floor": structure_score_floor,
    }
