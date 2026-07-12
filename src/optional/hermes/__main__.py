"""
Hermes CLI — ``python -m src.orchestrator`` entry point.

Usage:
    python -m src.orchestrator scan --project my_brand --url https://example.com
    python -m src.orchestrator status --scan scan_xxx
    python -m src.orchestrator cancel --scan scan_xxx
    python -m src.orchestrator retry --scan scan_xxx
    python -m src.orchestrator recs --scan scan_xxx
    python -m src.orchestrator list
    python -m src.orchestrator worker
"""

import argparse
import sys
import json

from . import create_api

api = None  # lazy init


def _get_api():
    global api
    if api is None:
        api = create_api()
    return api


def cmd_scan(args):
    a = _get_api()
    result = a.create_scan(
        project_id=args.project,
        target_url=args.url,
        queries=args.queries or [],
        metadata={"seed_url": args.url},
    )
    a.start_scan(result["scan_id"])
    print(json.dumps(result, default=str, indent=2))
    print(f"\nRun `python -m src.orchestrator status --scan {result['scan_id']}` to check progress")


def cmd_status(args):
    a = _get_api()
    status = a.get_scan_status(args.scan)
    if status:
        print(json.dumps(status, default=str, indent=2))
    else:
        print(f"Scan not found: {args.scan}")
        sys.exit(1)


def cmd_cancel(args):
    a = _get_api()
    result = a.cancel_scan(args.scan)
    if result:
        print(f"Cancelled: {result['workflow_id']}")
    else:
        print(f"Scan not found: {args.scan}")
        sys.exit(1)


def cmd_retry(args):
    a = _get_api()
    result = a.retry_scan(args.scan)
    if result:
        print(f"Retried: {result['workflow_id']} ({result['status']})")
        a.start_scan(args.scan)
    else:
        print(f"Scan not found: {args.scan}")
        sys.exit(1)


def cmd_recs(args):
    a = _get_api()
    recs = a.get_recommendations(args.scan)
    if recs:
        print(json.dumps(recs, default=str, indent=2))
    else:
        print("No recommendations yet — scan may still be running")


def cmd_list(args):
    a = _get_api()
    scans = a.list_scans(limit=args.limit)
    if scans:
        print(f"{'SCAN ID':<24} {'STATUS':<14} {'PROJECT':<16} {'PROGRESS':<10} URL")
        print("-" * 90)
        for s in scans:
            url = (s.get("metadata") or {}).get("target_url", "")
            print(f"{s['scan_id']:<24} {s['status']:<14} {s['project_id']:<16} "
                  f"{s['progress_percent']:<10.1f} {url}")
    else:
        print("No scans found")


def cmd_worker(args):
    print("Starting Hermes worker (Ctrl+C to stop)...")
    a = _get_api()
    a.start_worker()
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        a.stop_worker()


def main():
    parser = argparse.ArgumentParser(prog="hermes", description="Hermes orchestration CLI")
    sub = parser.add_subparsers(dest="command")

    p_scan = sub.add_parser("scan", help="Create and start a new scan")
    p_scan.add_argument("--project", default="default", help="Project name")
    p_scan.add_argument("--url", required=True, help="Target URL to crawl")
    p_scan.add_argument("--queries", nargs="*", default=[], help="Seed queries")
    p_scan.set_defaults(func=cmd_scan)

    p_st = sub.add_parser("status", help="Check scan status")
    p_st.add_argument("--scan", required=True, help="Scan ID")
    p_st.set_defaults(func=cmd_status)

    p_cancel = sub.add_parser("cancel", help="Cancel a scan")
    p_cancel.add_argument("--scan", required=True, help="Scan ID")
    p_cancel.set_defaults(func=cmd_cancel)

    p_retry = sub.add_parser("retry", help="Retry a failed scan")
    p_retry.add_argument("--scan", required=True, help="Scan ID")
    p_retry.set_defaults(func=cmd_retry)

    p_recs = sub.add_parser("recs", help="Get recommendations for a scan")
    p_recs.add_argument("--scan", required=True, help="Scan ID")
    p_recs.set_defaults(func=cmd_recs)

    p_list = sub.add_parser("list", help="List recent scans")
    p_list.add_argument("--limit", type=int, default=10, help="Max results")
    p_list.set_defaults(func=cmd_list)

    p_worker = sub.add_parser("worker", help="Run the background worker")
    p_worker.set_defaults(func=cmd_worker)

    args = parser.parse_args()
    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
