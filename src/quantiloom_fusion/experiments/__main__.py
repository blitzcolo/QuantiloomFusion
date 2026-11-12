"""CLI: python -m quantiloom_fusion.experiments run|summarize|protocol CONFIG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m quantiloom_fusion.experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run pending variants, resuming where rows exist")
    run_p.add_argument("config", type=Path)
    run_p.add_argument("--limit", type=int, default=None,
                       help="only the first N samples in the index")
    run_p.add_argument("--samples", default=None,
                       help="comma-separated sample ids")

    sum_p = sub.add_parser("summarize", help="rewrite summary.csv from the store")
    sum_p.add_argument("config", type=Path)

    pro_p = sub.add_parser("protocol", help="run or resume the chapter-6 protocol (see experiments.protocol_run)")
    pro_p.add_argument("config", type=Path)
    pro_p.add_argument("--limit", type=int, default=None, help="only the first N samples in the index")
    pro_p.add_argument("--samples", default=None, help="comma-separated sample ids")

    args = parser.parse_args(argv)
    if args.command == "protocol":
        from .protocol_run import run_protocol

        samples = args.samples.split(",") if args.samples else None
        return run_protocol(args.config, limit=args.limit, samples=samples)
    if args.command == "run":
        from .driver import run

        samples = args.samples.split(",") if args.samples else None
        return run(args.config, limit=args.limit, samples=samples)

    from .summary import summarize_config

    summarize_config(args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
