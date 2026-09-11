import argparse
from pathlib import Path
from . import pipeline


def main():
    parser = argparse.ArgumentParser(description="Ruan-style TextCraft debug pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--config", default="configs/debug.json")
    prep.add_argument("--run", required=True)
    prep.add_argument("--dry-run", action="store_true",
                      help="Build the tree and freeze tasks, print the fingerprint and selection, write nothing. "
                           "Run on two machines and diff to prove the task set is identical.")
    for command in ["collect", "extract", "audit", "analyze"]:
        p = sub.add_parser(command)
        p.add_argument("--run", required=True)
        if command in ("collect", "extract"):
            p.add_argument("--device", choices=["cpu", "mps", "cuda"])
        if command == "analyze":
            p.add_argument("--allow-incomplete", action="store_true",
                           help="Analyse despite missing episodes. Loss is not missing-at-random.")
    args = parser.parse_args()
    if args.command == "prepare":
        pipeline.prepare(args.config, args.run, dry_run=args.dry_run)
    elif args.command in ("collect", "extract"):
        getattr(pipeline, args.command)(Path(args.run), args.device)
    elif args.command == "audit":
        pipeline.audit(args.run)
    else:
        from .analysis import analyze
        analyze(args.run, allow_incomplete=args.allow_incomplete)


if __name__ == "__main__":
    main()
