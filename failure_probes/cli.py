import argparse
from pathlib import Path
from . import pipeline


def main():
    parser = argparse.ArgumentParser(description="Ruan-style TextCraft debug pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--config", default="configs/debug.json")
    prep.add_argument("--run", required=True)
    for command in ["collect", "extract", "audit", "analyze"]:
        p = sub.add_parser(command)
        p.add_argument("--run", required=True)
        if command in ("collect", "extract"):
            p.add_argument("--device", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()
    if args.command == "prepare":
        pipeline.prepare(args.config, args.run)
    elif args.command in ("collect", "extract"):
        getattr(pipeline, args.command)(Path(args.run), args.device)
    elif args.command == "audit":
        pipeline.audit(args.run)
    else:
        from .analysis import analyze
        analyze(args.run)


if __name__ == "__main__":
    main()
