#!/usr/bin/env python3
import logging
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 main3.py PORT")
    try:
        port = int(sys.argv[1])
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        raise SystemExit("PORT must be an integer from 1 to 65535")

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
    )
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", line_buffering=True)

    from agent.server import serve

    serve(port)


if __name__ == "__main__":
    main()
