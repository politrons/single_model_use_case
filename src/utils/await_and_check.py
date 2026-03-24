from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass

# ----------------------------- logging -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOG = logging.getLogger("databricks_mlops_stack.utils.await_and_check")

@dataclass(frozen=True)
class Config:
    seconds: int


def parse_args(argv: list[str]) -> Config:
    parser = argparse.ArgumentParser(description="Sleep for a number of seconds.")
    parser.add_argument("--seconds", required=True, type=int, help="Seconds to sleep (>= 0)")
    args = parser.parse_args(argv)
    if args.seconds < 0:
        parser.error("--seconds must be >= 0")
    return Config(seconds=args.seconds)


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv or sys.argv[1:])
    LOG.info("Sleeping for %s seconds...",cfg.seconds)
    time.sleep(cfg.seconds)
    LOG.info("Woke up. Continuing.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
