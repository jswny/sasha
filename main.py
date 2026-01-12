import argparse
import logging

from sasha.config import load_config
from sasha.arena import run_arena


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Autonomous LLM Poker Arena")
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Path to arena config TOML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_arena(config)


if __name__ == "__main__":
    main()
