"""CLI entry point for the Sansin MCP proxy.

Usage::

    sansin-mcp-proxy --upstream stdio:///path/to/server
    sansin-mcp-proxy --upstream stdio://npx -y @modelcontextprotocol/server-fs /tmp
    sansin-mcp-proxy --upstream /path/to/server.py
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from typing import List, Tuple

logger = logging.getLogger("sansin.mcp.cli")


def _parse_upstream(raw: str) -> Tuple[str, List[str]]:
    """Parse an upstream URI into (command, args).

    Accepted formats:
        stdio:///path/to/server
        stdio://command arg1 arg2
        /path/to/server.py          (bare path, auto-detect interpreter)
    """
    if raw.startswith("stdio://"):
        rest = raw[len("stdio://") :]
        parts = rest.split()
        if not parts or not parts[0]:
            raise argparse.ArgumentTypeError(
                f"Invalid upstream URI: {raw!r} — nothing after stdio://"
            )
        return parts[0], parts[1:]

    # Bare path — auto-detect interpreter.
    parts = raw.split()
    path = parts[0]
    extra_args = parts[1:]

    if path.endswith(".py"):
        return sys.executable, [path] + extra_args
    if path.endswith(".js") or path.endswith(".mjs"):
        return "node", [path] + extra_args

    # Assume it's a binary.
    return path, extra_args


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sansin-mcp-proxy",
        description="Sansin MCP safety proxy — sits between an MCP client and upstream server.",
    )
    parser.add_argument(
        "--upstream",
        required=True,
        help=(
            "URI of the upstream MCP server. "
            "Formats: stdio:///path/to/server, stdio://cmd arg1 arg2, "
            "or a bare path (auto-detects python/node)."
        ),
    )
    parser.add_argument(
        "--db-path",
        default="~/.sansin/decisions.db",
        help="Path to SQLite database (default: ~/.sansin/decisions.db)",
    )
    parser.add_argument(
        "--prefix",
        default="sansin_",
        help="Prefix for Sansin management tools (default: sansin_)",
    )
    parser.add_argument(
        "--fail-closed",
        action="store_true",
        help="Block tool calls on internal errors instead of failing open",
    )
    parser.add_argument(
        "--no-community",
        action="store_true",
        help="Don't load community priors on startup",
    )
    return parser


def main() -> None:
    """CLI entry point for sansin-mcp-proxy."""
    parser = _build_parser()
    args = parser.parse_args()

    # Ensure logs go to stderr (stdout is the MCP stdio transport).
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    command, upstream_args = _parse_upstream(args.upstream)
    db_path = os.path.expanduser(args.db_path)

    logger.info("Sansin MCP proxy starting")
    logger.info("  upstream: %s %s", command, " ".join(upstream_args))
    logger.info("  db_path:  %s", db_path)
    logger.info("  prefix:   %s", args.prefix)
    logger.info("  fail_closed: %s", args.fail_closed)
    logger.info("  community:   %s", not args.no_community)

    from sansin_core.mcp.proxy import SansinProxy

    proxy = SansinProxy(
        upstream_command=command,
        upstream_args=upstream_args,
        db_path=db_path,
        prefix=args.prefix,
        fail_closed=args.fail_closed,
    )

    # Graceful shutdown on SIGINT/SIGTERM.
    def _handle_signal(signum: int, _frame: object) -> None:
        logger.info("Received signal %d, shutting down", signum)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    proxy.run()


if __name__ == "__main__":
    main()
