"""The ``lmhdx`` command: run a schema 3 TOML file, or convert an older one (docs/reference/cli.md)."""

from __future__ import annotations

import json
import sys

from .config import run, upgrade

_USAGE = "usage: lmhdx CASE.toml [table.key=value ...]\n       lmhdx upgrade OLD.toml > CASE.toml"


def main(argv: list[str] | None = None) -> int:
    """Run ``lmhdx``; the exit status is 0, or 2 when a run was rejected."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "upgrade":
        print(upgrade(argv[1]), end="")
        return 0
    if argv[:1] in (["-h"], ["--help"]):
        print(_USAGE)
        return 0
    if not argv or not argv[0].endswith(".toml"):
        print(_USAGE, file=sys.stderr)
        return 2
    print(json.dumps(summary := run(*argv), indent=2))
    if summary.get("accepted") is False:
        rejected = summary["rejected"]
        print(f"rejected at step {rejected['step']}: {', '.join(rejected['reasons'])}", file=sys.stderr)
        return 2
    return 0
