#!/usr/bin/env python3
"""Fail if README links or repository-local command paths are missing."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shlex


MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
FENCED_BLOCK = re.compile(r"```(?P<language>[^\n]*)\n(?P<body>.*?)```", re.DOTALL)
LOCAL_PATH_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".csv"}
SHELL_LANGUAGES = {"bash", "sh", "shell", "console"}


def relative_markdown_targets(readme: str) -> list[str]:
    targets: list[str] = []
    for match in MARKDOWN_LINK.finditer(readme):
        target = match.group(1).strip().strip("<>")
        target = target.split("#", 1)[0]
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        targets.append(target)
    return targets


def _logical_shell_lines(body: str) -> list[str]:
    lines: list[str] = []
    pending = ""
    for raw in body.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pending = f"{pending} {stripped}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        lines.append(pending)
        pending = ""
    if pending:
        lines.append(pending)
    return lines


def relative_command_targets(readme: str) -> list[str]:
    targets: list[str] = []
    for block in FENCED_BLOCK.finditer(readme):
        language = block.group("language").strip().lower()
        if language not in SHELL_LANGUAGES:
            continue
        for line in _logical_shell_lines(block.group("body")):
            try:
                tokens = shlex.split(line)
            except ValueError as exception:
                raise ValueError(f"invalid shell command in README: {line!r}") from exception
            for token in tokens:
                if token.startswith("-") or token.startswith("$") or "=" in token:
                    continue
                if token in {".", ".."} or token.startswith(("http://", "https://")):
                    continue
                candidate = Path(token)
                if candidate.is_absolute():
                    continue
                if candidate.suffix.lower() in LOCAL_PATH_SUFFIXES:
                    targets.append(token)
    return targets


def check_readme(root: Path) -> dict[str, list[str]]:
    readme_path = root / "README.md"
    source = readme_path.read_text(encoding="utf-8")
    links = relative_markdown_targets(source)
    commands = relative_command_targets(source)
    missing_links = sorted(target for target in links if not (root / target).exists())
    missing_commands = sorted(
        target for target in commands if not (root / target).exists()
    )
    return {
        "relative_links": sorted(links),
        "relative_command_paths": sorted(commands),
        "missing_relative_links": missing_links,
        "missing_relative_command_paths": missing_commands,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = check_readme(args.repository.resolve())
    if result["missing_relative_links"] or result["missing_relative_command_paths"]:
        for kind in ("missing_relative_links", "missing_relative_command_paths"):
            for target in result[kind]:
                print(f"{kind}: {target}")
        return 1
    print(
        "README surface PASS: "
        f"{len(result['relative_links'])} links, "
        f"{len(result['relative_command_paths'])} command paths"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
