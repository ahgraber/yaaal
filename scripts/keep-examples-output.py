#! /usr/bin/env python

"""Ensure ipynbs in examples/ dir have metadata set to retain cell outputs."""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_repo_path(file: str | Path) -> Path:
    """Identify repo path with git."""
    import subprocess

    repo = subprocess.check_output(  # NOQA: S603
        ["git", "rev-parse", "--show-toplevel"],  # NOQA: S607
        cwd=Path(file).parent,
        encoding="utf-8",
    ).strip()

    repo = Path(repo).expanduser().resolve()
    return repo


def main():
    """Ensure ipynbs in examples/ dir have metadata set to retain cell outputs."""
    import json

    repo = get_repo_path(__file__)

    ipynbs = sorted((repo / "examples").glob("*.ipynb"))
    logger.info(f"Found {[nb.name for nb in ipynbs]}")
    for ipynb in ipynbs:
        with ipynb.open("r") as f:
            notebook = json.load(f)

        # Add or update the keep_output field
        if "metadata" not in notebook:
            notebook["metadata"] = {}
        if "keep_output" in notebook["metadata"]:
            continue
        else:
            logger.info(f"Editing metadata for {str(ipynb.name)}")
            notebook["metadata"]["keep_output"] = True

        # Write back the updated notebook
        with ipynb.open("w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1)


if __name__ == "__main__":
    main()
