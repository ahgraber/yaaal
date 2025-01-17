import logging
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Annotated

from pydantic import AfterValidator, BeforeValidator, TypeAdapter, ValidationError

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


def path_is_valid(path: Path | str) -> Path:
    """Check whether full path can be resolved."""
    path = Path(path) if isinstance(path, str) else path
    path = path.expanduser().absolute()  # resolve ~/ -> /home/<username/ and ../../
    _ = path.resolve()  # make sure symlinks can be resolved, but dont return resolved link
    return path


def path_is_dir(path: Path | str) -> Path:
    """Test whether path is dir."""
    path = path_is_valid(path)
    if os.path.isdir(path):
        if os.access(path, os.R_OK):
            return path
        else:
            raise PermissionError(f"Path is not readable: {path}")
    else:
        raise NotADirectoryError(f"Path is not a directory: {path}")


def path_is_file(path: Path | str) -> Path:
    """Test whether path is file."""
    path = path_is_valid(path)
    if os.path.isfile(path):
        if os.access(path, os.R_OK):
            return path
        else:
            raise PermissionError(f"Path is not readable: {path}")
    else:
        raise FileNotFoundError(f"Path is not a file: {path}")
