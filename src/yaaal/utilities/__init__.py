from pathlib import Path

from .log_helpers import LOG_FMT, basic_log_config, logging_redirect_tqdm

__all__ = [
    "LOG_FMT",
    "basic_log_config",
    "logging_redirect_tqdm",
    "get_repo_path",
    "to_snake_case",
    "detect_encoding",
]


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


def to_snake_case(text: str) -> str:
    """Convert text to snake_case."""
    import re

    # Replace spaces and hyphens with underscores
    text = text.strip()
    text = re.sub(r"[\s-]+", "_", text)

    # Convert camelCase, PascalCase, and cases like HTTPHeader to snake_case
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z0-9])", r"\1_\2", text)

    # Convert to lowercase
    return text.lower()


def detect_encoding(rawdata: bytes) -> str:
    """Detect the encoding of a byte string."""
    import chardet

    encoding = chardet.detect(rawdata)
    return encoding["encoding"] or "utf-8"
