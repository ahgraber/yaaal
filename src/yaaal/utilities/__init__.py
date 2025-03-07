from datetime import datetime, timezone
import json
from pathlib import Path
import textwrap

from .log_helpers import LOG_FMT, basic_log_config, logging_redirect_tqdm
from ..types_.base import JSON

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


def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def detect_encoding(rawdata: bytes) -> str:
    """Detect the encoding of a byte string."""
    import chardet

    encoding = chardet.detect(rawdata)
    return encoding["encoding"] or "utf-8"


def format_json(data: JSON, width: int = 100, indent: int = 2, level: int = 0) -> str:
    """Format JSON data with proper indentation and line wrapping."""
    prefix = " " * (level * indent)

    if isinstance(data, dict):
        if not data:
            return "{}"

        lines = []
        lines.append("{")

        items = list(data.items())
        for i, (key, value) in enumerate(items):
            key_prefix = f'{prefix}  "{key}": '
            key_indent = " " * len(key_prefix)

            formatted_value = format_json(value, width=width, indent=indent, level=level + 1)

            if isinstance(value, str):
                # Handle each line segment separately
                segments = []
                for segment in value.split("\n"):
                    wrapped = textwrap.fill(
                        segment,
                        width=width - len(key_prefix),
                        initial_indent=key_indent,
                        subsequent_indent=key_indent + " ",
                        drop_whitespace=False,
                    )
                    segments.append(wrapped)
                # Join segments with newlines and proper indentation
                formatted_value = '"{}"'.format((key_indent + "\n").join(segments).strip())

            comma = "," if i < len(items) else ""
            first_line = f"{key_prefix}{formatted_value}{comma}"
            lines.append(first_line)

        lines.append(prefix + "}")
        return "\n".join(lines)

    elif isinstance(data, list):
        if not data:
            return "[]"

        lines = []
        lines.append("[")

        for i, item in enumerate(data):
            formatted_item = format_json(item, width, indent, level + 1)
            comma = "," if i <= len(data) - 1 else ""
            lines.append(f"{prefix}  {formatted_item}{comma}")

        lines.append(prefix + "]")
        return "\n".join(lines)

    elif isinstance(data, str):
        try:
            return format_json(json.loads(data), width, indent, level)
        except json.JSONDecodeError:
            return '"{}"'.format(data)
            # return data
    else:
        return str(data).lower()
