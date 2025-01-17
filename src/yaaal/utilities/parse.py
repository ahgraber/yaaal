# ruff: NOQA: E731
import logging
import re

logger = logging.getLogger(__name__)


def to_snake_case(text: str) -> str:
    """Convert text to snake_case."""
    # Replace spaces and hyphens with underscores
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
    logger.info(encoding)
    return encoding["encoding"] or "utf-8"


def check_matched_pairs(string: str, open_char="(", close_char=")"):
    """Check that all parentheses in a string are balanced and nested properly."""
    count = 0
    for c in string:
        if c == open_char:
            count += 1
        elif c == close_char:
            count -= 1
        if count < 0:
            return False
    return count == 0


def extract_json(text: str) -> str:
    """Identify json from a text blob by matching '[]' or '{}'.

    Warning: This will identify the first json structure!
    """
    # check for markdown indicator; if present, start there
    md_json_idx = text.find("```json")
    if md_json_idx != -1:
        text = text[md_json_idx:]

    # search for json delimiter pairs
    left_bracket_idx = text.find("[")
    left_brace_idx = text.find("{")

    indices = [idx for idx in (left_bracket_idx, left_brace_idx) if idx != -1]
    start_idx = min(indices) if indices else None

    # If no delimiter found, return the original text
    if start_idx is None:
        return text

    # Identify the exterior delimiters defining JSON
    open_char = text[start_idx]
    close_char = "]" if open_char == "[" else "}"

    # Initialize a count to keep track of delimiter pairs
    count = 0
    for i, char in enumerate(text[start_idx:], start=start_idx):
        if char == open_char:
            count += 1
        elif char == close_char:
            count -= 1

        # When count returns to zero, we've found a complete structure
        if count == 0:
            return text[start_idx : i + 1]

    return text  # In case of unbalanced JSON, return the original text


# def parse_date(date: t.Any) -> datetime:
#     """Parse unix timestamps, iso format, and human-readable strings."""
#     if date is None:
#         return None  # type: ignore

#     if isinstance(date, datetime):
#         if date.tzinfo is None:
#             return date.replace(tzinfo=timezone.utc)

#         if date.tzinfo.utcoffset(datetime.now()).seconds != 0:
#             raise ValueError("Refusing to load a non-UTC date!")
#         return date

#     if isinstance(date, (float, int)):
#         date = str(date)

#     if isinstance(date, str):
#         return dateparser(date, settings={"TIMEZONE": "UTC"}).astimezone(timezone.utc)

#     raise ValueError("Tried to parse invalid date! {}".format(date))
