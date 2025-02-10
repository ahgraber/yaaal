from __future__ import annotations as _annotations

from datetime import datetime, timezone
import logging
import sys
import types
from typing import Annotated, Any, TypeVar, Union, get_args, get_origin

import typing_extensions

logger = logging.getLogger(__name__)


def is_type_annotation(annotation) -> bool:
    """Determine whether annotation is valid type annotation."""
    origin = get_origin(annotation)

    # Check if it's a generic type like List[int]
    if origin:
        # args_names = ", ".join(getattr(arg, "__name__", str(arg)) for arg in get_args(annotation))
        # logger.debug("Generic type: %s[%s]", getattr(origin, "__name__", str(origin)), args_names)
        return True

    # Check if it's a basic type annotation
    elif isinstance(annotation, type):
        # logger.debug(f"Basic type: {annotation.__name__}")
        return True

    return False


# same as `pydantic_ai_slim/pydantic_ai/_result.py:origin_is_union`
def origin_is_union(tp: type[Any] | None) -> bool:
    """Determine whether a given type parameter is a Union type."""
    return tp is Union or tp is types.UnionType


def get_union_args(tp: Any) -> tuple[Any, ...]:
    """Extract the arguments of a Union type if `response_type` is a union, otherwise return the original type."""
    # similar to `pydantic_ai_slim/pydantic_ai/_result.py:get_union_args`
    if isinstance(tp, typing_extensions.TypeAliasType):
        tp = tp.__value__

    origin = get_origin(tp)
    if origin_is_union(origin):
        return get_args(tp)
    else:
        return (tp,)


def unpack_annotated(tp: Any) -> tuple[Any, list[Any]]:
    """Strip `Annotated` from the type if present.

    Returns
    -------
        `(tp argument, ())` if not annotated, otherwise `(stripped type, annotations)`.
    """
    origin = get_origin(tp)
    if origin is Annotated or origin is typing_extensions.Annotated:
        inner_tp, *args = get_args(tp)
        return inner_tp, args
    else:
        return tp, []


def is_never(tp: Any) -> bool:
    """Check if a type is `Never`."""
    if tp is typing_extensions.Never:
        return True
    elif typing_never := getattr(typing_extensions, "Never", None):
        return tp is typing_never
    else:
        return False
