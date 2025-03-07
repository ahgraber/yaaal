from __future__ import annotations as _annotations

import collections.abc
from datetime import datetime, timezone
from functools import lru_cache
import inspect
import logging
import sys
import types
from typing import (
    Annotated,
    Any,
    Dict as TypingDict,  # Add this import
    FrozenSet as TypingFrozenSet,
    List as TypingList,
    Optional,
    Set as TypingSet,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic_core import PydanticUndefined
import typing_extensions

logger = logging.getLogger(__name__)


class TypeMergeError(TypeError):
    """Raised when types cannot be merged."""

    pass


def is_type_annotation(annotation) -> bool:
    """Determine whether annotation is valid type annotation."""
    origin = get_origin(annotation)

    # Check if it's a generic type like list[int]
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
    """Extract the arguments of a Union type if `tp` is a union, otherwise return the original type."""
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


@lru_cache(maxsize=128)
def _resolve_type_conflict(type1: Any, type2: Any) -> Any:
    """
    Resolve type conflicts between two annotations.

    Args:
        type1: First type annotation
        type2: Second type annotation

    Returns
    -------
        Resolved type annotation
    """
    # Handle identical or Any types
    if type1 == type2:
        return type1
    if type1 is Any or type2 is Any:
        return Any

    # Handle Optional types (Union with None)
    args1 = get_union_args(type1)
    args2 = get_union_args(type2)

    type1_is_optional = type(None) in args1
    type2_is_optional = type(None) in args2

    # Remove None from args for inner type resolution
    inner_args1 = tuple(arg for arg in args1 if arg is not type(None))
    inner_args2 = tuple(arg for arg in args2 if arg is not type(None))

    # Resolve inner types
    if len(inner_args1) == 1 and len(inner_args2) == 1:
        resolved_inner = _resolve_inner_types(inner_args1[0], inner_args2[0])
    else:
        # Handle multi-type unions
        resolved_inner = _resolve_union_types_direct(inner_args1, inner_args2)

    # Add None back if it was in either type
    return resolved_inner | None if (type1_is_optional or type2_is_optional) else resolved_inner


def _resolve_inner_types(type1: Any, type2: Any) -> Any:
    """Resolve non-Optional type conflicts."""
    # Handle basic types
    if _is_subclass_safe(type1, type2):
        return type2
    if _is_subclass_safe(type2, type1):
        return type1

    # Handle numeric types
    if {type1, type2}.issubset({int, float}):
        return float

    # Handle collection types
    origin1, origin2 = get_origin(type1), get_origin(type2)
    if origin1 is not None and origin1 == origin2:
        return _resolve_collection_types(type1, type2)

    # Handle union types
    if origin_is_union(origin1) or origin_is_union(origin2):
        return _resolve_union_types(type1, type2)

    # Instead of defaulting to Union, raise error for incompatible types
    raise TypeMergeError(f"Cannot merge incompatible types: {type1} and {type2}")


def _is_subclass_safe(type1: Any, type2: Any) -> bool:
    """Safely check subclass relationship."""
    try:
        return isinstance(type1, type) and isinstance(type2, type) and issubclass(type1, type2)
    except TypeError:
        return False


def _resolve_collection_types(type1: Any, type2: Any) -> Any:
    """Resolve collection type conflicts more robustly."""
    origin = get_origin(type1) or type1  # Handle both generic and non-generic types
    args1, args2 = get_args(type1), get_args(type2)

    # Verify collections are of the same type
    origin2 = get_origin(type2) or type2
    if origin is not origin2:
        raise TypeMergeError(f"Cannot merge different collection types: {origin} and {origin2}")

    # Special handling for tuple types
    if origin is tuple:
        # Handle empty tuples
        if not args1 or not args2:
            return tuple

        # Handle variable-length tuples (Tuple[T, ...])
        if args1[-1] is Ellipsis or args2[-1] is Ellipsis:
            # If both are variable-length, resolve element types
            if args1[-1] is Ellipsis and args2[-1] is Ellipsis:
                elem_type = _resolve_type_conflict(args1[0], args2[0])
                return tuple[elem_type, ...]
            # One is variable, one is fixed: prefer variable
            return args1[-1] is Ellipsis and type1 or type2

        # Both are fixed length but different sizes
        if len(args1) != len(args2):
            # Create a unified tuple with Any elements for all positions
            return tuple[Any, ...]

        # Same length tuples: resolve each position
        resolved_args = tuple(_resolve_type_conflict(a1, a2) for a1, a2 in zip(args1, args2))
        return tuple[*resolved_args]

    # Handle generic collections (list, set, frozenset, deque, etc.)
    if origin in (list, TypingList, set, TypingSet, frozenset, TypingFrozenSet, collections.deque):
        # For built-in collections, make sure we get the args
        if not args1 and hasattr(type1, "__origin__"):
            args1 = getattr(type1, "__args__", (Any,))
        if not args2 and hasattr(type2, "__origin__"):
            args2 = getattr(type2, "__args__", (Any,))

        # Default to Any if still no args
        args1 = args1 or (Any,)
        args2 = args2 or (Any,)

        elem_type = _resolve_type_conflict(args1[0], args2[0])
        return origin[elem_type]

    # Handle mapping types (dict, TypingDict) or (
    if origin in (dict, TypingDict) or (hasattr(origin, "__mro__") and collections.abc.Mapping in origin.__mro__):
        # For built-in dict, make sure we get the args
        if not args1 and hasattr(type1, "__origin__"):
            args1 = getattr(type1, "__args__", (Any, Any))
        if not args2 and hasattr(type2, "__origin__"):
            args2 = getattr(type2, "__args__", (Any, Any))

        # Default to Any if still no args
        args1 = args1 or (Any, Any)
        args2 = args2 or (Any, Any)

        key_type = _resolve_type_conflict(args1[0], args2[0])
        val_type = _resolve_type_conflict(args1[1], args2[1])
        return origin[key_type, val_type]

    # Instead of defaulting to Any, raise error for unknown collection types
    raise TypeMergeError(f"Unsupported collection type: {origin}")


def _resolve_union_types_direct(args1: tuple[Any, ...], args2: tuple[Any, ...]) -> Any:
    """Resolve union type conflicts with direct args."""
    # Combine all unique args
    combined_args = set(args1) | set(args2)

    # Filter out redundant types (those that are subclasses of others)
    filtered_args = set()
    for arg in combined_args:
        if not any(_is_subclass_safe(arg, other) for other in combined_args if other is not arg):
            filtered_args.add(arg)

    # Sort for consistency
    union_args = sorted(filtered_args, key=lambda t: str(t))

    # Return plain type if only one left
    if len(union_args) == 1:
        return union_args[0]

    return Union[*union_args]


def _resolve_union_types(type1: Any, type2: Any) -> Any:
    """Resolve union type conflicts."""
    args1 = get_union_args(type1)
    args2 = get_union_args(type2)
    return _resolve_union_types_direct(args1, args2)


# Type variable for BaseModel subclasses
BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


def merge_models(
    *models: Type[BaseModelT],
    name: str | None = None,
    config: ConfigDict | None = None,
    description: str | None = None,
) -> Type[BaseModel]:
    """
    Merge multiple Pydantic models into a single model.

    Parameters
    ----------
    *models : Type[BaseModel]
        Variable number of Pydantic models to be merged.
    name : str | None, optional
        Name for the merged model. If None, a default name will be generated.
    config : ConfigDict | None, optional
        Pydantic ConfigDict for the merged model. If None, default config will be used.
    description : str | None, optional
        Description for the merged model. If None, no description will be added.

    Returns
    -------
    Type[BaseModel]
        A new Pydantic model class that combines all fields and validators from the input models.

    Notes
    -----
    Field merging rules:
    - Type conflicts resolved by finding a common supertype
    - A field is required if required in any source model
    - Default values preserved from the first non-None value found
    - Field descriptions preserved from the first model where they appear
    - Validators are NOT merged, only fields

    Examples
    --------
    >>> class ModelA(BaseModel):
    ...     field1: str
    ...     field2: int = 42
    >>> class ModelB(BaseModel):
    ...     field2: int = 10
    ...     field3: float
    >>> MergedModel = merge_models(ModelA, ModelB)
    >>> merged_instance = MergedModel(field1="test", field2=15, field3=3.14)
    """
    if not models:
        raise ValueError("At least one model must be provided")

    # Default name if not provided
    if name is None:
        name = f"Merged{''.join(model.__name__ for model in models)}"

    # Collect fields from all models
    merge_fields: dict[str, dict[str, Any]] = {}

    for model in models:
        # Collect and merge fields (varnames, type annotations, defaults, and required status)
        for field_name, field_info in model.model_fields.items():
            if field_name in merge_fields:
                # Field name exists
                merged_field = merge_fields[field_name]

                # Resolve type conflicts
                merged_field["annotation"] = _resolve_type_conflict(merged_field["annotation"], field_info.annotation)

                # Field is required if required in any model
                merged_field["required"] = merged_field["required"] or field_info.is_required()

                # Keep first non-None default value
                if merged_field["default"] is None and (
                    field_info.default is not None and field_info.default is not PydanticUndefined
                ):
                    merged_field["default"] = field_info.default

                # Keep first non-None description
                if merged_field["description"] is None and isinstance(field_info.description, str):
                    merged_field["description"] = field_info.description

            else:  # New field
                merge_fields[field_name] = {
                    "annotation": field_info.annotation,
                    "required": field_info.is_required(),
                    "default": field_info.default if field_info.default is not PydanticUndefined else None,
                    "description": field_info.description if isinstance(field_info.description, str) else None,
                }

    # Create field definitions for the merged model
    field_definitions = {}
    for field_name, field_info in merge_fields.items():
        # Build field kwargs dict excluding None values
        field_kwargs = {"default": field_info["default"], "description": field_info["description"]}
        field_kwargs = {k: v for k, v in field_kwargs.items() if v is not None}

        field_definitions[field_name] = (field_info["annotation"], Field(**field_kwargs))

    # Create the merged model
    return create_model(
        name,
        __config__=config,
        __doc__=description,
        **field_definitions,
    )
