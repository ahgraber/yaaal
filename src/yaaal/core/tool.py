"""Functions and helpers for tool use.

ref: https://github.com/openai/openai-agents-python/blob/8d906f88f02d30b3cf6068e5de88a5f1e4bafd82/src/agents/function_schema.py
"""

from __future__ import annotations

import contextlib
from functools import wraps
import inspect
import json
import logging
import re
from typing import Any, Callable, Generic, Literal, Type, TypeVar, Union, cast, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, create_model
from typing_extensions import override, runtime_checkable

from .base import CallableReturnType, CallableWithSchema
from .exceptions import ValidationError
from ..types_.base import JSON
from ..types_.core import FunctionSchema, ToolResultMessage
from ..types_.utils import coerce_to_type, get_union_args, is_instance_of_type, is_type_annotation, origin_is_union

logger = logging.getLogger(__name__)


DocstringStyle = Literal["google", "numpy", "sphinx"]


def _detect_docstring_style(doc: str) -> DocstringStyle:
    """Detect the style of a docstring.

    As of Feb 2025, the automatic style detection in griffe is an Insiders feature. This code approximates it.

    Ref: https://github.com/openai/openai-agents-python/blob/8d906f88f02d30b3cf6068e5de88a5f1e4bafd82/src/agents/function_schema.py#L87-L129
    """
    scores: dict[DocstringStyle, int] = {"sphinx": 0, "numpy": 0, "google": 0}

    # Sphinx style detection: look for :param, :type, :return:, and :rtype:
    sphinx_patterns = [r"^:param\s", r"^:type\s", r"^:return:", r"^:rtype:"]
    for pattern in sphinx_patterns:
        if re.search(pattern, doc, re.MULTILINE):
            scores["sphinx"] += 1

    # Numpy style detection: look for headers like 'Parameters', 'Returns', or 'Yields' followed by
    # a dashed underline
    numpy_patterns = [
        r"^Parameters\s*\n\s*-{3,}",
        r"^Returns\s*\n\s*-{3,}",
        r"^Yields\s*\n\s*-{3,}",
    ]
    for pattern in numpy_patterns:
        if re.search(pattern, doc, re.MULTILINE):
            scores["numpy"] += 1

    # Google style detection: look for section headers with a trailing colon
    google_patterns = [r"^(Args|Arguments):", r"^(Returns):", r"^(Raises):"]
    for pattern in google_patterns:
        if re.search(pattern, doc, re.MULTILINE):
            scores["google"] += 1

    max_score = max(scores.values())
    if max_score == 0:
        return "google"

    # Priority order: sphinx > numpy > google in case of tie
    styles: list[DocstringStyle] = ["sphinx", "numpy", "google"]

    for style in styles:
        if scores[style] == max_score:
            return style

    return "google"


@contextlib.contextmanager
def _suppress_griffe_logging():
    """Suppresses warnings about missing annotations for params.

    Ref: https://github.com/openai/openai-agents-python/blob/8d906f88f02d30b3cf6068e5de88a5f1e4bafd82/src/agents/function_schema.py#L132-L141
    """
    logger = logging.getLogger("griffe")
    previous_level = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


def extract_function_description(fn: Callable) -> str | None:
    """Extract the description from a function's docstring."""
    from griffe import Docstring, DocstringSectionKind

    doc = inspect.getdoc(fn)
    if not doc:
        return None

    with _suppress_griffe_logging():
        docstring = Docstring(doc, lineno=1, parser=_detect_docstring_style(doc))
        parsed = docstring.parse()

    description: str | None = next(
        (section.value for section in parsed if section.kind == DocstringSectionKind.text), None
    )

    return description


def extract_param_descriptions(fn: Callable) -> dict[str, Any]:
    """Extract the parameter descriptions from a function's docstring."""
    from griffe import Docstring, DocstringSectionKind

    doc = inspect.getdoc(fn)
    if not doc:
        return {}

    with _suppress_griffe_logging():
        docstring = Docstring(doc, lineno=1, parser=_detect_docstring_style(doc))
        parsed = docstring.parse()

    param_descriptions = {
        param.name: param.description
        for section in parsed
        if section.kind == DocstringSectionKind.parameters
        for param in section.value
    }
    return param_descriptions


def pydantic_to_schema(model: Type[BaseModel], strict: bool = True) -> dict[str, Any]:
    """Convert a Pydantic model to an OpenAPI-compatible JSON schema.

    Args:
        model: Pydantic model to convert.
        strict: If True, use strict JSON schema conversion.

    Returns
    -------
        OpenAPI-compatible JSON schema.
    """
    if strict:
        from openai.lib._pydantic import to_strict_json_schema

        return to_strict_json_schema(model)
    else:
        return model.model_json_schema()


def function_schema(fn: Callable) -> FunctionSchema:
    """Given a python function, generate a FunctionSchema.

    Extracts type hints and default values (ignoring 'self' and 'cls')
    to create a model for validating input parameters.
    Requires type hints and docstrings for accurate schema.
    Ref:
      - https://medium.com/@wangxj03/schema-generation-for-llm-function-calling-5ab29cecbd49
      - https://github.com/pydantic/pydantic-ai/blob/b8d71369d5d7ab1b6b08fe020bfaf67cd6259ba4/pydantic_ai_slim/pydantic_ai/_pydantic.py#L41-L170
      - https://github.com/openai/openai-agents-python/blob/8d906f88f02d30b3cf6068e5de88a5f1e4bafd82/src/agents/function_schema.py#L186-L344
    """
    if inspect.getdoc(fn) is None:
        logger.warning(f"Function {fn.__name__} requires docstrings for viable signature.")
        description = ""
    else:
        description = extract_function_description(fn)

    # Handle bound methods by getting the original function
    if inspect.ismethod(fn):
        fn = fn.__func__

    # Get the signature of the function
    sig = inspect.signature(fn)
    type_hints = get_type_hints(fn)
    params = list(sig.parameters.items())
    param_descs = extract_param_descriptions(fn)

    fields = {}
    for param_name, param in params:
        if param_name in ("self", "cls"):
            continue

        # Get the type hint
        param_annotation = type_hints.get(param_name, param.annotation)
        if param_annotation == inspect._empty:
            # logger.warning(f"No type annotation provided for '{param_name}', using 'Any'")
            param_annotation = Any

        # If a docstring param description exists, use it
        field_description = param_descs.get(param_name, None)

        # Handle *args and **kwargs
        if param.kind == param.VAR_POSITIONAL:
            # e.g. *args: extend positional args
            if get_origin(param_annotation) is tuple:
                # e.g. def foo(*args: tuple[int, ...]) -> treat as List[int]
                args_of_tuple = get_args(param_annotation)
                if len(args_of_tuple) == 2 and args_of_tuple[1] is Ellipsis:
                    param_annotation = list[args_of_tuple[0]]  # type: ignore
                else:
                    param_annotation = list[Any]
            else:
                # If user wrote *args: int, treat as List[int]
                param_annotation = list[param_annotation]  # type: ignore

            # Default factory to empty list
            fields[param_name] = (
                param_annotation,
                Field(default_factory=list, description=field_description),  # type: ignore
            )

        elif param.kind == param.VAR_KEYWORD:
            # **kwargs handling
            if get_origin(param_annotation) is dict:
                # e.g. def foo(**kwargs: dict[str, int])
                dict_args = get_args(param_annotation)
                if len(dict_args) == 2:  # NOQA: SIM108
                    param_annotation = dict[dict_args[0], dict_args[1]]  # type: ignore
                else:
                    param_annotation = dict[str, Any]
            else:
                # e.g. def foo(**kwargs: int) -> Dict[str, int]
                param_annotation = dict[str, param_annotation]  # type: ignore

            fields[param_name] = (
                param_annotation,
                Field(default_factory=dict, description=field_description),  # type: ignore
            )

        else:
            # Normal parameter
            if param.default == inspect._empty:
                # Required field
                fields[param_name] = (
                    param_annotation,
                    Field(..., description=field_description),
                )
            else:
                # Parameter with a default value
                fields[param_name] = (
                    param_annotation,
                    Field(default=param.default, description=field_description),
                )

    model = create_model(
        fn.__name__,
        __doc__=description,
        __base__=BaseModel,
        **fields,
    )

    return FunctionSchema(
        pydantic_model=model,
        json_schema=pydantic_to_schema(model),  # enforces 'strict'
        signature=sig,
    )


def anthropic_pydantic_function_tool(model: Type[BaseModel]) -> dict:
    """Convert a Pydantic model into an Anthropic-compatible tool specification."""
    from openai import pydantic_function_tool

    schema = pydantic_function_tool(model)
    function = schema["function"]

    return {
        "name": function["name"],
        "description": function["description"],
        "input_schema": {
            "type": "object",
            "properties": function["parameters"]["properties"],
            "required": function["parameters"].get("required", []),
        },
    }


class Tool(Generic[CallableReturnType], CallableWithSchema[CallableReturnType]):
    """Wrap a callable to provide a validated function interface.

    This protocol defines a callable object that exposes metadata about its
    input/output structure through schemas and type information. It ensures
    that function calls are properly validated using Pydantic models and
    JSON schemas.
    """

    function_schema: FunctionSchema
    returns: Type[CallableReturnType]

    def __init__(
        self,
        func: Callable[..., CallableReturnType],
        returns: Type[CallableReturnType] | None = None,
    ) -> None:
        self._func = func
        self.function_schema = function_schema(func)
        self.returns = returns or get_type_hints(func).get("return", Any)

        self.__name__ = self.function_schema.pydantic_model.__name__
        self.__doc__ = self.function_schema.pydantic_model.__doc__

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> CallableReturnType:
        """Execute function and validate result."""
        # _args, _kwargs = self.function_schema.to_call_args(kwargs)
        result = self._func(*args, **kwargs)
        return self.validate_return_type(result)

    def validate_return_type(self, value: Any) -> CallableReturnType:
        """Validate (and coerce) to specified return type.

        Checks if the value matches the type specified in 'returns' and attempts to convert it for basic types
        or validate it for Pydantic models.


        Parameters
        ----------
        value : Any
            The value to validate/coerce
        returns : Any
            The target type to coerce to

        Returns
        -------
        Any
            The coerced value matching the target type

        Raises
        ------
        TypeError
            If coercion is not possible
        """
        # Early return if no return type specified
        if self.returns is None or self.returns is Any:
            return value
        if not is_type_annotation(self.returns):
            raise TypeError(f"Invalid return type annotation: {self.returns}")

        # If value already matches the type, return it
        if is_instance_of_type(value, self.returns):
            return value

        # Leverage coerce_to_type for conversion
        try:
            return coerce_to_type(value, self.returns)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Could not coerce {value!r} (of type {type(value).__name__}) to {self.returns}: {e}"
            ) from e

    @staticmethod
    def respond_as_tool(tool_call_id: str, response: str | JSON) -> ToolResultMessage:
        """Convert a response into a ToolResultMessage for tool output.

        Serializes the response into a JSON string if needed, ensuring it can be consumed as a tool result.
        """
        if tool_call_id is None:
            raise ValueError("tool_call_id is required")

        if isinstance(response, str):
            responsestr = response
        elif isinstance(response, BaseModel):
            responsestr = response.model_dump_json()
        else:
            try:
                responsestr = json.dumps(response)
            except Exception as e:
                logger.debug(f"Could not serialize result as json string: {e}")
                responsestr = str(response)

        return ToolResultMessage(tool_call_id=tool_call_id, content=responsestr)


def tool(
    func: Callable[..., CallableReturnType] | None = None,
    *,
    returns: Type[Any] | None = None,
) -> Callable[..., Tool[CallableReturnType]]:
    """Decorate a function into a Tool instance for type validation.

    Can be used either as a bare decorator (@tool) or with parameters (@tool(returns=type)).

    Parameters
    ----------
    func : Callable[..., Any] | None
        Function to wrap when used as @tool
    returns : Type[Any] | None
        Optional return type annotation when used as @tool(returns=type)

    Returns
    -------
    Tool | Callable[[Callable[..., Any]], Tool]
        Either a Tool instance or a decorator function

    Examples
    --------
    Basic usage as decorator:
    >>> @tool
    ... def add(x: int, y: int) -> int:
    ...     return x + y
    >>> add(1, 2)
    3

    With explicit return type:
    >>> from pydantic import BaseModel
    >>> class User(BaseModel):
    ...     name: str
    ...     age: int
    >>> @tool(returns=User)
    ... def create_user(name: str, age: int):
    ...     return {"name": name, "age": age}
    >>> user = create_user("Alice", 30)
    >>> isinstance(user, User)
    True

    Auto-converting return values:
    >>> @tool(returns=str)
    ... def add(x: int, y: int) -> int:
    ...     return x + y
    >>> add(40, 2)
    '42' # int converts to str
    """

    def decorator(f: Callable[..., Any]) -> Tool:
        return Tool(f, returns=returns)

    if func is not None:
        return decorator(func)
    return decorator
