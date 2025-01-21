from functools import wraps
import inspect
import json
import logging
import os
from typing import Annotated, Any, Callable, Type

from pydantic import BaseModel, create_model

import requests

from ._types import JSON, ToolMessage, URLContent

logger = logging.getLogger(__name__)


def function_schema_model(fn: Callable) -> Type[BaseModel]:
    """Generate a Pydantic BaseModel that represents a function's signature.

    Requires type hints and docstrings for accurate schema.
    Ref:
      - https://medium.com/@wangxj03/schema-generation-for-llm-function-calling-5ab29cecbd49
      - https://github.com/openai/swarm/blob/7ae87dee4594f1dbbdc9c1260fd608cc3c149345/swarm/util.py#L31-L87
      - https://github.com/pydantic/pydantic-ai/blob/b8d71369d5d7ab1b6b08fe020bfaf67cd6259ba4/pydantic_ai_slim/pydantic_ai/_pydantic.py#L41-L170
    """
    # Handle bound methods by getting the original function
    if inspect.ismethod(fn):
        fn = fn.__func__

    # Get the signature of the function
    sig = inspect.signature(fn)

    parameters = {}
    for param_name, param in sig.parameters.items():
        # Skip self and cls parameters for methods
        if param_name in ("self", "cls"):
            continue

        # Get the type hint
        if param.annotation != inspect.Parameter.empty:
            param_annotation = param.annotation
        else:
            # logger.info("No type annotatio provided, using 'Any'")
            # param_annotation = Any
            raise TypeError(f"No type annotation provided for param '{param.name}'")

        # Set up the field with or without a default value
        if param.default != inspect.Parameter.empty:
            parameters[param_name] = (param_annotation, param.default)
        else:
            parameters[param_name] = (param_annotation, ...)

        # Handle *args and **kwargs
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            parameters[f"{param_name}_list"] = (list[param_annotation], [])
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            parameters[f"{param_name}_dict"] = (dict[str, param_annotation], {})

    model = create_model(
        fn.__name__,
        __doc__=fn.__doc__,
        **parameters,
        __config__={"arbitrary_types_allowed": True},
    )

    return model


def respond_as_tool(tool_call_id: str, response: str | JSON) -> ToolMessage:
    """Return response as a ToolMessage."""
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

    return ToolMessage(tool_call_id=tool_call_id, content=responsestr)


class CallableWithSignature:
    def __init__(self, f):
        self.func = f
        wraps(f)(self)

    def __call__(self, *args, **kwargs):  # NOQA: D102
        return self.func(*args, **kwargs)

    def signature(self) -> Type[BaseModel]:
        """Return the signature of the wrapped function."""
        return function_schema_model(self.func)

    def tool_response(self, *args, tool_call_id: str, **kwargs) -> ToolMessage:
        """Call the function with tool_call_id (required) and arguments."""
        if tool_call_id is None:
            raise ValueError("tool_call_id is required")

        return respond_as_tool(
            tool_call_id=tool_call_id,
            response=self.func(*args, **kwargs),
        )


def tool(func: Callable) -> CallableWithSignature:
    """Convert Callable to CallableWithSignature as decorator."""
    return CallableWithSignature(func)
