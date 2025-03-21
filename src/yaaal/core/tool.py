from __future__ import annotations

from functools import wraps
import inspect
import json
import logging
from typing import Any, Callable, Generic, Type, TypeVar, cast, get_args, get_origin, get_type_hints

from pydantic import BaseModel, create_model
from typing_extensions import override, runtime_checkable

from .base import CallableReturnType, CallableWithSignature
from .exceptions import ValidationError
from ..types_.base import JSON
from ..types_.core import ToolResultMessage
from ..types_.utils import get_union_args, is_type_annotation

logger = logging.getLogger(__name__)


def pydantic_function_signature(fn: Callable) -> Type[BaseModel]:
    """Generate a Pydantic model representing a function's signature.

    Extracts type hints and default values (ignoring 'self' and 'cls')
    to create a model for validating input parameters.
    Requires type hints and docstrings for accurate schema.
    Ref:
      - https://medium.com/@wangxj03/schema-generation-for-llm-function-calling-5ab29cecbd49
      - https://github.com/openai/swarm/blob/7ae87dee4594f1dbbdc9c1260fd608cc3c149345/swarm/util.py#L31-L87
      - https://github.com/pydantic/pydantic-ai/blob/b8d71369d5d7ab1b6b08fe020bfaf67cd6259ba4/pydantic_ai_slim/pydantic_ai/_pydantic.py#L41-L170
    """
    if fn.__doc__ is None:
        logger.warning(f"Function {fn.__name__} requires docstrings for viable signature.")
        doc = ""
        # raise ValueError("Function requires docstrings for viable signature.")
    else:
        doc = fn.__doc__

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
            logger.warning(f"No type annotation provided for '{param_name}', using 'Any'")
            param_annotation = Any
            # raise TypeError(f"No type annotation provided for param '{param.name}'")

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

    return create_model(
        fn.__name__,
        __doc__=doc,
        **parameters,
        __config__={"arbitrary_types_allowed": True},
    )


class Tool(Generic[CallableReturnType], CallableWithSignature[CallableReturnType]):
    """
    Wrap a callable to provide type validation and generate an OpenAPI-compatible JSON schema.

    Attributes
    ----------
    signature : Type[BaseModel]
        Pydantic model for validating input parameters.
    schema : dict
        OpenAPI-compatible JSON schema of the parameters.
    returns : Type[CallableReturnType] | None
        Expected return type annotation.
    """

    def __init__(
        self, func: Callable[..., CallableReturnType], returns: Type[CallableReturnType] | None = None
    ) -> None:
        self._func = func
        self.returns: Type[CallableReturnType] | None = returns or get_type_hints(func).get("return")
        self.signature = pydantic_function_signature(func)
        self.schema = self.signature.model_json_schema()

        # treat Tool instance as func
        self.__name__ = self.signature.__name__
        self.__doc__ = self.signature.__doc__

    def __call__(self, *args: Any, **kwargs: Any) -> CallableReturnType:
        """Invoke the wrapped function and validate its return type."""
        result = self._func(*args, **kwargs)
        return self.validate_return_type(result)

    def validate_return_type(self, value: Any) -> CallableReturnType:
        """Validate (and coerce) to specified return type.

        Checks if the value matches the type specified in 'returns' and attempts to convert it for basic types
        or validate it for Pydantic models.
        """
        if self.returns in (None, Any):
            return value

        if not is_type_annotation(self.returns):
            raise TypeError(f"Return type hint {self.returns} is not a valid type")

        return_types = get_union_args(self.returns)
        for rt in return_types:
            effective_type = get_origin(rt) or rt
            if isinstance(value, effective_type):
                return value
            if effective_type in (str, int, float, bool):
                # Coerce basic types.
                return cast(CallableReturnType, rt(value))
            elif issubclass(rt, BaseModel):
                # Validate / coerce models.
                return cast(CallableReturnType, rt.model_validate(value))

        raise ValidationError(
            f"Return value {value} of type {type(value)} does not match expected type {return_types}"
        )

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
    func: Callable[..., CallableReturnType] | None = None, *, returns: Type[Any] | None = None
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


def anthropic_pydantic_function_tool(model: Type[BaseModel]) -> dict:
    """Convert a Pydantic model into an Anthropic-compatible tool specification.

    Formats the tool schema using the Anthropic interface to enable function calling.
    """
    import openai

    schema = openai.pydantic_function_tool(model)
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
