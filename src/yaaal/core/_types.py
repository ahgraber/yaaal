from __future__ import annotations

from functools import wraps
import inspect
import logging
from typing import Annotated, Any, Union

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    WrapValidator,
)
from pydantic_core import PydanticCustomError
from typing_extensions import TypeAliasType  # TODO: import from typing when drop support for 3.11

logger = logging.getLogger(__name__)


def json_simple_error_validator(value: Any, handler: ValidatorFunctionWrapHandler, _info: ValidationInfo) -> Any:
    """Simplify the error message to avoid a gross error stemming from exhaustive checking of all union options."""
    try:
        return handler(value)
    except ValidationError as e:
        raise PydanticCustomError("invalid_json", "Input is not valid json") from e


JSONValue = Union[
    str,  # JSON string
    int,  # JSON number (integer)
    float,  # JSON number (float)
    bool,  # JSON boolean
    None,  # JSON null
]
JSON = TypeAliasType(
    "JSON",
    Annotated[
        Union[dict[str, "JSON"], list["JSON"], JSONValue],
        WrapValidator(json_simple_error_validator),
    ],
)
# JSONValidator = TypeAdapter[JSON]
# JSONModel = RootModel[JSON]


class Message(BaseModel, extra="forbid"):
    role: str = Field(description="The role of the message author.", min_length=1)
    content: str = Field(description="The contents of the message.", min_length=1)


class ToolMessage(Message):
    role: str = "tool"
    tool_call_id: str = Field(description="The tool_call.id that requested this response")
    content: str = Field(description="The result of the tool call.")


class Conversation(BaseModel, extra="forbid"):
    messages: list[Message] = Field(description="The messages of the conversation.", min_length=1)


class URLContent(BaseModel, extra="ignore"):
    """Text content from a webpage."""

    url: str = Field(description="The webpage url")
    title: str = Field(description="The page title")
    content: str = Field(description="The webpage's text content")
