from __future__ import annotations

import logging
from typing import Annotated, Any, Literal, Union

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
    role: Literal["assistant", "system", "tool", "user"] = Field(
        description="The role of the message author.", min_length=1
    )
    content: str = Field(description="The contents of the message.", min_length=1)


class ToolMessage(Message):
    role: Literal["tool"] = "tool"
    content: str = Field(description="The result of the tool call.")
    tool_call_id: str = Field(description="The tool_call.id that requested this response")


class Conversation(BaseModel, extra="forbid"):
    messages: list[Message] = Field(description="The messages of the conversation.", min_length=1)


# ------------------------------------------------------------------
# OpenAI compatibility
class ChatCompletionResponse(BaseModel):
    choices: list[ChatCompletionChoice]


class ChatCompletionChoice(BaseModel):
    finish_reason: Literal["stop", "tool_calls"] | None = None
    message: ChatCompletionMessage


class ChatCompletionMessage(Message):
    # role: Literal["assistant", "system", "tool", "user"]
    content: str | None
    tool_calls: list[ChatCompletionMessageToolCall] | None
    refusal: str | None


class ChatCompletionMessageToolCall(BaseModel):
    id: str
    function: ChatCompletionToolCallFunction
    type: Literal["function"] = "function"


class ChatCompletionToolCallFunction(BaseModel):
    name: str
    arguments: str
