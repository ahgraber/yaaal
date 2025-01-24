from __future__ import annotations

import logging
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
)

from .core import Message

logger = logging.getLogger(__name__)


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
