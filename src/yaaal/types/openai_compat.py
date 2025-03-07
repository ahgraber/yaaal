from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel

from aisuite.framework import ChatCompletionResponse as AISuiteChatCompletion
from openai.types.chat import ChatCompletion as OpenAIChatCompletion

from .core import Message

logger = logging.getLogger(__name__)


# OpenAI compatibility
class ChatCompletionMessageToolCallFunction(BaseModel, extra="ignore"):
    name: str
    arguments: str


class ChatCompletionMessageToolCall(BaseModel, extra="ignore"):
    id: str
    function: ChatCompletionMessageToolCallFunction
    type: Literal["function"] = "function"


class ChatCompletionMessage(Message, extra="ignore"):
    # role: Literal["assistant", "system", "tool", "user"]
    content: str | None = None
    tool_calls: list[ChatCompletionMessageToolCall] | None = None
    refusal: str | None = None


class ChatCompletionChoice(BaseModel, extra="ignore"):
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"] | None = None
    message: ChatCompletionMessage


class ChatCompletion(BaseModel, extra="ignore"):
    id: int | str | None = None
    choices: list[ChatCompletionChoice]


def convert_response(response: OpenAIChatCompletion | AISuiteChatCompletion) -> ChatCompletion:
    """Unify aisuite response object types."""
    if isinstance(response, OpenAIChatCompletion):
        return ChatCompletion(**response.model_dump())
    else:
        choices = []
        for choice in response.choices:
            message = ChatCompletionMessage(**choice.message.model_dump())

            choices.append(
                ChatCompletionChoice(
                    message=message,
                    finish_reason=choice.finish_reason if hasattr(choice, "finish_reason") else None,
                )
            )

        completion_response = ChatCompletion(
            id=response.id if hasattr(response, "id") else None,
            choices=choices,
        )
        return completion_response
