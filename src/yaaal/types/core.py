from __future__ import annotations

from typing import Literal, TypeAlias, Union

from pydantic import (
    BaseModel,
    Field,
)

from ..utilities import format_json

Role = Literal["assistant", "system", "tool", "user"]


class Message(BaseModel):
    role: Role = Field(description="The role of the message author.", min_length=1)
    content: str = Field(description="The contents of the message.", min_length=1)

    def __repr__(self):
        return format_json(self.model_dump())


# These messages are for composing Conversations (i.e., inputs to the LLM)
class SystemMessage(Message):
    role: Literal["system"] = "system"
    # content: str = Field(description="The contents of the message.", min_length=1)


class UserMessage(Message):
    role: Literal["user"] = "user"
    # content: str = Field(description="The contents of the message.", min_length=1)


class AssistantMessage(Message):
    role: Literal["assistant"] = "assistant"
    # content: str = Field(description="The contents of the message.", min_length=1)


class ToolResultMessage(Message):
    role: Literal["tool"] = "tool"
    content: str = Field(description="The result of the tool call.")
    tool_call_id: str = Field(description="The tool_call.id that requested this response")


class Conversation(BaseModel):
    messages: list[Message] = Field(description="The messages of the conversation.", min_length=1)

    def __repr__(self):
        return format_json(self.model_dump())


ValidatorResult: TypeAlias = Union[str, BaseModel]
APIHandlerResult: TypeAlias = Union[
    AssistantMessage,
    ToolResultMessage,
    # BaseModel,
]
