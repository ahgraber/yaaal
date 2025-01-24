from __future__ import annotations

import logging
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
)

from ..utilities import format_json

logger = logging.getLogger(__name__)

Role = Literal["assistant", "system", "tool", "user"]


class Message(BaseModel):
    role: Role = Field(description="The role of the message author.", min_length=1)
    content: str = Field(description="The contents of the message.", min_length=1)

    def __repr__(self, **kwargs):
        return format_json(self.model_dump(), **kwargs)


class ToolMessage(Message):
    role: Literal["tool"] = "tool"
    content: str = Field(description="The result of the tool call.")
    tool_call_id: str = Field(description="The tool_call.id that requested this response")


class Conversation(BaseModel):
    messages: list[Message] = Field(description="The messages of the conversation.", min_length=1)

    def __repr__(self, **kwargs):
        return format_json(self.model_dump(), **kwargs)
