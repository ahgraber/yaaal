from __future__ import annotations

from typing import Any, Literal, Self, TypeAlias, Union

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


class ConversationBuilder:
    def __init__(self):
        self.messages: list[Message] = []

    def add_system(self, content: str) -> Self:
        """Append a system message to the conversation.

        Parameters
        ----------
        content : str
            The content of the system message.

        Returns
        -------
        Self
            The builder instance for chaining.
        """
        self.messages.append(SystemMessage(content=content))
        return self

    def add_user(self, content: str) -> Self:
        """Append a user message to the conversation.

        Parameters
        ----------
        content : str
            The content of the user message.

        Returns
        -------
        Self
            The builder instance for chaining.
        """
        self.messages.append(UserMessage(content=content))
        return self

    def add_assistant(self, content: str) -> Self:
        """Append an assistant message to the conversation.

        Parameters
        ----------
        content : str
            The content of the assistant message.

        Returns
        -------
        Self
            The builder instance for chaining.
        """
        self.messages.append(AssistantMessage(content=content))
        return self

    def build(self) -> Conversation:
        """Build and return a Conversation from the added messages.

        Returns
        -------
        Conversation
            The built conversation containing all messages.
        """
        return Conversation(messages=self.messages)


class Conversation(BaseModel):
    messages: list[Message] = Field(description="The messages of the conversation.", min_length=1)

    def __repr__(self):
        """Return a JSON-formatted string representation of the conversation."""
        return format_json(self.model_dump())

    @classmethod
    def builder(cls) -> ConversationBuilder:
        """Obtain a ConversationBuilder for constructing a Conversation.

        Returns
        -------
        ConversationBuilder
            An instance of the builder for chaining message additions.

        Examples
        --------
        >>> conversation = (
        ...     Conversation.builder()
        ...     .add_system("System message")
        ...     .add_user("User message")
        ...     .add_assistant("Assistant message")
        ...     .build()
        ... )
        >>> print(conversation)
        Conversation(messages=[...])
        """
        return ConversationBuilder()
