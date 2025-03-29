from __future__ import annotations

import inspect
from typing import Any, Literal, Self, Type, TypeAlias, Union

from pydantic import BaseModel, ConfigDict, Field

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


class FunctionSchema:
    """Wraps a Pydantic model to capture the schema for a python function."""

    def __init__(
        self,
        pydantic_model: Type[BaseModel],
        json_schema: dict[str, Any],
        signature: inspect.Signature,
    ) -> None:
        """Initialize the function schema wrapper."""
        self.pydantic_model = pydantic_model
        self.json_schema = json_schema
        self.signature = signature

    def __call__(self, /, **data: Any):
        """Create a new model by parsing and validating input data from keyword arguments."""
        return self.pydantic_model(**data)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped Pydantic model."""
        return getattr(self.pydantic_model, name)

    # Explicitly implement key Pydantic model methods
    def model_validate(self, obj: Any) -> BaseModel:
        """Validate data against the wrapped Pydantic model."""
        return self.pydantic_model.model_validate(obj)

    def model_validate_json(self, json_data: str | bytes) -> BaseModel:
        """Validate JSON data against the wrapped Pydantic model."""
        return self.pydantic_model.model_validate_json(json_data)

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Serialize model to dict."""
        return self.pydantic_model.model_dump(*args, **kwargs)

    def model_dump_json(self, *args: Any, **kwargs: Any) -> str:
        """Serialize model to JSON string."""
        return self.pydantic_model.model_dump_json(*args, **kwargs)

    def to_call_args(self, obj: Any) -> tuple[list[Any], dict[str, Any]]:
        """Validate and convert input into (args, kwargs), suitable for calling the original function."""
        data = self.pydantic_model.model_validate(obj)

        positional_args: list[Any] = []
        keyword_args: dict[str, Any] = {}
        seen_var_positional = False

        for _idx, (name, param) in enumerate(self.signature.parameters.items()):
            if name in ("self", "cls"):
                continue

            value = getattr(data, name, None)
            if param.kind == param.VAR_POSITIONAL:
                positional_args.extend(value or [])
                seen_var_positional = True
            elif param.kind == param.VAR_KEYWORD:
                keyword_args.update(value or {})
            elif param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                if not seen_var_positional:
                    positional_args.append(value)
                else:
                    keyword_args[name] = value
            else:
                keyword_args[name] = value
        return positional_args, keyword_args
