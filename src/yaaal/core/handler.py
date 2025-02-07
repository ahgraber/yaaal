"""Components for composable LLM calls.

A Handler processes LLM responses.
Whereas `Prompts` validate _inputs_ to the template, `Handlers` validate the LLM responses.

This involves recognizing content vs tool response types, validating the response with the requisite Validator, and returning a Message object to continue the conversation.
If a tool-call instruction is detected, the Handler can try to `invoke` that call and return the function result as the response.

This module provides the following handlers:
- ResponseHandler: Handles and validates content-based responses.
- ToolHandler: Processes tool calls with validation and optional invocation.
- CompositeHandler: Combines functionality of both content and tool handlers.

The handlers work with ChatCompletion responses and implement the BaseHandler interface.
"""

from __future__ import annotations

import json
import logging
from typing import Generic

from pydantic import BaseModel
from typing_extensions import override, runtime_checkable

from .base import ContentHandlerReturnType, Handler, ToolHandlerReturnType, Validator, ValidatorReturnType
from .exceptions import ResponseError
from .tool import CallableWithSignature
from .validator import ToolValidator
from ..types.base import JSON
from ..types.core import AssistantMessage, Conversation, UserMessage
from ..types.openai_compat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall

logger = logging.getLogger(__name__)


class ResponseHandler(Generic[ContentHandlerReturnType], Handler[ContentHandlerReturnType, None]):
    """Handles content responses with validation."""

    def __init__(
        self,
        validator: Validator[ContentHandlerReturnType],
        max_repair_attempts: int = 2,
    ):
        self.validator = validator
        self.max_repair_attempts = max_repair_attempts

    @property
    def max_repair_attempts(self) -> int:
        """Client called for every execution of the Caller instance."""
        return self._max_repair_attempts

    @max_repair_attempts.setter
    def max_repair_attempts(self, max_repair_attempts: int):
        self._max_repair_attempts = max_repair_attempts

    @override
    def process(self, response: ChatCompletion) -> ContentHandlerReturnType:
        """Process the LLM response."""
        msg = response.choices[0].message
        if msg.content:
            validated = self.validator.validate(msg.content)
            # !!!NOTE!!!
            # pydantic_model.model_dump_json() != json.dumps(pydantic_model.model_dump())
            # return json.dumps(validated.model_dump()) if isinstance(validated, BaseModel) else validated
            return validated
        raise ValueError("Expected content response but received none")

    @override
    def repair(self, message: ChatCompletionMessage, error: str) -> Conversation | None:
        """Generate repair instructions for invalid response."""
        if not message.content:
            return None
        return self.validator.repair_instructions(message.content, error)


class ToolHandler(Handler[None, ToolHandlerReturnType]):
    """Handles tool responses with validation and optional invocation."""

    def __init__(
        self,
        validator: ToolValidator,
        auto_invoke: bool = False,
        max_repair_attempts: int = 2,
    ):
        self.validator = validator
        self.auto_invoke = auto_invoke
        self.max_repair_attempts = max_repair_attempts

    @property
    def max_repair_attempts(self) -> int:
        """Client called for every execution of the Caller instance."""
        return self._max_repair_attempts

    @max_repair_attempts.setter
    def max_repair_attempts(self, max_repair_attempts: int):
        self._max_repair_attempts = max_repair_attempts

    @override
    def process(self, response: ChatCompletion) -> BaseModel | str:
        """Process the LLM response."""
        msg = response.choices[0].message
        if not msg.tool_calls:
            raise ValueError("Expected tool call but received none")

        tool_call = msg.tool_calls[0]
        function = tool_call.function

        validated = self.validator.validate(tool_call)

        if not self.auto_invoke:
            return validated

        # Invoke tool
        logger.debug(f"Invoking {function.name}")
        tool = self.validator.toolbox[function.name]
        result = tool(**validated.model_dump())
        if isinstance(result, BaseModel):
            # !!!NOTE!!!
            # pydantic_model.model_dump_json() != json.dumps(pydantic_model.model_dump())
            return result
        elif isinstance(result, str):
            content = result
        else:
            try:
                content = json.dumps(result)
            except (TypeError, ValueError) as e:
                logger.warning(f"JSON serialization failed: {e}")
                content = str(result)

        return content

    @override
    def repair(self, message: ChatCompletionMessage, error: str) -> Conversation | None:
        """Generate repair instructions for invalid response."""
        if not message.tool_calls:
            return None
        return self.validator.repair_instructions(message.tool_calls[0], error)


class CompositeHandler(
    Generic[ContentHandlerReturnType, ToolHandlerReturnType], Handler[ContentHandlerReturnType, ToolHandlerReturnType]
):
    """Handles both content and tool responses."""

    max_repair_attempts: None  # managed by sub-handlers

    def __init__(
        self,
        content_handler: ResponseHandler[ContentHandlerReturnType],
        tool_handler: ToolHandler,
    ):
        self.content_handler = content_handler or None
        self.tool_handler = tool_handler or None

    @override
    def process(self, response: ChatCompletion) -> ContentHandlerReturnType | BaseModel | str:
        """Process the LLM response."""
        msg = response.choices[0].message
        if msg.content:
            return self.content_handler.process(response)
        elif msg.tool_calls:
            return self.tool_handler.process(response)
        elif msg.refusal:
            raise ResponseError(f"Received refusal {msg.refusal}")
        else:
            raise ResponseError("Could not identify message type")

    @override
    def repair(self, message: ChatCompletionMessage, error: str) -> Conversation | None:
        """Generate repair instructions for invalid response."""
        if message.content:
            return self.content_handler.repair(message, error)
        elif message.tool_calls:
            return self.tool_handler.repair(message, error)
        else:
            return None
