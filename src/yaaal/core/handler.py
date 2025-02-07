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

from pydantic import BaseModel

from .base import BaseHandler, BaseValidator, ResponseError, ValidationError
from .tools import CallableWithSignature
from .validator import ToolValidator
from ..types.base import JSON
from ..types.core import APIHandlerResult, AssistantMessage, Conversation, ToolResultMessage, ValidatorResult
from ..types.openai_compat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall

logger = logging.getLogger(__name__)


class ResponseHandler(BaseHandler):
    """Handles content responses with validation."""

    def __init__(
        self,
        validator: BaseValidator,
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

    def __call__(self, response: ChatCompletion) -> APIHandlerResult:
        """Process the LLM response."""
        msg = response.choices[0].message
        if msg.content:
            validated = self.validator.validate(msg.content)
            return AssistantMessage(
                # !!!NOTE!!!
                # pydantic_model.model_dump_json() != json.dumps(pydantic_model.model_dump())
                content=json.dumps(validated.model_dump()) if isinstance(validated, BaseModel) else validated
            )
        raise ValueError("Expected content response but received none")

    def repair(self, message: ChatCompletionMessage, error: str) -> Conversation | None:
        """Generate repair instructions for invalid response."""
        if not message.content:
            return None
        return self.validator.repair_instructions(message.content, error)


class ToolHandler(BaseHandler):
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

    def __call__(self, response: ChatCompletion) -> APIHandlerResult:
        """Process the LLM response."""
        msg = response.choices[0].message
        if not msg.tool_calls:
            raise ValueError("Expected tool call but received none")

        tool_call = msg.tool_calls[0]
        function = tool_call.function

        validated = self.validator.validate(tool_call)

        if not self.auto_invoke:
            # !!!NOTE!!!
            # pydantic_model.model_dump_json() != json.dumps(pydantic_model.model_dump())
            return AssistantMessage(content=json.dumps(validated.model_dump()))

        # Invoke tool
        logger.debug(f"Invoking {function.name}")
        tool = self.validator.toolbox[function.name]
        result = tool(**validated.model_dump())
        if isinstance(result, APIHandlerResult):
            content = result.content
        elif isinstance(result, BaseModel):
            content = result.model_dump_json()
        elif isinstance(result, str):
            content = result
        else:
            content = json.dumps(result)

        # NOTE: The handler returns a ToolResultMessage if invocation succeeds
        #       which may result in adding ToolResultMessage to the Conversation
        #       without the corresponding prior ChatCompletionToolCall
        # TODO: Is this a problem?
        return ToolResultMessage(tool_call_id=tool_call.id, content=content)

    def repair(self, message: ChatCompletionMessage, error: str) -> Conversation | None:
        """Generate repair instructions for invalid response."""
        if not message.tool_calls:
            return None
        return self.validator.repair_instructions(message.tool_calls[0], error)


class CompositeHandler(BaseHandler):
    """Handles both content and tool responses."""

    max_repair_attempts: None  # managed by sub-handlers

    def __init__(self, content_handler: ResponseHandler, tool_handler: ToolHandler):
        self.content_handler = content_handler or None
        self.tool_handler = tool_handler or None

    def __call__(self, response: ChatCompletion) -> APIHandlerResult:
        """Process the LLM response."""
        msg = response.choices[0].message
        if msg.content:
            return self.content_handler(response)
        elif msg.tool_calls:
            return self.tool_handler(response)
        elif msg.refusal:
            raise ResponseError(f"Received refusal {msg.refusal}")
        else:
            raise ResponseError("Could not identify message type")

    def repair(self, message: ChatCompletionMessage, error: str) -> Conversation | None:
        """Generate repair instructions for invalid response."""
        if message.content:
            return self.content_handler.repair(message, error)
        elif message.tool_calls:
            return self.tool_handler.repair(message, error)
        else:
            return None
