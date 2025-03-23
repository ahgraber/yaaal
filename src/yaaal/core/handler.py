"""Components for composable LLM calls.

Handlers process LLM responses by validating and optionally repairing them.
This module provides:
- ResponseHandler: For validating plaintext responses.
- ToolHandler: For processing tool calls with validation and optional function execution.
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
from .validator import PassthroughValidator, ToolValidator
from ..types_.base import JSON
from ..types_.core import AssistantMessage, Conversation, UserMessage
from ..types_.openai_compat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallFunction,
)

logger = logging.getLogger(__name__)


class ResponseHandler(Generic[ContentHandlerReturnType], Handler[ContentHandlerReturnType, None]):
    """Handles plain text responses from the LLM with validation.

    Uses a configured validator to process a text response and, if necessary, generate repair instructions.
    """

    def __init__(
        self,
        validator: Validator[ContentHandlerReturnType],
        max_repair_attempts: int = 2,
    ):
        self.validator = validator
        self.max_repair_attempts = max_repair_attempts

    @property
    def max_repair_attempts(self) -> int:
        """Return the maximum number of repair attempts allowed."""
        return self._max_repair_attempts

    @max_repair_attempts.setter
    def max_repair_attempts(self, max_repair_attempts: int):
        self._max_repair_attempts = max_repair_attempts

    @override
    def process(self, response: ChatCompletion) -> ContentHandlerReturnType:
        """Extract and validate the content message from the LLM response.

        Returns
        -------
            The validated response content.

        Raises
        ------
            ValueError: If no content is found in the response.
        """
        msg = response.choices[0].message
        if msg.content:
            validated = self.validator.validate(msg.content)

        # handle case where we use function-calling as proxy for structured response
        elif msg.tool_calls:
            if len(msg.tool_calls) > 1:
                logger.warning("Received multiple tool calls, only the first will be processed")
            tool_call = msg.tool_calls[0]

            validated = self.validator.validate(tool_call.function.arguments)

        else:
            raise ValueError("Expected content response but received none")

        # !!!NOTE!!!
        # pydantic_model.model_dump_json() != json.dumps(pydantic_model.model_dump())
        # return json.dumps(validated.model_dump()) if isinstance(validated, BaseModel) else validated
        return validated

    @override
    def repair(self, message: ChatCompletionMessage, error: str) -> Conversation | None:
        """Generate repair instructions for an invalid text response.

        Returns
        -------
            A Conversation with repair instructions or None if no content is available.
        """
        if message.content:
            return self.validator.repair_instructions(message.content, error)
        elif message.tool_calls:
            return self.validator.repair_instructions(message.tool_calls[0].function.arguments, error)
        else:
            return None


class ToolHandler(Generic[ToolHandlerReturnType], Handler[None, ToolHandlerReturnType]):
    """Handles tool call responses from the LLM with validation and optional execution.

    If auto_invoke is enabled, the tool is automatically invoked with validated arguments.
    """

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
        """Return the maximum number of repair attempts allowed for tool calls."""
        return self._max_repair_attempts

    @max_repair_attempts.setter
    def max_repair_attempts(self, max_repair_attempts: int):
        self._max_repair_attempts = max_repair_attempts

    @override
    def process(self, response: ChatCompletion) -> BaseModel | str:
        """Extract and validate a tool call from the LLM response.

        If auto_invoke is enabled, call the corresponding tool with validated arguments.
        Otherwise, simply return the validated tool arguments.

        Returns
        -------
            The valid tool response or the result from the tool invocation.

        Raises
        ------
            ValueError: If no tool call was found in the response.
        """
        msg = response.choices[0].message

        # NOTE: Anthropic may send content and tool_calls in same response
        # In these cases we only use the tool call
        if msg.tool_calls:
            if len(msg.tool_calls) > 1:
                logger.warning("Received multiple tool calls, only the first will be processed")
            tool_call = msg.tool_calls[0]
            function = tool_call.function

            validated = self.validator.validate(tool_call)

            if not self.auto_invoke:
                return validated
            else:
                return self._invoke(function, validated)

        elif msg.content:
            logger.warning("ToolHandler handled 'message.content' instead of tool call. This should not happen.")
            return msg.content

        else:
            raise ValueError("Response did not contain content or tool call")

    def _invoke(self, function: ChatCompletionMessageToolCallFunction, params: BaseModel) -> BaseModel | str:
        """Invoke a tool with validated parameters.

        Parameters
        ----------
        function : ChatCompletionMessageToolCallFunction
            The function to invoke.
        params : BaseModel
            The validated parameters to pass to the tool
        """
        fn = self.validator.toolbox[function.name]

        logger.debug(f"Invoking {function.name} with params: {params}")
        result = fn(**params.model_dump())
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
        """Generate repair instructions for an invalid tool call response.

        Returns
        -------
            A Conversation with instructions to correct the tool call, or None if no tool call is found.
        """
        if not message.tool_calls:
            return None
        return self.validator.repair_instructions(message.tool_calls[0], error)
