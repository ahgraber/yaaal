"""Components for composable LLM calls.

A Validator ensures the LLM response matches our expectations and provides retry/repair instructions in case the validation fails.

Each validator implements the BaseValidator interface with two main methods:
- validate(): Validates the completion/response
- repair_instructions(): Generates instructions for fixing validation failures

The module supports various validation strategies including:
- Passthrough validation (i.e., no validation)
- Schema-based validation using Pydantic models
- Pattern matching using regular expressions
- Tool call validation with function signatures
"""

from __future__ import annotations

import json
import logging
import textwrap
from typing import Pattern

import json_repair
from pydantic import BaseModel
from typing_extensions import override

from .base import BaseCaller, BaseValidator, ValidationError
from .tools import CallableWithSignature
from ..types.core import (
    AssistantMessage,
    Conversation,
    ResponseMessage,
    ToolResultMessage,
    UserMessage,
    ValidatorResult,
)
from ..types.openai_compat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallFunction,
    convert_response,
)

logger = logging.getLogger(__name__)


class PassthroughValidator(BaseValidator):
    """Simple validator that passes content through unchanged."""

    @override
    def validate(self, completion: str) -> str:
        return completion

    @override
    def repair_instructions(self, completion: str, error: str) -> None:
        return None


class PydanticValidator(BaseValidator):
    """Validate using Pydantic models."""

    def __init__(self, model: type[BaseModel]):
        self.model = model

    @override
    def validate(self, completion: str) -> BaseModel:
        return self.model.model_validate(json_repair.loads(completion))

    @override
    def repair_instructions(self, completion: str, error: str) -> Conversation:
        return Conversation(
            messages=[
                AssistantMessage(content=completion),
                UserMessage(
                    content=textwrap.dedent(
                        f"""
                Validation failed: {error}

                Please update your response to match this schema:
                {json.dumps(self.model.model_json_schema())}
                """.strip()
                    ),
                ),
            ]
        )


class RegexValidator(BaseValidator):
    """Validate using regex patterns."""

    def __init__(self, pattern: Pattern):
        self.pattern = pattern

    @override
    def validate(self, completion: str) -> str:
        match = self.pattern.search(completion)
        if not match:
            raise ValidationError("Response did not match expected pattern")
        return match.group()

    @override
    def repair_instructions(self, completion: str, error: str) -> Conversation:
        return Conversation(
            messages=[
                AssistantMessage(content=completion),
                UserMessage(
                    content=textwrap.dedent(
                        f"""
                    Response must match the following regex pattern: {self.pattern.pattern}

                    Update your response to ensure it is valid.
                    """.strip()
                    ),
                ),
            ]
        )


class ToolValidator(BaseValidator):
    """Validate tool calls."""

    def __init__(self, toolbox: list[BaseCaller | CallableWithSignature]):
        self.toolbox = toolbox

    @property
    def toolbox(self) -> dict[str, BaseCaller | CallableWithSignature]:
        """Available tools."""
        return self._toolbox

    @toolbox.setter
    def toolbox(self, toolbox: list[BaseCaller | CallableWithSignature]):
        tb = {}
        for tool in toolbox:
            if not isinstance(tool, (BaseCaller, CallableWithSignature)):
                raise TypeError(
                    f"Toolbox requires Caller or CallableWithSignature objects.  Received {tool}: {type(tool)}"
                )

            if not tool.signature().__doc__:
                logger.warning(
                    f"Did not find a 'description' for {tool.signature().__name__}.  Does the function need docstrings?"
                )
            try:
                tb[tool.signature().__name__] = tool
            except Exception:
                logger.exception(f"Error while defining toolbox entry for {str(tool)}")
                raise
        self._toolbox = tb

    @override
    def validate(self, completion: ChatCompletionMessageToolCall) -> BaseModel:
        """Validate tool call."""
        name = completion.function.name
        arguments = completion.function.arguments
        try:
            return self.toolbox[name].signature().model_validate(json_repair.loads(arguments))
        except KeyError as e:
            raise ValidationError(f"Tool {name} does not exist in toolbox") from e

    @override
    def repair_instructions(self, completion: ChatCompletionMessageToolCall, error: str) -> Conversation:
        function = completion.function
        return Conversation(
            messages=[
                AssistantMessage(content=json.dumps(completion.function.model_dump())),
                UserMessage(
                    content=textwrap.dedent(
                        f"""
                        Validation failed: {error}

                        Update your response to match this schema for function {function.name}:
                        {json.dumps(self.toolbox[function.name].signature().model_json_schema())}
                    """.strip()
                    ),
                ),
            ]
        )
