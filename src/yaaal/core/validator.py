"""Validation components for LLM responses.

Validators ensure that LLM responses match expected formats and provide mechanisms for repair.
Each validator implements methods to:
- validate(): Check and/or transform the response.
- repair_instructions(): Generate guidance to fix invalid responses.
Supports simple passthrough, schema-based (Pydantic) validation, regex pattern matching, and tool call validation.
"""

from __future__ import annotations

import json
import logging
import textwrap
from typing import Pattern

import json_repair
from pydantic import BaseModel
from typing_extensions import override, runtime_checkable

from .base import Validator, ValidatorReturnType
from .exceptions import ValidationError
from .tool import CallableWithSignature
from ..types.core import (
    AssistantMessage,
    Conversation,
    UserMessage,
)
from ..types.openai_compat import (
    ChatCompletionMessageToolCall,
)

logger = logging.getLogger(__name__)


class PassthroughValidator(Validator[str]):
    """A validator that returns the input unchanged if it is a string.

    This validator performs minimal checking and is useful as a baseline or for raw text responses.

    Examples
    --------
    >>> validator = PassthroughValidator()
    >>> validator.validate("Hello")
    'Hello'

    Raises
    ------
        TypeError: If the input is not a string.
    """

    @override
    def validate(self, completion: str) -> str:
        if isinstance(completion, str):
            return completion
        else:
            raise TypeError(f"Expected 'str' completion, received '{type(completion)}'")

    @override
    def repair_instructions(self, completion: str, error: str) -> None:
        return None


class PydanticValidator(Validator[BaseModel]):
    """A validator that uses a Pydantic model to parse and verify JSON responses.

    Attempts to fix common JSON formatting issues using a repair mechanism.

    Args:
        model: A Pydantic model class defining the expected schema.

    Examples
    --------
    >>> from pydantic import BaseModel
    >>> class User(BaseModel):
    ...     name: str
    ...     age: int
    >>> validator = PydanticValidator(User)
    >>> result = validator.validate('{"name": "Bob", "age": 42}')
    >>> isinstance(result, User)
    True

    Raises
    ------
        ValidationError: If the input fails to match the model schema.
    """

    def __init__(self, model: type[BaseModel]):
        """Initialize the Pydantic validation model.

        Args:
            model: Pydantic model class defining the schema.
        """
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
                        Please update your response to conform to this schema:
                        {json.dumps(self.model.model_json_schema())}
                        """.strip()
                    ),
                ),
            ]
        )


class RegexValidator(Validator[str]):
    r"""A validator that extracts a substring matching a regex pattern.

    Uses a compiled regex to search the text and returns the first match.

    Args:
        pattern: A compiled regular expression to match against the response.

    Examples
    --------
    >>> import re
    >>> validator = RegexValidator(re.compile(r"[\w\.-]+@[\w\.-]+\.\w+"))
    >>> validator.validate("Email: user@example.com")
    'user@example.com'

    Raises
    ------
        ValidationError: If no match is found.
    """

    def __init__(self, pattern: Pattern):
        """Initialize with a regex pattern.

        Args:
            pattern: Compiled regex pattern used for matching.
        """
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
                        The response must match the regex pattern: {self.pattern.pattern}
                        Please adjust your response accordingly.
                        """.strip()
                    ),
                ),
            ]
        )


class ToolValidator(Validator[BaseModel]):
    """A validator for tool calls that verifies function arguments against registered tool schemas.

    Manages a toolbox of callable tools and validates their call arguments.

    Args:
        toolbox: A list of callable tools that implement a signature for validation.

    Examples
    --------
    >>> @tool
    ... def add(a: int, b: int) -> int:
    ...     "Add two numbers."
    ...     return a + b
    >>> validator = ToolValidator([add])
    >>> result = validator.validate(
    ...     ChatCompletionMessageToolCall(function=FunctionCall(name="add", arguments='{"a": 1, "b": 2}'))
    ... )
    >>> isinstance(result, add.signature)
    True

    Raises
    ------
        ValidationError: If the tool name is not found or the arguments are invalid.
    """

    def __init__(self, toolbox: list[CallableWithSignature]):
        """Initialize the ToolValidator with a toolbox of callable tools.

        Args:
            toolbox: List of tool callables to validate against.

        Raises
        ------
            TypeError: If any tool does not adhere to the required interface.
        """
        # Add a check for an empty toolbox if needed
        if not toolbox:
            logger.warning("ToolValidator initialized with an empty toolbox")
        self.toolbox = toolbox

    @property
    def toolbox(self) -> dict[str, CallableWithSignature]:
        """Return the registered toolbox of tools."""
        return self._toolbox

    @toolbox.setter
    def toolbox(self, toolbox: list[CallableWithSignature]):
        """Register tools by mapping their names to the callable."""
        tb = {}
        for tool in toolbox:
            if not isinstance(tool, CallableWithSignature):
                raise TypeError(f"Toolbox requires CallableWithSignature objects. Received {tool}: {type(tool)}")

            if not tool.signature.__doc__:
                logger.warning(
                    f"Function {tool.signature.__name__} should have a docstring for proper signature export."
                )
            try:
                tb[tool.signature.__name__] = tool
            except Exception:
                logger.exception(f"Error while registering toolbox entry for {str(tool)}")
                raise
        self._toolbox = tb

    @override
    def validate(self, completion: ChatCompletionMessageToolCall) -> BaseModel:
        """Validate a tool call by matching its name and validating its arguments.

        Args:
            completion: The tool call message from the LLM.

        Returns
        -------
            A Pydantic model instance representing the validated arguments.

        Raises
        ------
            ValidationError: If the tool name is not found or the arguments fail to validate.
        """
        name = completion.function.name
        arguments = completion.function.arguments
        try:
            return self.toolbox[name].signature.model_validate(json_repair.loads(arguments))
        except KeyError as e:
            raise ValidationError(f"Tool {name} does not exist in toolbox") from e

    @override
    def repair_instructions(self, completion: ChatCompletionMessageToolCall, error: str) -> Conversation:
        """Generate repair instructions for an invalid tool call.

        Args:
            completion: The original tool call message.
            error: The validation error message.

        Returns
        -------
            A Conversation with instructions for correcting the tool call.
        """
        function = completion.function
        return Conversation(
            messages=[
                AssistantMessage(content=json.dumps(completion.function.model_dump())),
                UserMessage(
                    content=textwrap.dedent(
                        f"""
                        Validation failed: {error}
                        Please update your response to conform to the following schema for function '{function.name}':
                        {json.dumps(self.toolbox[function.name].schema)}
                        """.strip()
                    ),
                ),
            ]
        )
