"""Components for composable LLM calls.

A Validator ensures the LLM response matches our expectations and provides retry/repair instructions in case the validation fails.

Each validator implements the BaseValidator interface with two main methods:
- validate(): Validates the completion/response
- repair_instructions(): Generates instructions for fixing validation failures

The module supports various validation strategies including:
- Passthrough validation (i.e., no validation)
- Schema-based validation using pydantic models
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
    """Simple validator that passes content through unchanged.

    Validates that input is a string but performs no other checks.
    Used as baseline validator or for raw text handling.

    Examples
    --------
        >>> validator = PassthroughValidator()
        >>> result = validator.validate("Hello")
        >>> result
        'Hello'
        >>> assert result == "Hello"

        Raises TypeError for non-string input:

        >>> validator.validate(123)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        TypeError: Expected 'str' completion, received '<class 'int'>'
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
    """Validate using pydantic models.

    Parses JSON responses and validates against provided model schema.
    Handles malformed JSON through repair mechanism.

    Args:
        model: Pydantic model class for validation

    Examples
    --------
        Define and use a model:

        >>> from pydantic import BaseModel
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        >>> validator = PydanticValidator(User)
        >>> result = validator.validate('{"name": "Bob", "age": 42}')
        >>> isinstance(result, User)
        True

        Missing fields raise ValidationError:

        >>> validator.validate('{"name": "Bob"}')  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValidationError: Input validation failed
    """

    def __init__(self, model: type[BaseModel]):
        """Initialize with validation model.

        Args:
            model: Pydantic model class defining schema
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

                Please update your response to match this schema:
                {json.dumps(self.model.model_json_schema())}
                """.strip()
                    ),
                ),
            ]
        )


class RegexValidator(Validator[str]):
    r"""Validate using regex patterns.

    Validates text against provided regex pattern and extracts first match.

    Args:
        pattern: Compiled regex pattern for validation

    Examples
    --------
        Match email addresses:

        >>> import re
        >>> validator = RegexValidator(re.compile(r"[\w\.-]+@[\w\.-]+\.\w+"))
        >>> validator.validate("Contact: user@example.com")
        'user@example.com'

        Invalid input raises ValidationError:

        >>> validator.validate("Invalid email")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValidationError: Response did not match expected pattern
    """

    def __init__(self, pattern: Pattern):
        """Initialize with regex pattern.

        Args:
            pattern: Compiled regex pattern
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
                    Response must match the following regex pattern: {self.pattern.pattern}

                    Update your response to ensure it is valid.
                    """.strip()
                    ),
                ),
            ]
        )


class ToolValidator(Validator[BaseModel]):
    """Validate tool calls against registered function signatures.

    Manages toolbox of callable tools and validates calls against their schemas.

    Args:
        toolbox: List of callable tools with signatures

    Examples
    --------
        >>> @tool
        ... def add(a: int, b: int) -> int:
        ...     "Add two numbers."
        ...     return a + b
        >>> validator = ToolValidator([add])
        >>> result = validator.validate(
        ...     ChatCompletionMessageToolCall(
        ...         function=FunctionCall(
        ...             name="add",
        ...             arguments='{"a": 1, "b": 2}',
        ...         )
        ...     )
        ... )
        >>> isinstance(result, add.signature)
    """

    def __init__(self, toolbox: list[CallableWithSignature]):
        """Initialize with list of tools.

        Args:
            toolbox: List of callable tools with signatures

        Raises
        ------
            TypeError: If tools lack required interfaces
        """
        self.toolbox = toolbox

    @property
    def toolbox(self) -> dict[str, CallableWithSignature]:
        """Available tools."""
        return self._toolbox

    @toolbox.setter
    def toolbox(self, toolbox: list[CallableWithSignature]):
        """Register list of tools by name."""
        tb = {}
        for tool in toolbox:
            if not isinstance(tool, CallableWithSignature):
                raise TypeError(
                    f"Toolbox requires Caller or CallableWithSignature objects.  Received {tool}: {type(tool)}"
                )

            if not tool.signature.__doc__:
                logger.warning(f"Function {tool.signature.__name__} requires docstrings for viable signature.")
            try:
                tb[tool.signature.__name__] = tool
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
            return self.toolbox[name].signature.model_validate(json_repair.loads(arguments))
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
                        {json.dumps(self.toolbox[function.name].schema)}
                    """.strip()
                    ),
                ),
            ]
        )
