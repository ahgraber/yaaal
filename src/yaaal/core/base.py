from __future__ import annotations

import logging
from typing import Generic, Protocol, Type, Union

from pydantic import BaseModel
from typing_extensions import TypeAlias, TypeVar, runtime_checkable

from .prompt import Prompt
from ..types.base import JSON
from ..types.core import Conversation, Message
from ..types.openai_compat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallFunction,
    convert_response,
)

logger = logging.getLogger(__name__)

ValidatorReturnType = TypeVar("ValidatorReturnType", covariant=True)
ContentHandlerReturnType = TypeVar("ContentHandlerReturnType", covariant=True)
ToolHandlerReturnType = TypeVar("ToolHandlerReturnType", bound=Union[BaseModel, str, None], covariant=True)
CallableReturnType = TypeVar("CallableReturnType", covariant=False)


@runtime_checkable
class Validator(Generic[ValidatorReturnType], Protocol):
    """Base protocol for all validators."""

    def validate(self, completion: str | ChatCompletionMessageToolCall) -> ValidatorReturnType:
        """Validate response."""
        ...

    def repair_instructions(self, failed_content: str, error: str) -> Conversation | None:
        """Generate repair instructions when validation fails."""
        ...


@runtime_checkable
class Handler(Generic[ContentHandlerReturnType, ToolHandlerReturnType], Protocol):
    """Protocol for response handlers."""

    max_repair_attempts: int
    """Maximum number of times to retry validation."""

    def process(self, response: ChatCompletion) -> ContentHandlerReturnType | ToolHandlerReturnType:
        """Process the LLM response."""
        ...

    def repair(self, message: Message, error: str) -> Conversation | None:
        """Generate repair instructions for invalid response."""
        ...


@runtime_checkable
class CallableWithSignature(Generic[CallableReturnType], Protocol):
    """Base protocol for Callables with signature.

    Attributes
    ----------
        signature (Type[BaseModel]): Pydantic model defining callable parameters
        schema (JSON): Provide the callable's signature as JSON schema
        returns (Type[CallableReturnType]): Return type annotation
    """

    signature: Type[BaseModel]
    schema: JSON
    returns: Type[CallableReturnType]

    def __call__(self, *args, **kwargs) -> CallableReturnType:
        """Execute the operation."""
        ...
