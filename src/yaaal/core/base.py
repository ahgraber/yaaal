from __future__ import annotations

import logging
from typing import Protocol, Type, Union

from pydantic import BaseModel
from typing_extensions import TypeVar, runtime_checkable

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

ValidatorReturnType = TypeVar("ValidatorReturnType", bound=Union[str, BaseModel], covariant=True)
ContentHandlerReturnType = TypeVar("ContentHandlerReturnType", bound=Union[str, BaseModel], covariant=True)
ToolHandlerReturnType = TypeVar("ToolHandlerReturnType", bound=BaseModel, covariant=True)
CallableReturnType = TypeVar("CallableReturnType", covariant=False)


@runtime_checkable
class Validator(Protocol[ValidatorReturnType]):
    """Base protocol for all validators."""

    def validate(self, completion: str | ChatCompletionMessageToolCall) -> ValidatorReturnType:
        """Validate response."""
        ...

    def repair_instructions(self, failed_content: str, error: str) -> Conversation | None:
        """Generate repair instructions when validation fails."""
        ...


@runtime_checkable
class Handler(Protocol[ContentHandlerReturnType, ToolHandlerReturnType]):
    """Protocol for response handlers."""

    max_repair_attempts: int
    """Maximum number of times to retry validation."""

    def __call__(self, response: ChatCompletion) -> ContentHandlerReturnType | ToolHandlerReturnType:
        """Process the LLM response."""
        ...

    def repair(self, message: Message, error: str) -> Conversation | None:
        """Generate repair instructions for invalid response."""
        ...


@runtime_checkable
class CallableWithSignature(Protocol[CallableReturnType]):
    """Base protocol for Callables with signature.

    Attributes
    ----------
        signature (Type[BaseModel]): Pydantic model defining callable parameters
        schema (JSON): Provide the callable's signature as JSON schema
        returns (Type[CallableReturnType]): Return type annotation
    """

    signature: Type[BaseModel]
    schema: JSON
    returns: CallableReturnType

    def __call__(self, *args, **kwargs) -> CallableReturnType:
        """Execute the operation."""
        ...
