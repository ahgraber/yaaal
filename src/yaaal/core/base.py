from __future__ import annotations

import logging
from typing import Protocol, Type

from pydantic import BaseModel
from typing_extensions import TypeVar, runtime_checkable

from aisuite import Client

from .prompt import Prompt
from ..types.base import JSON
from ..types.core import Conversation, Message, ResponseMessage, ValidatorResult
from ..types.openai_compat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallFunction,
    convert_response,
)

logger = logging.getLogger(__name__)

CallableReturnType = TypeVar("CallableReturnType", covariant=True)


@runtime_checkable
class CallableWithSignature(Protocol[CallableReturnType]):
    """Base protocol for Callables with signature."""

    def __call__(self, *args, **kwargs) -> CallableReturnType:
        """Execute the operation."""
        ...

    def signature(self) -> Type[BaseModel]:
        """Provide the callable signature as json schema."""
        ...


class Validator(Protocol):
    """Base protocol for all validators."""

    def validate(self, completion: str | ChatCompletionMessageToolCall) -> ValidatorResult:
        """Validate response."""
        raise NotImplementedError

    def repair_instructions(self, failed_content: str, error: str) -> Conversation | None:
        """Generate repair instructions when validation fails."""
        raise NotImplementedError


@runtime_checkable
class Handler(Protocol):
    """Protocol for response handlers."""

    @property
    def max_repair_attempts(self) -> int:
        """Maximum number of times to retry validation."""
        ...

    def __call__(self, response: ChatCompletion) -> ResponseMessage:
        """Process the LLM response."""
        raise NotImplementedError

    def repair(self, message: Message, error: str) -> Conversation | None:
        """Generate repair instructions for invalid response."""
        raise NotImplementedError


@runtime_checkable
class BaseCaller(Protocol):
    """Protocol for all Callers."""

    @property
    def client(self) -> Client:
        """Client called for every execution of the Caller instance."""
        ...

    @property
    def model(self) -> str:
        """Model called for every execution of the Caller instance."""
        ...

    @property
    def prompt(self) -> Prompt:
        """Prompt used to construct messages arrays."""
        ...

    @property
    def request_params(self) -> dict[str, JSON]:
        """Request parameters used for every execution."""
        ...

    def signature(self) -> Type[BaseModel]:
        """Provide the Caller's function signature as json schema."""
        # It seems weird that the signature is defined in the Prompt when the Caller is callable,
        # but the Prompt has everything required to define the signatuer
        # whereas the Caller is just a wrapper to generate the request.
        ...

    def __call__(
        self,
        *,
        system_vars: dict | None = None,
        user_vars: dict | None = None,
        conversation: Conversation | None = None,
    ) -> ResponseMessage:
        """Execute the API Call."""
        ...
