"""Core protocols for composable LLM calls.

This module defines the base protocols for validators, response handlers, and callables with signatures.
Validators check and potentially repair LLM responses.
Handlers process the responses using the validators.
Callables expose a typed interface via a Pydantic model for input validation.
"""

from __future__ import annotations

import logging
from typing import Generic, Protocol, Type, Union

from pydantic import BaseModel
from typing_extensions import TypeAlias, TypeVar, runtime_checkable

from ..types_.base import JSON
from ..types_.core import Conversation, Message
from ..types_.openai_compat import (
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
    """Protocol for validators that check LLM responses and provide repair instructions.

    Validators implement:
      - validate(): To verify and possibly transform an LLM response.
      - repair_instructions(): To generate guidance as a Conversation if the response is invalid.
    """

    def validate(self, completion: str | ChatCompletionMessageToolCall) -> ValidatorReturnType:
        """Validate a provided completion.

        Parameters
        ----------
        completion : str or ChatCompletionMessageToolCall
            The LLM response to be validated.

        Returns
        -------
        ValidatorReturnType
            The validated (and possibly transformed) response.
        """
        ...

    def repair_instructions(self, failed_content: str, error: str) -> Conversation | None:
        """Generate repair instructions for an invalid response.

        Parameters
        ----------
        failed_content : str
            The response content that failed validation.
        error : str
            The error message describing the validation failure.

        Returns
        -------
        Conversation or None
            A Conversation containing repair instructions, or None if not applicable.
        """
        ...


@runtime_checkable
class Handler(Generic[ContentHandlerReturnType, ToolHandlerReturnType], Protocol):
    """Protocol for processing LLM responses with validation and repair steps.

    Handlers determine how to extract, validate, and possibly repair responses received from the LLM.

    Attributes
    ----------
    max_repair_attempts : int
        The maximum number of times to attempt a repair before failing.
    """

    max_repair_attempts: int

    def process(self, response: ChatCompletion) -> ContentHandlerReturnType | ToolHandlerReturnType:
        """Process an LLM response.

        Parameters
        ----------
        response : ChatCompletion
            The response obtained from the LLM.

        Returns
        -------
        ContentHandlerReturnType or ToolHandlerReturnType
            The processed and validated result.
        """
        ...

    def repair(self, message: Message, error: str) -> Conversation | None:
        """Generate repair instructions based on an invalid response.

        Parameters
        ----------
        message : Message
            The invalid response message.
        error : str
            The error that triggered the repair.

        Returns
        -------
        Conversation or None
            A Conversation containing the repair steps, or None if repair is not possible.
        """
        ...


@runtime_checkable
class CallableWithSignature(Generic[CallableReturnType], Protocol):
    """Protocol for callables that expose a Pydantic validated signature.

    Attributes
    ----------
    signature : Type[BaseModel]
        A Pydantic model that defines the structure and types of input parameters.
    schema : JSON
        A JSON schema derived from the signature for external validation.
    returns : Type[CallableReturnType]
        The type that the callable is expected to return.
    """

    signature: Type[BaseModel]
    schema: JSON
    returns: Type[CallableReturnType]

    def __call__(self, *args, **kwargs) -> CallableReturnType:
        """Execute the callable with validated inputs.

        Parameters
        ----------
        *args :
            Positional arguments.
        **kwargs :
            Keyword arguments.

        Returns
        -------
        CallableReturnType
            The result of executing the callable.
        """
        ...
