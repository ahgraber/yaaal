"""Components for composable LLM calls.

A Caller associates a Prompt with a specific LLM client and call parameters (assumes OpenAI-compatibility).
This allows every Caller instance to use a different model and/or parameters, and sets expectations for the Caller instance.
Since Callers leverage the LLM API directly, they can do things like function-calling / tool use.
Callers can be used as functions/tools in tool-calling workflows by leveraging Caller.signature() which denotes the inputs the Caller's Prompt requires.

Furthermore, since Callers can behave as functions themselves, we enable complex workflows where Callers can call Callers (ad infinitum ad nauseum);
however, Agents are needed to actually invoke the tool call generated from the Caller

Optional validator mixins provide response validation based on anticipated response formatting.

PydanticResponseValidatorMixin validates the response based on a Pydantic object (it is recommended to use JSON mode, Structured Outputs, or Function-Calling for reliability).

RegexResponseValidatorMixin validates the response based on regex matching.
"""

from __future__ import annotations

from abc import abstractmethod
import json
import logging
from typing import Any, Match, Pattern, Type

import json_repair
from pydantic import BaseModel, Field

from ._types import JSON, Conversation, Message, ToolMessage
from .prompt import Prompt
from .tools import CallableWithSignature, respond_as_tool

logger = logging.getLogger(__file__)


class Caller:
    """Provides client and standard parameters used for each call."""

    _client: Any
    _history: Conversation | None
    _prompt: Prompt
    _request_params: dict[str, JSON] = {}
    _toolbox: dict[str, Caller | CallableWithSignature]
    # _tools: list[dict[str, JSON]] = []

    @property
    def client(self):
        """Client called for every execution of the Caller instance."""
        return self._client

    @client.setter
    def client(self, client):
        self._client = client

    @property
    def prompt(self) -> Prompt:
        """BasePrompt object used to construct messages arrays."""
        return self._prompt

    @prompt.setter
    def prompt(self, prompt: Prompt):
        self._prompt = prompt

    @property
    def toolbox(self) -> dict[str, Caller | CallableWithSignature] | None:
        """Tools available to the Agent."""
        return self._toolbox

    @toolbox.setter
    def toolbox(self, toolbox: list[Caller | CallableWithSignature] | None):
        self._toolbox = {tool.signature().model_json_schema()["title"]: tool for tool in toolbox} if toolbox else {}

    @property
    def request_params(self) -> dict[str, JSON]:
        """Request parameters used for every execution of the Caller instance."""
        return self._request_params

    @request_params.setter
    def request_params(self, request_params: dict[str, JSON] | None):
        self._request_params = request_params or {}

    @abstractmethod
    def __call__(self):
        """Call the API.

        This is not implemented to provide flexibility to use a variety of providers.

        Example (assuming templated prompt and validator mixin):
        >>> def __call__(self, *, system_vars: dict, user_vars: dict | None):
        >>>     messages = self.prompt.render(system_vars=system_vars, user_vars=user_vars)
        >>>     response = self.client.chat.completions.create(
        ...         messages=messages,
        ...         **self.request_params,
        ...         tools=[openai.pydantic_function_tool(tool) for tool in self.tools.values()]
        >>>     )
        >>>     content = response.choices[0].message.content
        >>>
        >>>     try:
        ...         validated = self.validate(response)
        >>>     except Exception as e:
        >>>         logger.debug(f"Attempting fix for exception raised during validation: {e}")
        >>>         repair_messages  = self.render_repair(
        ...             response_content=response.choices[0].message.content,
        ...             exception=str(e),
        >>>         )
        >>>         messages.extend(repair_messages)
        >>>
        >>>         response = self.client.chat.completions.create(
        ...             messages=messages,
        ...             **self.request_params,
        >>>         )
        >>>         content = response.choices[0].message.content
        >>>         return self.validate(content)
        >>>
        >>>     return validated
        """
        raise NotImplementedError

    def validate(self, response: str) -> str:
        """Validate the model's response."""
        return response

    def signature(self) -> Type[BaseModel]:
        """Provide the Caller's function signature as json schema."""
        # It seems weird that the signature is defined in the Prompt when the Caller is callable,
        # but the Prompt has everything required to define the signatuer
        # whereas the Caller is just a wrapper to generate the request.
        return self.prompt.signature()


# ----------------------------
# TODO
# Streaming validation
# https://docs.pydantic.dev/latest/concepts/experimental/#partial-validation
# Maybe not needed, since we'll need to wait for the full response before whatever the next step is can use it


class PydanticResponseValidatorMixin:
    """Validate response with Pydantic."""

    _response_validator: Type[BaseModel]

    @property
    def response_validator(self) -> Type[BaseModel]:
        """Pydantic model used to validate responses."""
        return self._response_validator

    @response_validator.setter
    def response_validator(self, response_validator: Type[BaseModel]):
        self._response_validator = response_validator

    def validate(self, response: str) -> BaseModel:
        """Validate the model's response."""
        return self.response_validator.model_validate(json_repair.loads(response))

    def render_repair(self, response_content: str, exception: str) -> Conversation:
        """Render Conversation containing instructions to attempt to fix the validation error."""
        messages = [
            Message(role="assistant", content=response_content),
            Message(
                role="user",
                content=f"Received the following exception while validating the above message: {exception}\n\nUpdate your response to ensure it is valid.",
            ),
        ]
        return Conversation(messages=messages)


class RegexResponseValidatorMixin:
    """Validate response with Regular Expressions."""

    _response_validator: Pattern

    @property
    def response_validator(self) -> Pattern:
        """Compiled regex pattern used to validate responses."""
        return self._response_validator

    @response_validator.setter
    def response_validator(self, response_validator: Pattern):
        self._response_validator = response_validator

    def validate(self, response: str) -> Match:
        """Validate the response against regex pattern."""
        match = self.response_validator.match(response)
        if not match:
            raise ValueError("Response did not match expected pattern")
        return match

    def render_repair(self, response_content: str, exception: str) -> Conversation:
        """Render messages array containing instructions to attempt to fix the validation error."""
        messages = [
            Message(role="assistant", content=response_content),
            Message(
                role="user",
                content=f"Response must match the following regex pattern: {self.response_validator.pattern}\n\nUpdate your response to ensure it is valid.",
            ),
        ]
        return Conversation(messages=messages)
