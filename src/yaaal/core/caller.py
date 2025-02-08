"""Components for composable LLM calls.

A Caller is the basic structure that wraps all logic required for LLM call-and-response.

A Caller associates a Prompt with a specific LLM client and call parameters (assumes OpenAI-compatibility through a framework like `aisuite`).
This allows every Caller instance to use a different model and/or parameters, and sets expectations for the Caller instance.
Whereas `Prompts` validate _inputs_ to the template and `Handlers` validate the LLM responses, `Callers` make it all happen.

Additionally, Callers can be used as functions/tools in tool-calling workflows by leveraging Caller.signature() which provides the inputs the Caller's Prompt requires as a JSON schema.
Since a Caller has a specific client and model assigned, this effectively allows us to use Callers to route to specific models for specific use cases.
Since Callers can behave as functions themselves, we enable complex workflows where Callers can call Callers (ad infinitum ad nauseum).

Simple factory functions create Callers where the use case is defined by their handlers:

- `ChatCaller`: a simple Caller implementation designed for chat messages without response validation.
- `RegexCaller`: uses regex for response validation.
- `StructuredCaller`:  is intended for structured responses, and uses Pydantic for response validation.
- `ToolCaller`: a configuration for tool-use; can optionally invoke the tool based on arguments in the LLM's response and return the function results.
"""

from __future__ import annotations

import logging
from typing import Type

from pydantic import BaseModel

from aisuite import Client
from openai import pydantic_function_tool as openai_pydantic_function_tool

from .base import CallableWithSignature
from .exceptions import ValidationError
from .handler import CompositeHandler, ResponseHandler, ToolHandler
from .prompt import Prompt
from .tools import anthropic_pydantic_function_tool
from .validator import PassthroughValidator, PydanticValidator, RegexValidator, ToolValidator
from ..types.base import JSON
from ..types.core import Conversation, ResponseMessage
from ..types.openai_compat import ChatCompletion, convert_response

logger = logging.getLogger(__name__)


class Caller(CallableWithSignature):
    """Caller implementation."""

    def __init__(
        self,
        client: Client,
        model: str,
        prompt: Prompt,
        handler: ResponseHandler | ToolHandler | CompositeHandler,
        request_params: dict[str, JSON] | None = None,
        max_repair_attempts: int = 2,
    ):
        self._client = client
        self._model = model
        self._prompt = prompt
        self.handler = handler
        self.request_params = self._make_request_params(request_params)
        self.max_repair_attempts = max_repair_attempts

    @property
    def client(self) -> Client:
        """Client called for every execution of the Caller instance."""
        return self._client

    @client.setter
    def client(self, client: Client):
        self._client = client

    @property
    def model(self) -> str:
        """Model called for every execution of the Caller instance."""
        return self._model

    @model.setter
    def model(self, model: str):
        self._model = model
        logger.debug(f"All API requests for {self.__class__.__name__} will use model : {self._model}")

    @property
    def prompt(self) -> Prompt:
        """Prompt used to construct messages arrays."""
        return self._prompt

    @prompt.setter
    def prompt(self, prompt: Prompt):
        self._prompt = prompt

    @property
    def request_params(self) -> dict[str, JSON]:
        """Request parameters used for every execution."""
        return self._request_params

    @request_params.setter
    def request_params(self, request_params: dict[str, JSON] | None):
        self._request_params = self._make_request_params(request_params)
        logger.debug(f"All API requests for {self.__class__.__name__} will use params : {self._request_params}")

    def _make_request_params(self, request_params: dict[str, JSON] | None) -> dict[str, JSON]:
        params = request_params or {}
        if "model" in params:
            raise ValueError("'model' should be set separately")
        return params

    @property
    def max_repair_attempts(self) -> int:
        """Maximum number of retries for validation failures."""
        return self._max_repair_attempts

    @max_repair_attempts.setter
    def max_repair_attempts(self, max_repair_attempts: int):
        self._max_repair_attempts = max_repair_attempts

    def __call__(
        self,
        *,
        system_vars: dict | None = None,
        user_vars: dict | None = None,
        conversation: Conversation | None = None,
    ) -> ResponseMessage:
        """Call the API."""
        rendered = self.prompt.render(system_vars=system_vars, user_vars=user_vars)
        if conversation:
            conversation.messages.extend(rendered.messages)
        else:
            conversation = rendered

        response = self._chat_completions_create(conversation)
        return self._handle_with_repair(conversation=conversation, response=response)

    def _chat_completions_create(self, conversation: Conversation) -> ChatCompletion:
        """Call the LLM chat endpoint."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=conversation.model_dump()["messages"],
            **self.request_params,
        )
        return convert_response(response)

    def _handle_with_repair(
        self, conversation: Conversation, response: ChatCompletion, repair_attempt: int = 0
    ) -> ResponseMessage:
        """Handle response with repair attempts."""
        try:
            return self.handler(response)
        except Exception as e:
            if repair_attempt >= self.max_repair_attempts:
                raise ValidationError("Max repair attempts reached") from e

            logger.debug(f"Repair {repair_attempt} after error handling response {e}")
            msg = response.choices[0].message
            repair_prompt = self.handler.repair(msg, str(e))

            if not repair_prompt:
                raise ValidationError("No repair instructions available") from e

            conversation.messages.extend(repair_prompt.messages)
            new_response = self._chat_completions_create(conversation)
            return self._handle_with_repair(conversation, new_response, repair_attempt + 1)

    def signature(self) -> Type[BaseModel]:
        """Provide the Caller's function signature.

        It seems weird that the signature is defined in the Prompt when the Caller is "callable",
        but the Prompt defines the signature requirements whereas the Caller is just a wrapper to generate the request.
        """
        return self.prompt.signature()


# Factory functions
def create_chat_caller(
    client: Client,
    model: str,
    prompt: Prompt,
    request_params: dict[str, JSON] | None = None,
) -> Caller:
    """Create a basic chat Caller without validation."""
    handler = ResponseHandler(PassthroughValidator())
    return Caller(client, model, prompt, handler, request_params)


def create_structured_caller(
    client: Client,
    model: str,
    prompt: Prompt,
    response_model: Type[BaseModel],
    request_params: dict[str, JSON] | None = None,
) -> Caller:
    """Create a Caller for structured responses."""
    handler = ResponseHandler(PydanticValidator(response_model))
    params = request_params or {} | _make_structured_params(model)
    return Caller(client, model, prompt, handler, params)


def create_tool_caller(
    client: Client,
    model: str,
    prompt: Prompt,
    toolbox: list[CallableWithSignature],
    request_params: dict[str, JSON] | None = None,
    auto_invoke: bool = False,
) -> Caller:
    """Create a Caller for tool use."""
    handler = ToolHandler(ToolValidator(toolbox), auto_invoke)
    params = (request_params or {}) | _make_tool_params(model, toolbox)
    return Caller(client, model, prompt, handler, params)


# Helper functions
def _make_structured_params(model: str) -> dict[str, JSON]:
    """Make request params for structured output."""
    if "anthropic" in model:
        return {"response_format": {"type": "json"}}
    return {"response_format": {"type": "json_object"}}


def _make_tool_params(model: str, toolbox: list[CallableWithSignature]) -> dict[str, JSON]:
    """Make request params for tool use."""
    tools = [
        anthropic_pydantic_function_tool(t.signature())
        if "anthropic" in model
        else openai_pydantic_function_tool(t.signature())
        for t in toolbox
    ]

    if "anthropic" in model:
        return {"tools": tools, "tool_choice": {"type": "auto"}}
    return {"tools": tools, "tool_choice": "auto"}
