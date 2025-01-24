"""Components for composable LLM calls.

A Caller associates a Prompt with a specific LLM client and call parameters (assumes OpenAI-compatibility through a framework like `aisuite`).
This allows every Caller instance to use a different model and/or parameters, and sets expectations for the Caller instance.

Since Callers leverage the LLM API directly, they can do things like function-calling / tool use.
If a tool-call instruction is detected, the Caller can try to `invoke` that call and return the function result as the response.

Additionally, Callers can be used as functions/tools in tool-calling workflows by leveraging Caller.signature() which denotes the inputs the Caller's Prompt requires.
Since a Caller has a specific client and model assigned, this effectively allows us to use Callers to route to specific models for specific use cases.
Since Callers can behave as functions themselves, we enable complex workflows where Callers can call Callers (ad infinitum ad nauseum).

Optional validator mixins provide response validation based on anticipated response formatting.

PydanticResponseValidatorMixin validates the response based on a Pydantic object (it is recommended to use JSON mode, Structured Outputs, or Function-Calling for reliability).

RegexResponseValidatorMixin validates the response based on regex matching.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import logging
from typing import Generic, Match, Pattern, Type, TypeVar, override

import json_repair
from pydantic import BaseModel, ValidationError

from aisuite import Client

from .prompt import Prompt
from .tools import CallableWithSignature, respond_as_tool
from ..types.base import JSON
from ..types.core import Conversation, Message, ToolMessage
from ..types.openai_compat import (
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionResponse,
    ChatCompletionToolCallFunction,
)

logger = logging.getLogger(__name__)


class CallerValidationError(Exception):
    pass


# TODO: use Caller as ABC
# make StructuredCaller, ToolCaller, RegexCaller
# for structuredCaller, it seems like tool use is the way to cover both anthropic and openai


class Caller(ABC):
    """Provides client and standard parameters used for each call."""

    _client: Client
    _model: str
    _history: Conversation | None
    _prompt: Prompt
    _request_params: dict[str, JSON] = {}
    _toolbox: dict[str, Caller | CallableWithSignature] = {}
    _auto_invoke: bool = False
    # _tools: list[dict[str, JSON]] = []

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

    @property
    def prompt(self) -> Prompt:
        """BasePrompt object used to construct messages arrays."""
        return self._prompt

    @prompt.setter
    def prompt(self, prompt: Prompt):
        self._prompt = prompt

    @property
    def toolbox(self) -> dict[str, Caller | CallableWithSignature]:
        """Tools available to the Agent."""
        return self._toolbox

    @toolbox.setter
    def toolbox(self, toolbox: list[Caller | CallableWithSignature] | None):
        self._toolbox = {tool.signature().model_json_schema()["title"]: tool for tool in toolbox} if toolbox else {}

    @property
    def auto_invoke(self) -> bool:
        """Boolean flag determining whether to automatically invoke the function call or just return the params."""
        return self._auto_invoke

    @auto_invoke.setter
    def auto_invoke(self, auto_invoke: bool):
        self._auto_invoke = auto_invoke

    @property
    def request_params(self) -> dict[str, JSON]:
        """Request parameters used for every execution of the Caller instance."""
        return self._request_params

    @request_params.setter
    def request_params(self, request_params: dict[str, JSON] | None):
        _request_params = request_params or {}
        if "model" in _request_params:
            raise ValueError("'model' should be set separately and not included in 'request_params'.")

        # # TODO: if we provide a pydantic model for response validation, we should set request params to specify structured generation
        # # TODO: how can we make this work with tools?
        # # TODO: how can we ensure correct params for all providers?
        # if hasattr(self, "response_validator") and issubclass(self.response_validator, BaseModel):
        #     _request_params["response_format"] = {"type": "json_object"}

        self._request_params = _request_params

    def signature(self) -> Type[BaseModel]:
        """Provide the Caller's function signature as json schema."""
        # It seems weird that the signature is defined in the Prompt when the Caller is callable,
        # but the Prompt has everything required to define the signatuer
        # whereas the Caller is just a wrapper to generate the request.
        return self.prompt.signature()

    def __call__(
        self,
        *,
        system_vars: dict,
        user_vars: dict | None,
        conversation: Conversation | None = None,
    ) -> str | BaseModel | ToolMessage:
        """Call the API."""
        _rendered = self.prompt.render(system_vars=system_vars, user_vars=user_vars)
        if conversation:
            conversation.messages.extend(_rendered.messages)
        else:
            conversation = _rendered

        response = self._chat_completions_create(conversation=conversation)
        return self._handle_response(conversation=conversation, response=response, repair=0)

    def _chat_completions_create(self, conversation: Conversation) -> ChatCompletionResponse:
        """Call the LLM chat endpoint."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=conversation.model_dump()["messages"],
            **self.request_params,
        )

        return ChatCompletionResponse(**response.dict())

    def _handle_response(
        self, conversation: Conversation, response: ChatCompletionResponse, repair: int = 0
    ) -> str | BaseModel | ToolMessage:
        """Handle the response object."""
        # TODO: allow/disable multiple generations per input?
        if content := response.choices[0].message.content:
            return self._handle_content(conversation=conversation, content=content, repair=repair)

        elif response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]

            return self._handle_tool_call(conversation=conversation, tool_call=tool_call, repair=repair)

        else:
            raise ValueError("Unexpected response object - could not identify message.content or message.tool_calls")

    def _handle_content(self, conversation: Conversation, content: str, repair: int = 0) -> str | BaseModel:
        """Handle the message content."""
        try:
            if hasattr(self, "_validate_content"):
                return self._validate_content(content)
            else:
                return self._default_validate_content(content)
        except Exception as e:
            if repair >= 2:  # Max 3 attempts (0, 1, 2)
                raise CallerValidationError(f"Max repair attempts reached: {e}") from e

            if hasattr(self, "_render_repair"):
                repair_msgs = self._render_repair(content, str(e))
                if not repair_msgs:
                    raise CallerValidationError(f"No repair strategy: {e}") from e

                logger.debug(f"Attempting repair for exception raised during validation: {e}")
                conversation.messages.extend(repair_msgs.messages)
                return self._handle_response(
                    conversation=conversation,
                    response=self._chat_completions_create(conversation),
                    repair=repair + 1,
                )
            else:
                logger.warning("Content failed validation, returning raw")
                return content

    # @abstractmethod
    def _default_validate_content(self, content: str) -> str:
        """Validate the model's response content."""
        logger.debug("Using default (passthrough) validator.")
        return content

    # @abstractmethod
    # def _default_render_repair(self, response_content: str, exception: str) -> None:
    #     """Render Conversation containing instructions to attempt to fix the validation error."""
    #     return None

    # TODO: ChatCompletionMessageToolCall when aisuite updates
    def _handle_tool_call(
        self, conversation: Conversation, tool_call: ChatCompletionMessageToolCall, repair: int = 0
    ) -> BaseModel | ToolMessage:
        """Handle the tool call."""
        name = tool_call.function.name
        arguments = tool_call.function.arguments

        try:
            validated = self._validate_tool(name=name, arguments=arguments)
        except KeyError:
            logger.exception(f"Tool {name} does not exist in the toolbox.")
            raise
        except ValidationError:
            # TODO: retry for tool call?
            # if repair >= 2:  # Max 3 attempts (0, 1, 2)
            #     raise CallerValidationError(f"Max repair attempts reached: {e}") from e
            # ...
            logger.exception(
                f"Validation failed while validating tool_call for function {name} with arguments {arguments}"
            )
            raise

        except Exception:
            logger.exception(
                f"Unexpected Exception while validating tool_call for function {name} with arguments {arguments}"
            )
            raise
        if self.auto_invoke:
            try:
                result = self.toolbox[name](**validated.model_dump())
            except Exception:
                logger.exception(
                    f"Unexpected Exception while invoking tool_call for function {name} with arguments {arguments}"
                )
                raise
            return ToolMessage(
                tool_call_id=tool_call.id,
                content=result if isinstance(result, str) else json.dumps(result),
            )
        else:
            return validated

    def _validate_tool(self, name: str, arguments: str) -> BaseModel:
        """Validate the model's tool call."""
        return self.toolbox[name].signature().model_validate(json_repair.loads(arguments))


# ----------------------------
T = TypeVar("T")  # Return type for validate method


class ValidatorMixin(ABC, Generic[T]):
    """Abstract base class for validation mixins."""

    @abstractmethod
    def _validate_content(self, response: str) -> T:
        """Validate the model's response."""
        pass

    @abstractmethod
    def _render_repair(self, response_content: str, exception: str) -> Conversation:
        """Render Conversation containing instructions to attempt to fix the validation error."""
        pass


# TODO
# Streaming validation
# https://docs.pydantic.dev/latest/concepts/experimental/#partial-validation
# Maybe not needed, since we'll need to wait for the full response before whatever the next step is can use it
class PydanticResponseValidatorMixin(ValidatorMixin):
    """Validate response with Pydantic."""

    _response_validator: Type[BaseModel]

    @property
    def response_validator(self) -> Type[BaseModel]:
        """Pydantic model used to validate responses."""
        return self._response_validator

    @response_validator.setter
    def response_validator(self, response_validator: Type[BaseModel]):
        self._response_validator = response_validator

    @override
    def _render_repair(self, response_content: str, exception: str) -> Conversation:
        """Render Conversation containing instructions to attempt to fix the validation error."""
        messages = [
            Message(role="assistant", content=response_content),
            Message(
                role="user",
                content=f"Received the following exception while validating the above message: {exception}\n\nUpdate your response to ensure it is valid.",
            ),
        ]
        return Conversation(messages=messages)

    @override
    def _validate_content(self, response: str) -> BaseModel:
        """Validate the model's response."""
        logger.debug(f"Validating {response}")
        return self.response_validator.model_validate(json_repair.loads(response))


class RegexResponseValidatorMixin(ValidatorMixin):
    """Validate response with Regular Expressions."""

    _response_validator: Pattern

    @property
    def response_validator(self) -> Pattern:
        """Compiled regex pattern used to validate responses."""
        return self._response_validator

    @response_validator.setter
    def response_validator(self, response_validator: Pattern):
        self._response_validator = response_validator

    @override
    def _render_repair(self, response_content: str, exception: str) -> Conversation:
        """Render messages array containing instructions to attempt to fix the validation error."""
        messages = [
            Message(role="assistant", content=response_content),
            Message(
                role="user",
                content=f"Response must match the following regex pattern: {self.response_validator.pattern}\n\nUpdate your response to ensure it is valid.",
            ),
        ]
        return Conversation(messages=messages)

    @override
    def _validate_content(self, response: str) -> Match:
        """Validate the response against regex pattern."""
        match = self.response_validator.match(response)
        if not match:
            raise ValueError("Response did not match expected pattern")
        return match
