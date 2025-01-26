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

import json
import logging
import textwrap
from typing import Pattern, Type, TypeVar

import json_repair
from pydantic import BaseModel, ValidationError
from typing_extensions import override  # TODO: import from typing when drop support for 3.11

from aisuite import Client
import openai

from .prompt import Prompt
from .tools import CallableWithSignature, anthropic_pydantic_function_tool, respond_as_tool
from ..types.base import JSON
from ..types.core import Conversation, Message, ToolMessage
from ..types.openai_compat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallFunction,
    convert_response,
)

logger = logging.getLogger(__name__)


class CallerValidationError(Exception):
    pass


# TODO: use Caller as ABC
# make StructuredCaller, ToolCaller, RegexCaller
# for structuredCaller, it seems like tool use is the way to cover both anthropic and openai


class BaseCaller:
    """Base Caller implementation."""

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
    def request_params(self) -> dict[str, JSON]:
        """Request parameters used for every execution of the Caller instance."""
        return self._request_params

    @request_params.setter
    def request_params(self, request_params: dict[str, JSON] | None):
        self._request_params = self._make_request_params(request_params)
        logger.debug(f"All API requests for {self.__class__.__name__} will use : {self._request_params}")

    def _make_request_params(self, request_params: dict[str, JSON] | None) -> dict[str, JSON]:
        """Construct the request parameters."""
        _request_params = request_params or {}
        if "model" in _request_params:
            raise ValueError("'model' should be set separately and not included in 'request_params'.")

        # # TODO: if we provide a pydantic model for response validation, we should set request params to specify structured generation
        # # TODO: how can we make this work with tools?
        # # TODO: how can we ensure correct params for all providers?
        # if hasattr(self, "response_validator") and issubclass(self.response_validator, BaseModel):
        #     _request_params["response_format"] = {"type": "json_object"}

        return _request_params

    @property
    def max_repair_attempts(self) -> int:
        """Maximum number of retries when trying to pass validation."""
        return self._max_repair_attempts

    @max_repair_attempts.setter
    def max_repair_attempts(self, max_repair_attempts: int = 2):
        self._max_repair_attempts = max_repair_attempts

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

    def _chat_completions_create(self, conversation: Conversation) -> ChatCompletion:
        """Call the LLM chat endpoint."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=conversation.model_dump()["messages"],
            **self.request_params,
        )

        # NOTE: 'response'' is an openai object OR converted by aisuite to openai-compatible
        # refs:
        # - (types) https://github.com/andrewyng/aisuite/issues/98
        # - (fn call) https://github.com/andrewyng/aisuite/issues/55
        # - (fn call pt1) https://github.com/andrewyng/aisuite/commit/bd6b23fd72b3a391d96659b7faf5cdaed6a415dc
        logger.debug("Converting response object to ChatCompletion")
        return convert_response(response)

    def _handle_response(
        self, conversation: Conversation, response: ChatCompletion, repair: int = 0
    ) -> str | BaseModel | ToolMessage:
        """Handle the response object."""
        # TODO: allow/disable multiple generations per input?
        if content := response.choices[0].message.content:
            logger.debug("Response object has message.content")
            return self._handle_content(conversation=conversation, content=content, repair=repair)

        elif response.choices[0].message.tool_calls:
            logger.debug("Response object has message.tool_call(s), using first.")
            tool_call = response.choices[0].message.tool_calls[0]
            return self._handle_tool_call(conversation=conversation, tool_call=tool_call, repair=repair)

        else:
            raise ValueError("Unexpected response object - could not identify message.content or message.tool_calls")

    def _handle_content(self, conversation: Conversation, content: str, repair: int = 0) -> str | BaseModel:
        """Handle the message content."""
        try:
            return self._validate_content(content)
        except Exception as e:
            repair_msgs = self._repair_response(content, str(e))

            if repair > self.max_repair_attempts:  # Max 2 attempts (original, repair)
                raise CallerValidationError("Max repair attempts reached.") from e
                # logger.warning("Max repair attempts reached and could not validate. Returning failed content.")
                # return content

            if not repair_msgs:
                raise CallerValidationError(
                    f"{self.__class__.__name__}._render_repair() did not provide instructions for repair retry."
                ) from e
                # logger.warning(
                #     f"{self.__class__.__name__}._render_repair() did not provide instructions for repair retry, returning failed content."
                # )
                # return content

            else:
                logger.debug(f"Attempting repair for exception raised during content validation: {e}")
                conversation.messages.extend(repair_msgs.messages)
                return self._handle_response(
                    conversation=conversation,
                    response=self._chat_completions_create(conversation),
                    repair=repair + 1,
                )

    def _validate_content(self, content: str) -> str:
        """Validate the model's response content."""
        logger.debug("Using default (passthrough) validator.")
        return content

    def _repair_response(self, response_content: str, exception: str) -> None:
        """Render Conversation containing instructions to attempt to fix the response validation error."""
        return None

    def _handle_tool_call(
        self, conversation: Conversation, tool_call: ChatCompletionMessageToolCall, repair: int = 0
    ) -> BaseModel | ToolMessage:
        """Handle the tool call."""
        # logger.debug("Using default (passthrough) tool handler.")
        # return tool_call
        raise NotImplementedError

    def _validate_tool(self, name: str, arguments: str) -> BaseModel:
        """Validate the model's tool call."""
        raise NotImplementedError

    def _repair_tool(self, tool_call: ChatCompletionMessageToolCall, exception: str) -> None:
        """Render Conversation containing instructions to attempt to fix the tool call validation error."""
        # return None
        raise NotImplementedError


class ChatCaller(BaseCaller):
    """Simple Caller implementation that is designed for chat messages without validation."""

    def __init__(
        self,
        client: Client,
        model: str,
        prompt: Prompt,
        request_params: dict[str, JSON] | None = None,
    ):
        self.client = client
        self.model = model
        self.prompt = prompt

        # must come last since logic depends on other properties
        self.request_params = request_params


class RegexCaller(BaseCaller):
    """Caller implementation that is designed for chat messages with simple regex validation."""

    def __init__(
        self,
        client: Client,
        model: str,
        prompt: Prompt,
        response_validator: Pattern,
        request_params: dict[str, JSON] | None = None,
        max_repair_attempts: int = 2,
    ):
        self.client = client
        self.model = model
        self.prompt = prompt
        self.response_validator = response_validator
        self.max_repair_attempts = max_repair_attempts

        # must come last since logic depends on other properties
        self.request_params = request_params

    @property
    def response_validator(self) -> Pattern:
        """Compiled regex pattern used to validate responses."""
        return self._response_validator

    @response_validator.setter
    def response_validator(self, response_validator: Pattern):
        self._response_validator = response_validator

    @override
    def _validate_content(self, response: str) -> str:
        """Validate the response against regex pattern."""
        logger.debug("Validating response against regex pattern.")
        match = self.response_validator.search(response)
        if not match:
            raise ValueError("Response did not match expected pattern")

        return match.group()

    @override
    def _repair_response(self, response_content: str, exception: str) -> Conversation:
        """Render messages array containing instructions to attempt to fix the validation error."""
        messages = [
            Message(role="assistant", content=response_content),
            Message(
                role="user",
                content=textwrap.dedent(
                    f"""
                    Response must match the following regex pattern: {self.response_validator.pattern}

                    Update your response to ensure it is valid.
                    """.strip(),
                ),
            ),
        ]
        return Conversation(messages=messages)


class StructuredCaller(BaseCaller):
    """Caller implementation that is designed for chat messages with Pydantic validation.

    Tool-calling is the easiest way to ensure structured outputs that is supported by multiple providers.
    Therefore, we map the response_validator a tool
    """

    def __init__(
        self,
        client: Client,
        model: str,
        prompt: Prompt,
        response_validator: Type[BaseModel],
        request_params: dict[str, JSON] | None = None,
        max_repair_attempts: int = 2,
    ):
        self.client = client
        self.model = model
        self.prompt = prompt
        self.response_validator = response_validator
        self.max_repair_attempts = max_repair_attempts

        # must come last since logic depends on other properties
        self.request_params = request_params

    @override
    def _make_request_params(self, request_params: dict[str, JSON] | None) -> dict[str, JSON]:
        """Construct the request parameters."""
        params = request_params or {}
        if "model" in params:
            raise ValueError("'model' should be set separately and not included in 'request_params'.")

        # NOTE: Tool Calling is the easiest way to ensure structured outputs that is supported by multiple providers
        # https://platform.openai.com/docs/guides/function-calling
        # https://platform.openai.com/docs/guides/structured-outputs
        # https://docs.anthropic.com/en/docs/build-with-claude/tool-use#json-mode
        # https://docs.mistral.ai/capabilities/function_calling/
        # https://ollama.com/blog/tool-support
        # https://ollama.com/blog/structured-outputs

        if "anthropic" in self.model:
            tool = anthropic_pydantic_function_tool(self.response_validator)
            name = tool["name"]
            tools_params = {
                "tools": [anthropic_pydantic_function_tool(self.response_validator)],
                "tool_choice": {"type": "tool", "name": name},
            }
        else:
            tool = openai.pydantic_function_tool(self.response_validator)
            name = tool["function"]["name"]
            tools_params = {
                "tools": [tool],
                "tool_choice": {"type": "function", "function": {"name": name}},
            }

        return params | tools_params

    @property
    def response_validator(self) -> Type[BaseModel]:
        """Compiled regex pattern used to validate responses."""
        return self._response_validator

    @response_validator.setter
    def response_validator(self, response_validator: Type[BaseModel]):
        self._response_validator = response_validator

    @override
    def _validate_content(self, response: str) -> BaseModel:
        """Validate the model's response."""
        logger.warning("Expected a tool call but received a content response.")
        logger.debug("Validating response against response_validator Pydantic model.")
        return self.response_validator.model_validate(json_repair.loads(response))

    @override
    def _repair_response(self, response_content: str, exception: str) -> Conversation:
        """Render Conversation containing instructions to attempt to fix the validation error."""
        messages = [
            Message(role="assistant", content=response_content),
            Message(
                role="user",
                content=textwrap.dedent(
                    f"""
                    Received the following exception while validating the previous message:
                    {exception}

                    Update your response to ensure conforms the json schema.

                    <schema>
                    {self.response_validator.model_json_schema()}
                    </schema>
                    """.strip()
                ),
            ),
        ]
        return Conversation(messages=messages)

    def _handle_tool_call(
        self, conversation: Conversation, tool_call: ChatCompletionMessageToolCall, repair: int = 0
    ) -> str | BaseModel | ToolMessage:
        """Handle the tool call."""
        function = tool_call.function

        try:
            return self._validate_tool(name=function.name, arguments=function.arguments)
        except ValidationError as e:
            repair_msgs = self._repair_tool(tool_call, str(e))

            if repair > self.max_repair_attempts:  # Max 2 attempts (original, repair)
                raise CallerValidationError("Max repair attempts reached.") from e
                # logger.warning(
                #     f"Max repair attempts reached and could not validate tool_call for function {function.name} with arguments {function.arguments}. Returning failed tool call"
                # )
                # return ToolMessage(tool_call_id=tool_call.id, content=function.model_dump_json())

            if not repair_msgs:
                raise CallerValidationError(
                    f"{self.__class__.__name__}._render_tool() did not provide instructions for repair retry."
                ) from e
                # logger.warning(
                #     f"{self.__class__.__name__}._render_tool() did not provide instructions for repair retry, returning failed content."
                # )
                # return ToolMessage(tool_call_id=tool_call.id, content=function.model_dump_json())
            else:
                logger.debug(f"Attempting repair for exception raised during tool call validation: {e}")
                conversation.messages.extend(repair_msgs.messages)
                return self._handle_response(
                    conversation=conversation,
                    response=self._chat_completions_create(conversation),
                    repair=repair + 1,
                )

        except Exception as e:
            raise CallerValidationError(
                f"Unexpected Exception while validating tool_call for function {function.name} with arguments {function.arguments}"
            ) from e

    def _validate_tool(self, name: str, arguments: str) -> BaseModel:
        """Validate the model's tool call."""
        logger.debug(f"Validating tool_call response against response_validator Pydantic model ({name}).")
        return self.response_validator.model_validate(json_repair.loads(arguments))

    @override
    def _repair_tool(self, tool_call: ChatCompletionMessageToolCall, exception: str) -> Conversation:
        """Render messages array containing instructions to attempt to fix the validation error."""
        name = tool_call.function.name

        messages = [
            Message(role="assistant", content=tool_call.function.model_dump_json()),
            Message(
                role="user",
                content=textwrap.dedent(
                    f"""
                    Received the following exception while validating function call:
                    {exception}

                    Update your response to ensure conforms the json schema for function {name}:

                    <schema>
                    {self.response_validator.model_json_schema()}
                    </schema>
                    """.strip()
                ),
            ),
        ]
        return Conversation(messages=messages)


class ToolCaller(BaseCaller):
    """Provides client and standard parameters used for each call."""

    def __init__(
        self,
        client: Client,
        model: str,
        prompt: Prompt,
        toolbox: list[BaseCaller | CallableWithSignature],
        request_params: dict[str, JSON] | None = None,
        max_repair_attempts: int = 2,
        auto_invoke: bool = False,
    ):
        self.client = client
        self.model = model
        self.prompt = prompt
        self.toolbox = toolbox
        self.max_repair_attempts = max_repair_attempts
        self.auto_invoke = auto_invoke

        # must come last since logic depends on other properties
        self.request_params = request_params

    @property
    def toolbox(self) -> dict[str, BaseCaller | CallableWithSignature]:
        """Tools available to the Agent."""
        return self._toolbox

    @toolbox.setter
    def toolbox(self, toolbox: list[BaseCaller | CallableWithSignature]):
        self._toolbox = {tool.signature().__name__: tool for tool in toolbox}

    @property
    def auto_invoke(self) -> bool:
        """Boolean flag determining whether to automatically invoke the function call or just return the params."""
        return self._auto_invoke

    @auto_invoke.setter
    def auto_invoke(self, auto_invoke: bool):
        self._auto_invoke = auto_invoke

    @override
    def _make_request_params(self, request_params: dict[str, JSON] | None) -> dict[str, JSON]:
        """Construct the request parameters."""
        params = request_params or {}
        if "model" in params:
            raise ValueError("'model' should be set separately and not included in 'request_params'.")

        if "anthropic" in self.model:
            tools_params = {
                "tools": [anthropic_pydantic_function_tool(tool.signature()) for _name, tool in self.toolbox.items()],
                "tool_choice": {"type": "auto"},
            }

        else:
            tools_params = {
                "tools": [openai.pydantic_function_tool(tool.signature()) for _name, tool in self.toolbox.items()],
                "tool_choice": "auto",
            }

        return params | tools_params

    def _handle_tool_call(
        self, conversation: Conversation, tool_call: ChatCompletionMessageToolCall, repair: int = 0
    ) -> str | BaseModel | ToolMessage:
        """Handle the tool call."""
        function = tool_call.function

        try:
            validated = self._validate_tool(name=function.name, arguments=function.arguments)
        except KeyError:
            logger.exception(f"Tool {function.name} does not exist in the toolbox.")
            raise
        except ValidationError as e:
            repair_msgs = self._repair_tool(tool_call, str(e))

            if repair > self.max_repair_attempts:  # Max 2 attempts (original, repair)
                raise CallerValidationError("Max repair attempts reached.") from e
                # logger.warning(
                #     f"Max repair attempts reached and could not validate tool_call for function {function.name} with arguments {function.arguments}. Returning failed tool call"
                # )
                # return ToolMessage(tool_call_id=tool_call.id, content=function.model_dump_json())

            if not repair_msgs:
                raise CallerValidationError(
                    f"{self.__class__.__name__}._render_tool() did not provide instructions for repair retry."
                ) from e
                # logger.warning(
                #     f"{self.__class__.__name__}._render_tool() did not provide instructions for repair retry, returning failed content."
                # )
                # return ToolMessage(tool_call_id=tool_call.id, content=function.model_dump_json())
            else:
                logger.debug(f"Attempting repair for exception raised during tool call validation: {e}")
                conversation.messages.extend(repair_msgs.messages)
                return self._handle_response(
                    conversation=conversation,
                    response=self._chat_completions_create(conversation),
                    repair=repair + 1,
                )

        except Exception as e:
            raise CallerValidationError(
                f"Unexpected Exception while validating tool_call for function {function.name} with arguments {function.arguments}"
            ) from e

        if self.auto_invoke:
            try:
                result = self.toolbox[function.name](**validated.model_dump())
            except Exception:
                logger.exception(f"Unexpected Exception while invoking function {function.name}({function.arguments})")
                raise
            return ToolMessage(
                tool_call_id=tool_call.id,
                content=result if isinstance(result, str) else json.dumps(result),
            )
        else:
            return validated

    def _validate_tool(self, name: str, arguments: str) -> BaseModel:
        """Validate the model's tool call."""
        logger.debug("Validating tool call against tool signature.")
        return self.toolbox[name].signature().model_validate(json_repair.loads(arguments))

    @override
    def _repair_tool(self, tool_call: ChatCompletionMessageToolCall, exception: str) -> Conversation:
        """Render messages array containing instructions to attempt to fix the validation error."""
        name = tool_call.function.name

        messages = [
            Message(role="assistant", content=json.dumps(tool_call.function)),
            Message(
                role="user",
                content=textwrap.dedent(
                    f"""
                    Received the following exception while validating function call:
                    {exception}

                    Update your response to ensure conforms the json schema for function {name}:

                    <schema>
                    {self.toolbox[name].signature().model_json_schema()}
                    </schema>
                    """.strip()
                ),
            ),
        ]
        return Conversation(messages=messages)
