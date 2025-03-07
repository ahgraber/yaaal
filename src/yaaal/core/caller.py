"""Components for composable LLM calls.

A Caller executes LLM requests with enhanced response validation and automatic error recovery.
It manages message construction via a ConversationTemplate, performs API calls with a specified client,
and validates responses using associated handlers (which may include tool execution).

Use factory functions (create_chat_caller, create_structured_caller, create_tool_caller) to instantiate
Callers with specific behaviors.
"""

from __future__ import annotations

import logging
from typing import Any, Generic, Literal, Type, cast, get_type_hints

from pydantic import BaseModel
from typing_extensions import override, runtime_checkable

from aisuite import Client
from openai import pydantic_function_tool as openai_pydantic_function_tool

from .base import CallableReturnType, CallableWithSignature
from .exceptions import ValidationError
from .handler import ResponseHandler, ToolHandler  # , CompositeHandler
from .template import ConversationTemplate
from .tool import anthropic_pydantic_function_tool
from .validator import PassthroughValidator, PydanticValidator, RegexValidator, ToolValidator
from ..types_.base import JSON
from ..types_.core import Conversation, Message, UserMessage
from ..types_.openai_compat import ChatCompletion, convert_response

logger = logging.getLogger(__name__)


class Caller(Generic[CallableReturnType], CallableWithSignature[CallableReturnType]):
    """Executes LLM requests with validation and automated error recovery.

    Manages the full lifecycle of LLM interactions including:
    - Constructing conversation messages from prompts
    - Executing API requests and converting responses
    - Validating responses using the provided handler
    - Performing automatic repair attempts in case of validation errors
    - Optionally invoking tools based on the response

    Attributes
    ----------
        client (Client): OpenAI-compatible API client.
        model (str): Identifier for the model (e.g., "gpt-4").
        conversation_template (ConversationTemplate): Template to generate conversation messages.
        handler (ResponseHandler | ToolHandler | CompositeHandler): Processes and validates responses.
        request_params (dict[str, JSON]): Additional API request parameters.
        max_repair_attempts (int): Maximum allowed repair iterations.
    """

    def __init__(
        self,
        client: Client,
        model: str,
        conversation_template: ConversationTemplate,
        handler: ResponseHandler[CallableReturnType] | ToolHandler,
        request_params: dict[str, JSON] | None = None,
        max_repair_attempts: int = 2,
    ):
        self.client = client
        self.model = model
        self.conversation_template = conversation_template
        self.handler = handler
        self.max_repair_attempts = max_repair_attempts

        self.signature = self.conversation_template.signature
        self.schema = self.signature.model_json_schema()
        self.returns: Type[CallableReturnType] | None = get_type_hints(
            self.__class__.__call__, self.__class__.__call__.__globals__
        ).get("return")

        self.request_params = self._make_request_params(request_params)

    @property
    def model(self) -> str:
        """The model name identifier in the format 'provider:identifier'."""
        return self._model

    @model.setter
    def model(self, model: str):
        """Set and validate the model identifier.

        Args:
            value (str): Model identifier (e.g., 'openai:gpt-4' or 'anthropic:claude')

        Raises
        ------
            ValueError: If model string is invalid or missing provider prefix
        """
        if not model or not isinstance(model, str):
            raise ValueError("Model must be a non-empty string")
        if ":" not in model:
            raise ValueError(
                "Model must be in format 'provider:identifier' (e.g., 'openai:gpt-4o' or 'anthropic:claude-3-5-haiku-latest')"
            )
        self._model = model

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

        # Add structured format if using PydanticValidator
        if isinstance(self.handler, ResponseHandler) and isinstance(self.handler.validator, PydanticValidator):
            params |= self._make_structured_params()

        # Add tool configuration if using ToolHandler
        if toolbox := self._get_toolbox():
            params |= self._make_tool_params(toolbox)

        return params

    def _get_toolbox(self) -> list[CallableWithSignature] | None:
        """Extract toolbox from handler if available."""
        if isinstance(self.handler, ToolHandler):
            return list(self.handler.validator.toolbox.values())
        # if isinstance(self.handler, CompositeHandler) and isinstance(self.handler.tool_handler, ToolHandler):
        #     return list(self.handler.tool_handler.validator.toolbox.values())
        return None

    def _make_structured_params(self) -> dict[str, JSON]:
        """Generate provider-specific structured response parameters."""
        if not isinstance(self.handler.validator, PydanticValidator):
            raise TypeError("Handler must use PydanticValidator for structured response handling.")

        # Hack function-calling for models that do not support structured outputs
        tool = openai_pydantic_function_tool(self.handler.validator.model)
        configs = {
            "anthropic": {
                "tools": [cast(JSON, tool)],
                "tool_choice": {"type": "tool", "name": tool["function"]["name"]},
            },
            "openai": {"response_format": cast(JSON, {"type": "json_object"})},
            # Add other providers as needed
        }
        provider = self.model.split(":")[0]
        return configs.get(provider, configs["openai"])

    def _make_tool_params(self, toolbox: list[CallableWithSignature]) -> dict[str, JSON]:
        """Generate provider-specific tool configuration parameters."""
        tools = [openai_pydantic_function_tool(t.signature) for t in toolbox]
        tools = cast(JSON, tools)

        configs = {
            "anthropic": {"tools": tools, "tool_choice": {"type": "auto"}},
            "openai": {"tools": tools, "tool_choice": "auto"},
            # Add other providers as needed
        }
        provider = self.model.split(":")[0]
        return configs.get(provider, configs["openai"])

    def __call__(
        self,
        *,
        conversation_history: Conversation | None = None,
        **template_vars: dict[str, Any],
    ) -> CallableReturnType | BaseModel | str:
        """Execute the LLM API call with conversation rendering and validation.

        Renders the conversation using the provided template variables and merges with any existing conversation
        history. Validates that required 'system' and 'user' messages exist.
        It then sends the conversation to the LLM endpoint and processes the response via the handler,
        including repair attempts if validation fails.

        Returns
        -------
            The processed response which may be of type CallableReturnType, BaseModel, or str.

        Raises
        ------
            ValueError: If the conversation is empty or missing required 'system' or 'user' messages.
        """
        # Render the conversation using the ConversationTemplate
        rendered = self.conversation_template.render(template_vars)

        if conversation_history:
            conversation_history.messages.extend(rendered.messages)
        else:
            conversation_history = rendered

        if not conversation_history:
            raise ValueError("Conversation cannot be empty.")

        if not any(message.role == "system" for message in conversation_history.messages):
            raise ValueError("Conversation must have at least one 'system' message.")

        if not any(message.role == "user" for message in conversation_history.messages):
            raise ValueError("Conversation must have at least one 'user' message.")

        response = self._chat_completions_create(conversation_history)
        return self._handle_with_repair(conversation=conversation_history, response=response)

    def _chat_completions_create(self, conversation: Conversation) -> ChatCompletion:
        """Call the LLM chat endpoint and convert its response.

        Args:
            conversation (Conversation): The conversation to send to the LLM.

        Returns
        -------
            ChatCompletion: The response converted to a standard format.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=conversation.model_dump()["messages"],
            **self.request_params,
        )
        return convert_response(response)

    def _handle_with_repair(
        self, conversation: Conversation, response: ChatCompletion, repair_attempt: int = 0
    ) -> CallableReturnType | BaseModel | str:
        """Process the LLM response with potential repair attempts.

        If the handler raises an exception during processing, the method will attempt to repair the conversation
        by appending repair instructions and retrying the API call. The process repeats until the response is valid
        or the maximum number of repair attempts is reached.

        Args:
            conversation (Conversation): The conversation used in the API call.
            response (ChatCompletion): The initial response from the LLM.
            repair_attempt (int, optional): The current repair attempt count. Defaults to 0.

        Returns
        -------
            The processed response after successful validation.

        Raises
        ------
            ValidationError: If the maximum repair attempts are exceeded or no repair instructions are available.
        """
        try:
            return self.handler.process(response)
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


# Factory functions
def create_chat_caller(
    client: Client,
    model: str,
    conversation_template: ConversationTemplate,
    request_params: dict[str, JSON] | None = None,
) -> Caller:
    """Create a basic Caller for chat without extra response validation.

    Uses a passthrough validator to simply return the LLM response content.

    Returns
    -------
        Caller: Configured for basic chat interactions.

    Raises
    ------
        ValueError: If model or template is invalid
    """
    handler = ResponseHandler(PassthroughValidator())
    return Caller(client, model, conversation_template, handler, request_params)


def create_structured_caller(
    client: Client,
    model: str,
    conversation_template: ConversationTemplate,
    response_model: Type[BaseModel],
    request_params: dict[str, JSON] | None = None,
) -> Caller:
    """Create a Caller for structured responses with validation via a Pydantic model.

    The response is validated and converted to the provided Pydantic model.

    Returns
    -------
        Caller: Configured for structured output.

    Raises
    ------
        ValueError: If model, template or response_model is invalid
    """
    if not response_model or not issubclass(response_model, BaseModel):
        raise ValueError("Response model must be a Pydantic model class")

    handler = ResponseHandler(PydanticValidator(response_model))
    return Caller(client, model, conversation_template, handler, request_params)


def create_tool_caller(
    client: Client,
    model: str,
    conversation_template: ConversationTemplate,
    toolbox: list[CallableWithSignature],
    request_params: dict[str, JSON] | None = None,
    auto_invoke: bool = False,
) -> Caller:
    """Create a Caller configured for tool invocation.

    Attaches a ToolHandler to allow optional execution of tools based on LLM response.

    Returns
    -------
        Caller: Configured for tool-using interactions.

    Raises
    ------
        ValueError: If model, template or toolbox is invalid
    """
    if not toolbox:
        raise ValueError("Toolbox cannot be empty")

    # Validate all tools have proper signatures
    for tool in toolbox:
        if not isinstance(tool, CallableWithSignature):
            raise TypeError(f"Tool {tool} must implement CallableWithSignature")
        if not tool.signature.__doc__:
            logger.warning(f"Tool {tool.signature.__name__} missing docstring - this may affect LLM usage")

    handler = ToolHandler(ToolValidator(toolbox), auto_invoke)
    return Caller(client, model, conversation_template, handler, request_params)
