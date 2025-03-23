"""Components for composable LLM calls.

A Caller is the basic structure that wraps all logic required for LLM call-and-response.

A Caller executes LLM requests with enhanced response validation and automatic error recovery.
It manages message rendering, performs API calls with a specified client,
and validates responses using associated handlers (which may include tool execution).
"""

from __future__ import annotations

import json
import logging
from string import Template as StringTemplate
from typing import Any, Generic, Literal, Pattern, Type, cast, get_type_hints

from jinja2 import StrictUndefined, Template as JinjaTemplate
from pydantic import BaseModel, Field, create_model
from typing_extensions import override, runtime_checkable

from aisuite import Client

from .base import CallableReturnType, CallableWithSignature
from .exceptions import ValidationError
from .handler import ResponseHandler, ToolHandler
from .tool import Tool
from .validator import PassthroughValidator, PydanticValidator, RegexValidator, ToolValidator
from ..types_.base import JSON
from ..types_.core import Conversation, SystemMessage, UserMessage
from ..types_.openai_compat import ChatCompletion, convert_response
from ..utilities import to_snake_case

logger = logging.getLogger(__name__)


class Caller(Generic[CallableReturnType], CallableWithSignature[CallableReturnType]):
    """Executes LLM requests with validation and automated error recovery.

    Manages the full lifecycle of LLM interactions including:
    - Constructing conversation messages from prompt templates and inputs
    - Executing API requests and converting responses
    - Optionally invoking tools based on the response
    - Validating responses
    - Performing automatic repair attempts in case of validation errors

    Attributes
    ----------
        client (Client): client for OpenAI-compatible API.
        model (str): Identifier for the model (e.g., "gpt-4o").
        name (str): Name of the caller.
        description (str): Description of the caller's purpose.
        instruction (Template | str): The instruction template for system messages.
        input_template (Template | None): Optional user template. If provided, the input string will be passed to the template using the `input` template variable.
        input_params (BaseModel):Pydantic baseModel that defines expectations for template inputs.
        output_validator (BaseModel | Pattern | None): Validator for output messages.
        request_params (dict[str, JSON]): Additional API request parameters.
        max_repair_attempts (int): Maximum allowed repair iterations.
    """

    def __init__(
        self,
        client: Client,
        model: str,
        # name: str | None = None,
        description: str,
        instruction: StringTemplate | JinjaTemplate | str,
        input_template: StringTemplate | JinjaTemplate | None = None,
        input_params: Type[BaseModel] | None = None,
        request_params: dict[str, JSON] | None = None,
        tools: list[CallableWithSignature | Tool] | None = None,
        auto_invoke: bool = False,
        output_validator: Type[BaseModel] | Pattern | None = None,
        max_repair_attempts: int = 2,
    ):
        self.name = to_snake_case(self.__class__.__name__)
        self.description = description

        # Using templates requires input_params
        if (
            isinstance(instruction, (StringTemplate, JinjaTemplate))
            or isinstance(input_template, (StringTemplate, JinjaTemplate))
        ) and not input_params:
            raise ValueError("Input parameter specification is required when templates are used.")

        self.instruction = instruction
        self.input_template = input_template
        self.input_params = input_params

        self.max_repair_attempts = max_repair_attempts
        self.tool_handler = self._create_tool_handler(
            tools=tools,
            auto_invoke=auto_invoke,
            max_repair_attempts=max_repair_attempts,
        )
        self.response_handler = self._create_response_handler(
            output_validator=output_validator,
            max_repair_attempts=max_repair_attempts,
        )

        self._signature = self._define_signature()
        self._schema = cast(JSON, self.signature.model_json_schema())
        self.returns: Type[CallableReturnType] | None = get_type_hints(
            self.__class__.__call__, self.__class__.__call__.__globals__
        ).get("return")

        self.client = client
        self.model = model
        self.request_params = self._make_request_params(request_params)

    def _create_response_handler(
        self, output_validator: Type[BaseModel] | Pattern | None = None, max_repair_attempts: int = 2
    ) -> ResponseHandler:
        """Create the appropriate handler based on the output_validator type."""
        if output_validator is None:
            return ResponseHandler(PassthroughValidator(), max_repair_attempts=max_repair_attempts)

        elif isinstance(output_validator, type) and issubclass(output_validator, BaseModel):
            return ResponseHandler(PydanticValidator(output_validator), max_repair_attempts=max_repair_attempts)

        elif isinstance(output_validator, Pattern):
            return ResponseHandler(RegexValidator(output_validator), max_repair_attempts=max_repair_attempts)

        else:
            raise ValueError(f"Unsupported output_validator type: {type(output_validator)}")

    def _create_tool_handler(
        self,
        tools: list[CallableWithSignature | Tool] | None = None,
        auto_invoke: bool = False,
        max_repair_attempts: int = 2,
    ) -> ToolHandler | None:
        if tools is None:
            return None
        else:
            return ToolHandler(
                validator=ToolValidator(toolbox=tools),
                auto_invoke=auto_invoke,
                max_repair_attempts=max_repair_attempts,
            )

    def _define_signature(self) -> Type[BaseModel]:
        """Define the signature for the caller based on the input_params."""
        if self.input_params:
            fields = {}
            for field_name, field_info in self.input_params.model_fields.items():
                if field_info.description:
                    fields[field_name] = (field_info.annotation, Field(..., description=field_info.description))
                else:
                    fields[field_name] = (field_info.annotation, ...)

            return create_model(
                to_snake_case(self.name),
                __doc__=self.description,
                **fields,
                __config__={"arbitrary_types_allowed": True},
            )

        return create_model(
            to_snake_case(self.name),
            __doc__=self.description,
            __config__={"arbitrary_types_allowed": True},
        )

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
    def signature(self) -> Type[BaseModel]:
        """Get the Pydantic model for the template signature."""
        return self._signature

    @property
    def schema(self) -> JSON:
        """Get the OpenAPI-compatible schema."""
        return self._schema

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

        # TODO: what if both pydantic validator and tool handler?

        # Add structured format if using PydanticValidator
        if isinstance(self.response_handler.validator, PydanticValidator):
            params |= self._make_structured_params()

        # Add tool configuration if using ToolHandler
        if self.tool_handler:
            params |= self._make_tool_params()

        return params

    def _tools(self) -> list[CallableWithSignature] | None:
        """Extract tools from tool_handler if available."""
        if self.tool_handler:
            return list(self.tool_handler.validator.toolbox.values())
        return None

    def _make_structured_params(self) -> dict[str, JSON]:
        """Generate provider-specific structured response parameters."""
        from openai import pydantic_function_tool as openai_pydantic_function_tool

        if not isinstance(self.response_handler.validator, PydanticValidator):
            raise TypeError("Handler must use PydanticValidator for structured response handling.")

        # Hack function-calling for models that do not support structured outputs
        tool = openai_pydantic_function_tool(self.response_handler.validator.model)
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

    def _make_tool_params(self) -> dict[str, JSON]:
        """Generate provider-specific tool configuration parameters."""
        from openai import pydantic_function_tool as openai_pydantic_function_tool

        tools = [openai_pydantic_function_tool(t.signature) for t in self._tools()]
        tools = cast(JSON, tools)

        configs = {
            "anthropic": {"tools": tools, "tool_choice": {"type": "auto"}},
            "openai": {"tools": tools, "tool_choice": "auto"},
            # Add other providers as needed
        }
        provider = self.model.split(":")[0]
        return configs.get(provider, configs["openai"])

    def render(
        self,
        input: str,  # NOQA: A002
        conversation_history: Conversation | None = None,
        state: dict[str, Any] | None = None,
    ) -> Conversation:
        """Render a complete Conversation using the provided variables.

        Parameters
        ----------
        input : str
            Input string to be processed by the LLM.
        conversation_history : Conversation | None, optional
            Existing conversation history to append to, by default None
        state : dict[str, Any], optional
            Additional state variables that can be used in rendering, by default None

        Returns
        -------
        Conversation
            A Conversation object with rendered messages.
        """
        state_vars = {} if state is None else state.copy()

        # Validate input if validator is provided
        if self.input_params:
            _ = self.input_params(input=input, **state_vars)

        messages = []

        # Add system message with instruction
        if isinstance(self.instruction, str):
            instruction_content = self.instruction
        elif isinstance(self.instruction, StringTemplate):
            instruction_content = self.instruction.substitute(input=input, **state_vars)
        elif isinstance(self.instruction, JinjaTemplate):
            instruction_content = self.instruction.render(input=input, **state_vars)
        else:
            raise TypeError(f"Unsupported instruction type: {type(self.instruction)}")

        messages.append(SystemMessage(content=instruction_content))

        # Add user message with template if provided
        if self.input_template is None:
            messages.append(UserMessage(content=input))
        else:
            if isinstance(self.input_template, StringTemplate):
                input_content = self.input_template.substitute(input=input, **state_vars)
            elif isinstance(self.input_template, JinjaTemplate):
                input_content = self.input_template.render(input=input, **state_vars)
            else:
                raise TypeError(f"Unsupported template type: {type(self.input_template)}")

            messages.append(UserMessage(content=input_content))

        if conversation_history:
            conversation = Conversation(
                messages=[
                    *conversation_history.messages,
                    *messages,
                ]
            )
        else:
            conversation = Conversation(messages=messages)

        if not conversation:
            raise ValueError("Rendered empty Conversation.")

        if not any(message.role == "system" for message in conversation.messages):
            raise ValueError("Conversation must have at least one 'system' message.")

        if not any(message.role == "user" for message in conversation.messages):
            raise ValueError("Conversation must have at least one 'user' message.")

        return conversation

    def __call__(
        self,
        *,
        input: str,  # NOQA: A002
        conversation_history: Conversation | None = None,
        state: dict[str, Any] | None = None,
    ) -> CallableReturnType | BaseModel | str:
        """Execute the LLM API call with conversation rendering and validation.

        Renders the conversation using the provided template variables and merges with any existing conversation
        history. Validates that required 'system' and 'user' messages exist.
        It then sends the conversation to the LLM endpoint and processes the response via the handler,
        including repair attempts if validation fails.

        Parameters
        ----------
        input : str
            Input string to be processed by the LLM.
        conversation_history : Conversation | None, optional
            Existing conversation history to append to, by default None
        state : dict[str, Any], optional
            Additional state variables that can be used in rendering, by default None

        Returns
        -------
            The processed response which may be of type CallableReturnType, BaseModel, or str.

        Raises
        ------
            ValueError: If the conversation is empty or missing required 'system' or 'user' messages.
        """
        conversation = self.render(input=input, conversation_history=conversation_history, state=state)

        response = self._chat_completions_create(conversation)
        return self._handle_with_repair(conversation=conversation, response=response)

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
        target_handler = "tool" if self.tool_handler and response.choices[0].message.tool_calls else "response"
        try:
            if target_handler == "tool":
                return self.tool_handler.process(response)
            else:
                return self.response_handler.process(response)

        except Exception as e:
            if repair_attempt >= self.max_repair_attempts:
                raise ValidationError("Max repair attempts reached") from e

            logger.debug(f"Repair {repair_attempt} after error handling response {e}")
            msg = response.choices[0].message
            if target_handler == "tool":
                repair_prompt = self.tool_handler.repair(msg, str(e))
            else:
                repair_prompt = self.response_handler.repair(msg, str(e))

            if not repair_prompt:
                raise ValidationError("No repair instructions available") from e

            conversation.messages.extend(repair_prompt.messages)
            new_response = self._chat_completions_create(conversation)
            return self._handle_with_repair(conversation, new_response, repair_attempt + 1)
