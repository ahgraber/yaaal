"""Components for composable LLM calls.

A Caller is the basic structure that wraps all logic required for LLM call-and-response.

A Caller executes LLM requests with enhanced response validation and automatic error recovery.
It manages message rendering, performs API calls with a specified client,
and validates responses using associated handlers (which may include tool execution).
"""

from __future__ import annotations

import inspect
import json
import logging
from string import Template as StringTemplate
from typing import Any, Generic, Literal, Pattern, Type, cast, get_type_hints

from jinja2 import StrictUndefined, Template as JinjaTemplate
from pydantic import BaseModel, Field, create_model
from typing_extensions import override, runtime_checkable

from aisuite import Client

from .base import CallableReturnType, CallableWithSchema
from .exceptions import ValidationError
from .handler import ResponseHandler, ToolHandler
from .tool import Tool, function_schema, pydantic_to_schema
from .validator import PassthroughValidator, PydanticValidator, RegexValidator, ToolValidator
from ..types_.base import JSON
from ..types_.core import Conversation, FunctionSchema, SystemMessage, UserMessage
from ..types_.openai_compat import ChatCompletion, convert_response
from ..utilities import to_snake_case

logger = logging.getLogger(__name__)


class Caller(Generic[CallableReturnType], CallableWithSchema[CallableReturnType]):
    """Protocol for callables that provide validated LLM interactions.

    This class implements the CallableWithSchema protocol to provide a standard
    interface for LLM operations. It manages the full lifecycle of LLM interactions
    including prompt rendering, API calls, response validation, and error recovery.
    """

    def __init__(
        self,
        client: Client,
        model: str,
        description: str,
        instruction: StringTemplate | JinjaTemplate | str,
        input_template: StringTemplate | JinjaTemplate | None = None,
        input_params: Type[BaseModel] | None = None,
        request_params: dict[str, JSON] | None = None,
        tools: list[CallableWithSchema] | None = None,
        auto_invoke: bool = False,
        output_validator: Type[BaseModel] | Pattern | None = None,
        max_repair_attempts: int = 2,
    ):
        """Initialize a Caller with LLM configuration and validation settings.

        Parameters
        ----------
        client : Client
            OpenAI-compatible API client
        model : str
            Model identifier (e.g. 'openai:gpt-4')
        description : str
            Description of the caller's purpose
        instruction : StringTemplate | JinjaTemplate | str
            Template for system messages
        input_template : StringTemplate | JinjaTemplate | None, optional
            Template for user messages, by default None
        input_params : Type[BaseModel] | None, optional
            Input validation model, by default None
        request_params : dict[str, JSON] | None, optional
            Additional API parameters, by default None
        tools : list[CallableWithSchema | Tool] | None, optional
            Available tools for function calling, by default None
        auto_invoke : bool, optional
            Whether to auto-invoke tools, by default False
        output_validator : Type[BaseModel] | Pattern | None, optional
            Validator for responses, by default None
        max_repair_attempts : int, optional
            Maximum repair attempts, by default 2

        Raises
        ------
        ValueError
            If templates are used without input_params
        """
        # Initialize remaining attributes
        self.name = to_snake_case(self.__class__.__name__)
        self.description = description

        # Using templates requires input_params
        if (
            isinstance(instruction, (StringTemplate, JinjaTemplate))
            or isinstance(input_template, (StringTemplate, JinjaTemplate))
        ) and not input_params:
            raise ValueError("Input parameter specification is required when templates are used.")

        self.instruction = instruction

        # set defaults for when input_template and input_params are none
        if input_template is None and input_params is None:
            self.input_template = StringTemplate("$input")
            self.input_params = create_model(
                self.name,
                __doc__=description,
                input=(str, ...),
            )
        else:
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

        # Create function schema
        self.function_schema = FunctionSchema(
            pydantic_model=input_params
            or create_model(
                self.name,
                __doc__=description,
                input=(str, ...),
            ),
            json_schema=pydantic_to_schema(self.input_params) if self.input_params else {},
            signature=inspect.signature(self.__call__),
        )

        # Set returns from type hints
        self.returns = get_type_hints(
            self.__class__.__call__,
            # self.__class__.__call__.__globals__,
        ).get("return", Any)

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
        tools: list[CallableWithSchema | Tool] | None = None,
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

    @property
    def model(self) -> str:
        """Get the model identifier in 'provider:name' format."""
        return self._model

    @model.setter
    def model(self, model: str):
        """Set and validate the model identifier.

        Parameters
        ----------
        model : str
            Model identifier (e.g. 'openai:gpt-4')

        Raises
        ------
        ValueError
            If model string is invalid or missing provider prefix
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

        # TODO: what if both pydantic validator and tool handler?

        # Add structured format if using PydanticValidator
        if isinstance(self.response_handler.validator, PydanticValidator):
            params |= self._make_structured_params()

        # Add tool configuration if using ToolHandler
        if self.tool_handler:
            params |= self._make_tool_params()

        return params

    def _tools(self) -> list[CallableWithSchema] | None:
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
        tool = openai_pydantic_function_tool(self.response_handler.validator.pydantic_model)
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

        tools = [openai_pydantic_function_tool(t.function_schema.pydantic_model) for t in self._tools()]
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

    @override
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
        CallableReturnType | BaseModel | str
            The validated response from the LLM

        Raises
        ------
        ValidationError
            If response validation fails after all repair attempts
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
