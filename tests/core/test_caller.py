from __future__ import annotations

import json
from string import Template as StringTemplate
from typing import Any
from unittest.mock import Mock, create_autospec

from jinja2 import Template as JinjaTemplate
from pydantic import BaseModel, Field, create_model
import pytest

from aisuite import Client
from aisuite.framework import ChatCompletionResponse

from yaaal.core.caller import Caller
from yaaal.core.exceptions import ValidationError
from yaaal.core.handler import ResponseHandler, ToolHandler
from yaaal.core.tool import tool
from yaaal.core.validator import PassthroughValidator, PydanticValidator, ToolValidator
from yaaal.types_.core import AssistantMessage, Conversation, Message, UserMessage
from yaaal.types_.openai_compat import (
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallFunction,
)


# --- Test Fixtures ---
@pytest.fixture
def mock_client():
    """Mock API client"""
    return create_autospec(Client)


@pytest.fixture
def system_instruction():
    """Basic system instruction for testing"""
    return "You are a test assistant"


@pytest.fixture
def user_template():
    """Basic user template for testing"""
    return StringTemplate("Process $input")


@pytest.fixture
def input_params():
    """Basic input validator for testing"""

    class InputParams(BaseModel):
        input: str = Field(description="Input text")

    return InputParams


@pytest.fixture
def simple_response():
    """Basic text response from LLM"""
    return ChatCompletion(
        choices=[
            ChatCompletionChoice(
                finish_reason="stop", message=ChatCompletionMessage(role="assistant", content="test response")
            )
        ]
    )


@pytest.fixture
def mock_response_handler():
    """Mock response handler with tracking capabilities"""
    handler = Mock(spec=ResponseHandler)
    handler.process.return_value = "processed"
    handler.repair.return_value = Conversation(
        messages=[
            AssistantMessage(content="bad response"),
            UserMessage(content="please fix"),
        ]
    )
    return handler


@pytest.fixture
def mock_tool_handler():
    """Mock tool handler with tracking capabilities"""
    handler = Mock(spec=ToolHandler)
    handler.process.return_value = "processed"
    handler.repair.return_value = Conversation(
        messages=[
            AssistantMessage(content="bad response"),
            UserMessage(content="please fix"),
        ]
    )
    return handler


@pytest.fixture
def error_response():
    """Response with content filter/error"""
    return ChatCompletion(
        choices=[
            ChatCompletionChoice(
                finish_reason="content_filter",
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=None,  # Explicitly set tool_calls to None
                    refusal="Content was filtered",
                ),
            )
        ]
    )


# --- Unit Tests: Core Caller Functionality ---
class TestCallerCore:
    """Unit tests for core Caller functionality"""

    def test_templates_require_input_params(self, mock_client):
        """Test that using templates requires input_params."""
        with pytest.raises(ValueError, match="Input parameter specification is required when templates are used."):
            Caller(
                client=mock_client,
                model="provider:model",
                description="Test caller",
                instruction=StringTemplate("System message needs $vars"),
                input_template=None,
                input_params=None,
            )

        with pytest.raises(ValueError, match="Input parameter specification is required when templates are used."):
            Caller(
                client=mock_client,
                model="provider:model",
                description="Test caller",
                instruction="You are a helpful assistant.",
                input_template=JinjaTemplate("User message needs $vars"),
                input_params=None,
            )

    def test_model_validation(self, mock_client):
        """Test model name validation"""
        # Valid model format
        caller = Caller(
            client=mock_client,
            model="provider:model",
            description="Test caller",
            instruction="You are a test assistant",
        )
        assert caller.model == "provider:model"

        # Invalid model format
        with pytest.raises(ValueError, match="must be in format"):
            Caller(
                client=mock_client,
                model="invalid_model",
                description="Test caller",
                instruction="You are a test assistant",
            )

    def test_request_params_validation(self, mock_client):
        """Test request parameter validation"""
        # Valid params
        params = {"temperature": 0.7}
        caller = Caller(
            client=mock_client,
            model="provider:model",
            description="Test caller",
            instruction="You are a test assistant",
            request_params=params,
        )
        assert caller.request_params == params

        # Invalid params (model specified)
        with pytest.raises(ValueError, match="'model' should be set separately"):
            Caller(
                client=mock_client,
                model="provider:model",
                description="Test caller",
                instruction="You are a test assistant",
                request_params={"model": "other:model"},
            )

    def test_max_repair_attempts(self, mock_client, mock_response_handler, simple_response, system_instruction):
        """Test max repair attempts respected"""
        # Setup handler to always fail
        mock_response_handler.process.side_effect = ValueError("Always fail")

        # Setup client to always return a valid response
        mock_client.chat.completions.create.return_value = simple_response

        # Create a caller with the mock response handler
        caller = Caller(
            client=mock_client,
            model="provider:model",
            description="Test caller",
            instruction=system_instruction,
            max_repair_attempts=2,
        )

        # Replace the auto-created response handler with our mock
        caller.response_handler = mock_response_handler

        with pytest.raises(ValidationError, match="Max repair attempts reached"):
            caller(input="test")

        assert mock_response_handler.process.call_count == 3  # Initial + 2 repairs
        assert mock_response_handler.repair.call_count == 2  # Should attempt repair twice

    def test_provider_specific_params(self, mock_client, system_instruction):
        """Test provider-specific parameter generation"""

        class TestResponse(BaseModel):
            value: str

        # Test OpenAI format
        caller = Caller(
            client=mock_client,
            model="openai:gpt-4",
            description="Test caller",
            instruction=system_instruction,
            output_validator=TestResponse,
        )
        assert caller.request_params["response_format"] == {"type": "json_object"}

        # Test Anthropic format
        caller = Caller(
            client=mock_client,
            model="anthropic:fraude",
            description="Test caller",
            instruction=system_instruction,
            output_validator=TestResponse,
        )
        assert "tools" in caller.request_params


class TestCallerRender:
    def test_render_with_str_instruction_no_template(self, mock_client):
        # Caller with plain string instruction and no input_template
        caller = Caller(
            client=mock_client,
            model="provider:model",
            description="Test caller",
            instruction="System message: static",
        )
        conversation = caller.render(input="User input")
        # Expect system message from instruction and user message with given input
        assert len(conversation.messages) == 2
        assert conversation.messages[0].role == "system"
        assert conversation.messages[0].content == "System message: static"
        assert conversation.messages[1].role == "user"
        assert conversation.messages[1].content == "User input"

    def test_render_with_stringtemplate_instruction_and_input_template(self, mock_client):
        from string import Template as StringTemplate

        # Create caller with StringTemplate for both instruction and input_template
        instruction_template = StringTemplate("System: Process $input from $user")
        input_template = StringTemplate("User: $input with extra $statevar")

        # Define a dummy input_params model with additional variables
        class DummyParams(BaseModel):
            input: str
            user: str = ""
            statevar: str = ""

        caller = Caller(
            client=mock_client,
            model="provider:model",
            description="Test caller",
            instruction=instruction_template,
            input_template=input_template,
            input_params=DummyParams,
        )
        conversation = caller.render(input="hello", state={"user": "Alice", "statevar": "value"})
        expected_sys = "System: Process hello from Alice"
        expected_user = "User: hello with extra value"
        assert conversation.messages[0].role == "system"
        assert conversation.messages[0].content == expected_sys
        assert conversation.messages[1].role == "user"
        assert conversation.messages[1].content == expected_user

    def test_render_with_jinjatemplate_instruction_and_input_template(self, mock_client):
        # Create caller with Jinja templates for instruction and input_template
        instruction_template = JinjaTemplate("System: Process {{ input }} and {{ var }}")
        input_template = JinjaTemplate("User says: {{ input }} with {{ var }}")

        class DummyParams(BaseModel):
            input: str
            var: str

        caller = Caller(
            client=mock_client,
            model="provider:model",
            description="Test caller",
            instruction=instruction_template,
            input_template=input_template,
            input_params=DummyParams,
        )
        conversation = caller.render(input="greetings", state={"var": "data"})
        expected_sys = "System: Process greetings and data"
        expected_user = "User says: greetings with data"
        assert conversation.messages[0].role == "system"
        assert conversation.messages[0].content == expected_sys
        assert conversation.messages[1].role == "user"
        assert conversation.messages[1].content == expected_user

    def test_render_with_conversation_history(self, mock_client):
        # Test that render appends new messages to provided conversation_history
        from string import Template as StringTemplate

        # Use plain string instruction and no input_template
        caller = Caller(
            client=mock_client,
            model="provider:model",
            description="Test caller",
            instruction="System base",
        )
        # Create a conversation history with one user message
        history = Conversation(messages=[UserMessage(content="old message")])
        conversation = caller.render(input="new input", conversation_history=history)
        # Expect history messages plus two new rendered messages (system and user)
        assert len(conversation.messages) == len(history.messages) + 2
        # Verify history is preserved and new messages are appended
        assert conversation.messages[0].content == "old message"
        assert conversation.messages[-2].role == "system"
        assert conversation.messages[-2].content == "System base"
        assert conversation.messages[-1].role == "user"
        assert conversation.messages[-1].content == "new input"


class TestCallerHandlerIntegration:
    """Tests for Caller interaction with handlers"""

    def test_response_handler_processing(
        self, mock_client, mock_response_handler, simple_response, system_instruction
    ):
        """Test basic response handler processing flow"""
        # Setup client to return a valid response
        mock_client.chat.completions.create.return_value = simple_response

        # Create caller with mock handler
        caller = Caller(
            client=mock_client,
            model="provider:model",
            description="Test caller",
            instruction=system_instruction,
        )

        # Replace the auto-created response handler with our mock
        caller.response_handler = mock_response_handler

        # Call the caller
        result = caller(input="test")

        # Verify handler was used correctly
        mock_response_handler.process.assert_called_once()
        assert result == "processed"

    def test_tool_handler_processing(self, mock_client, mock_tool_handler, simple_response, system_instruction):
        """Test basic tool handler processing flow"""

        @tool
        def test_tool(x: int, y: int) -> str:
            """Test tool that requires two arguments"""
            return f"Processed {x} and {y}"

        # Setup client to return a valid response
        mock_client.chat.completions.create.return_value = mock_client.chat.completions.create.return_value = (
            ChatCompletion(
                choices=[
                    ChatCompletionChoice(
                        finish_reason="tool_calls",
                        message=ChatCompletionMessage(
                            role="assistant",
                            content=None,
                            tool_calls=[
                                ChatCompletionMessageToolCall(
                                    id="call_123",
                                    type="function",
                                    function=ChatCompletionMessageToolCallFunction(
                                        name="test_tool",
                                        arguments=json.dumps({"x": 42}),
                                    ),
                                )
                            ],
                        ),
                    )
                ]
            )
        )

        # Create caller with mock handler
        caller = Caller(
            client=mock_client,
            model="provider:model",
            description="Test caller",
            instruction=system_instruction,
            tools=[test_tool],
        )
        caller.tool_handler = mock_tool_handler

        _result = caller(input="test")
        mock_tool_handler.process.assert_called_once()

    def test_repair_mechanism(self, mock_client, mock_response_handler, simple_response, system_instruction):
        """Test repair flow when handler raises error"""
        # Setup handler to fail once then succeed
        mock_response_handler.process.side_effect = [ValueError("First call fails"), "fixed response"]

        # Setup client to return a valid response
        mock_client.chat.completions.create.return_value = simple_response

        # Create caller with mock handler
        caller = Caller(
            client=mock_client,
            model="provider:model",
            description="Test caller",
            instruction=system_instruction,
            max_repair_attempts=2,
        )

        # Replace the auto-created response handler with our mock
        caller.response_handler = mock_response_handler

        # Call should succeed after repair
        result = caller(input="test")

        # Verify repair was attempted and succeeded
        assert mock_response_handler.process.call_count == 2
        assert mock_response_handler.repair.call_count == 1
        assert result == "fixed response"

    def test_render_with_state(self, mock_client, system_instruction, input_params):
        """Test rendering with state variables"""

        # Create a template with state variables
        template = StringTemplate("Process $input with variables from state: ($var1, $var2)")

        class TemplateVars(BaseModel):
            input: str
            var1: str
            var2: str

        # Create caller with template
        caller = Caller(
            client=mock_client,
            model="provider:model",
            description="Test caller",
            instruction=system_instruction,
            input_template=template,
            input_params=TemplateVars,
        )

        # TODO: don't mock the render; just render and check for state variables
        # TODO: repeat for state substitution into system message

        # Mock the render method to check the rendered messages
        original_render = caller.render
        rendered_messages = []

        def mock_render(*args, **kwargs):
            result = original_render(*args, **kwargs)
            rendered_messages.append(result)
            return result

        caller.render = mock_render

        # Mock client to avoid actual API call
        mock_client.chat.completions.create.return_value = ChatCompletion(
            choices=[
                ChatCompletionChoice(
                    finish_reason="stop", message=ChatCompletionMessage(role="assistant", content="test response")
                )
            ]
        )

        # Call with state
        caller(input="test", state={"var1": "this is", "var2": "a test"})

        # Check that state was included in the rendered message
        assert any(
            "Process test with variables from state: (this is, a test)" in str(msg) for msg in rendered_messages
        )


# --- Integration Tests: Complete Flows ---
class TestCallerIntegration:
    """Full integration tests for different caller configurations"""

    def test_chat_flow(self, mock_client, simple_response, system_instruction):
        """Test basic chat flow with PassthroughValidator"""
        # Setup client to return a valid response
        mock_client.chat.completions.create.return_value = simple_response

        # Create chat caller
        caller = Caller(
            client=mock_client,
            model="provider:model",
            description="Test caller",
            instruction=system_instruction,
        )

        # Call the caller
        result = caller(input="test")

        # Verify result
        assert result == "test response"
        mock_client.chat.completions.create.assert_called_once()

    def test_structured_flow(self, mock_client, system_instruction):
        """Test structured response flow with PydanticValidator"""

        # Create a structured response model
        class TestResponse(BaseModel):
            message: str
            value: int

        # Setup client to return a valid JSON response
        mock_client.chat.completions.create.return_value = ChatCompletion(
            choices=[
                ChatCompletionChoice(
                    finish_reason="stop",
                    message=ChatCompletionMessage(
                        role="assistant", content='{"message": "test message", "value": 42}'
                    ),
                )
            ]
        )

        # Create structured caller
        caller = Caller(
            client=mock_client,
            model="provider:model",
            description="Test caller",
            instruction=system_instruction,
            output_validator=TestResponse,
        )

        # Call the caller
        result = caller(input="test")

        # Verify result
        assert isinstance(result, TestResponse)
        assert result.message == "test message"
        assert result.value == 42
        mock_client.chat.completions.create.assert_called_once()

    def test_structured_validation_failure(self, mock_client, system_instruction):
        """Test structured response with invalid JSON"""

        # Create a structured response model
        class TestResponse(BaseModel):
            message: str
            value: int

        # Setup client to return an invalid JSON response, then a valid one after repair
        mock_client.chat.completions.create.side_effect = [
            ChatCompletion(
                choices=[
                    ChatCompletionChoice(
                        finish_reason="stop",
                        message=ChatCompletionMessage(
                            role="assistant", content='{"message": "test", "value": "not_an_int"}'
                        ),
                    )
                ]
            ),
            ChatCompletion(
                choices=[
                    ChatCompletionChoice(
                        finish_reason="stop",
                        message=ChatCompletionMessage(role="assistant", content='{"message": "fixed", "value": 42}'),
                    )
                ]
            ),
        ]

        # Create structured caller
        caller = Caller(
            client=mock_client,
            model="provider:model",
            description="Test caller",
            instruction=system_instruction,
            output_validator=TestResponse,
        )

        # Call the caller
        result = caller(input="test")

        # Verify result
        assert isinstance(result, TestResponse)
        assert result.message == "fixed"
        assert result.value == 42
        assert mock_client.chat.completions.create.call_count == 2

    def test_tool_flow(self, mock_client, system_instruction):
        """Test tool calling flow"""

        @tool
        def test_tool(x: int) -> str:
            """Test tool that returns a string"""
            return f"Processed {x}"

        # Setup client to return a tool call
        mock_client.chat.completions.create.return_value = ChatCompletion(
            choices=[
                ChatCompletionChoice(
                    finish_reason="tool_calls",
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="call_123",
                                type="function",
                                function=ChatCompletionMessageToolCallFunction(
                                    name="test_tool",
                                    arguments=json.dumps({"x": 42}),
                                ),
                            )
                        ],
                    ),
                )
            ]
        )

        # Create tool caller
        caller = Caller(
            client=mock_client,
            model="provider:model",
            description="Test caller",
            instruction=system_instruction,
            tools=[test_tool],
            auto_invoke=True,
        )

        # Call the caller
        result = caller(input="test")

        # Verify result
        assert result == "Processed 42"
        mock_client.chat.completions.create.assert_called_once()

    def test_tool_validation_failure(self, mock_client, system_instruction):
        """Test tool call with invalid arguments"""

        @tool
        def test_tool(x: int, y: int) -> str:
            """Test tool that requires two arguments"""
            return f"Processed {x} and {y}"

        # Setup client to return an invalid tool call, then a valid one after repair
        mock_client.chat.completions.create.side_effect = [
            ChatCompletion(
                choices=[
                    ChatCompletionChoice(
                        finish_reason="tool_calls",
                        message=ChatCompletionMessage(
                            role="assistant",
                            content=None,
                            tool_calls=[
                                ChatCompletionMessageToolCall(
                                    id="call_123",
                                    type="function",
                                    function=ChatCompletionMessageToolCallFunction(
                                        name="test_tool",
                                        arguments=json.dumps({"x": 42}),  # Missing required argument y
                                    ),
                                )
                            ],
                        ),
                    )
                ]
            ),
            ChatCompletion(
                choices=[
                    ChatCompletionChoice(
                        finish_reason="tool_calls",
                        message=ChatCompletionMessage(
                            role="assistant",
                            content=None,
                            tool_calls=[
                                ChatCompletionMessageToolCall(
                                    id="call_456",
                                    type="function",
                                    function=ChatCompletionMessageToolCallFunction(
                                        name="test_tool",
                                        arguments=json.dumps({"x": 42, "y": 24}),
                                    ),
                                )
                            ],
                        ),
                    )
                ]
            ),
        ]

        # Create tool caller
        caller = Caller(
            client=mock_client,
            model="provider:model",
            description="Test caller",
            instruction=system_instruction,
            tools=[test_tool],
            auto_invoke=True,
        )

        # Call the caller
        result = caller(input="test")

        # Verify result
        assert result == "Processed 42 and 24"
        assert mock_client.chat.completions.create.call_count == 2

    def test_conversation_history_merging(self, mock_client, simple_response, system_instruction):
        """Test merging of conversation history"""
        # Setup client to return a valid response
        mock_client.chat.completions.create.return_value = simple_response

        # Create chat caller
        caller = Caller(
            client=mock_client,
            model="provider:model",
            description="Test caller",
            instruction=system_instruction,
        )

        # Create a conversation history
        history = Conversation(
            messages=[
                UserMessage(content="previous message"),
                AssistantMessage(content="previous response"),
            ]
        )

        # Call with history
        result = caller(input="test", conversation_history=history)

        # Verify result
        assert result == "test response"

        # Check that the API was called with the merged conversation
        call_args = mock_client.chat.completions.create.call_args[1]
        messages = call_args["messages"]

        # Should have system message + 2 history messages + 1 new user message
        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "previous message"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "previous response"
        assert messages[2]["role"] == "system"
        assert messages[2]["content"] == system_instruction
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "test"
