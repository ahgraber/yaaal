from __future__ import annotations

import json
from typing import Any
from unittest.mock import Mock, create_autospec

from pydantic import BaseModel, create_model
import pytest

from aisuite import Client

from yaaal.core.caller import Caller, create_chat_caller, create_structured_caller, create_tool_caller
from yaaal.core.exceptions import ValidationError
from yaaal.core.handler import ResponseHandler, ToolHandler
from yaaal.core.template import ConversationTemplate, StaticMessageTemplate, StringMessageTemplate
from yaaal.core.tool import tool
from yaaal.core.validator import PassthroughValidator, PydanticValidator, ToolValidator
from yaaal.types.core import AssistantMessage, Conversation, UserMessage
from yaaal.types.openai_compat import (
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallFunction,
)


# --- Test Fixtures ---
@pytest.fixture
def client():
    """Mock API client"""
    return create_autospec(Client)


@pytest.fixture
def conversation_template():
    """Basic conversation template for testing"""
    return ConversationTemplate(
        name="test_template",
        description="Test template",
        conversation_spec=[
            StaticMessageTemplate(role="system", template="You are a test assistant"),
            StringMessageTemplate(
                role="user", template="Process $input", validation_model=create_model("Vars", input=(str, ...))
            ),
        ],
    )


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
def mock_handler():
    """Mock handler with tracking capabilities"""
    handler = Mock()
    handler.process.return_value = "processed"
    handler.repair.return_value = Conversation(
        messages=[AssistantMessage(content="bad response"), UserMessage(content="please fix")]
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

    def test_model_validation(self, client, conversation_template, mock_handler):
        """Test model name validation"""
        # Valid model format
        caller = Caller(client, "provider:model", conversation_template, mock_handler)
        assert caller.model == "provider:model"

        # Invalid model format
        with pytest.raises(ValueError, match="must be in format"):
            Caller(client, "invalid_model", conversation_template, mock_handler)

    def test_request_params_validation(self, client, conversation_template, mock_handler):
        """Test request parameter validation"""
        # Valid params
        params = {"temperature": 0.7}
        caller = Caller(client, "provider:model", conversation_template, mock_handler, request_params=params)
        assert caller.request_params == params

        # Invalid params (model specified)
        with pytest.raises(ValueError, match="'model' should be set separately"):
            Caller(
                client, "provider:model", conversation_template, mock_handler, request_params={"model": "other:model"}
            )

    def test_max_repair_attempts(self, client, conversation_template, mock_handler, simple_response):
        """Test max repair attempts respected"""
        # Setup handler to always fail
        mock_handler.process.side_effect = ValueError("Always fail")

        # Setup client to always return a valid response
        client.chat.completions.create.return_value = simple_response

        caller = Caller(
            client=client,
            model="provider:model",
            conversation_template=conversation_template,
            handler=mock_handler,
            max_repair_attempts=2,
        )

        with pytest.raises(ValidationError, match="Max repair attempts reached"):
            caller(input="test")

        assert mock_handler.process.call_count == 3  # Initial + 2 repairs
        assert mock_handler.repair.call_count == 2  # Should attempt repair twice

    def test_provider_specific_params(self, client, conversation_template):
        """Test provider-specific parameter generation"""

        class TestResponse(BaseModel):
            value: str

        # Test OpenAI format
        caller = create_structured_caller(client, "openai:gpt-4", conversation_template, TestResponse)
        assert caller.request_params["response_format"] == {"type": "json_object"}

        # Test Anthropic format
        caller = create_structured_caller(client, "anthropic:claude", conversation_template, TestResponse)
        assert "tools" in caller.request_params
        assert caller.request_params["tool_choice"]["type"] == "tool"


# --- Unit Tests: Handler Integration ---
class TestCallerHandlerIntegration:
    """Tests for Caller interaction with handlers"""

    def test_handler_processing(self, client, conversation_template, mock_handler, simple_response):
        """Test basic handler processing flow"""
        caller = Caller(client, "provider:model", conversation_template, mock_handler)
        client.chat.completions.create.return_value = simple_response

        result = caller(input="test")
        assert result == "processed"
        mock_handler.process.assert_called_once()

    def test_repair_mechanism(self, client, conversation_template, mock_handler, simple_response):
        """Test repair flow when handler raises error"""
        mock_handler.process.side_effect = [ValueError("error"), "fixed"]
        caller = Caller(client, "provider:model", conversation_template, mock_handler)
        client.chat.completions.create.return_value = simple_response

        result = caller(input="test")
        assert result == "fixed"
        assert mock_handler.process.call_count == 2
        mock_handler.repair.assert_called_once()

    def test_empty_conversation_rejected(self, client, mock_handler):
        """Test empty conversation validation"""
        with pytest.raises(ValueError, match="Conversation list cannot be empty"):
            ConversationTemplate(
                name="empty",
                description="Empty template",
                conversation_spec=[],  # Empty spec
            )

    def test_conversation_validation(self, client, mock_handler):
        """Test conversation template validation"""
        # Missing system message
        with pytest.raises(ValueError, match="must contain at least one system message"):
            ConversationTemplate(
                name="bad",
                description="Bad template",
                conversation_spec=[
                    StringMessageTemplate(
                        role="user", template="$input", validation_model=create_model("Vars", input=(str, ...))
                    )
                ],
            )

    def test_invalid_response_repair(self, client, conversation_template, mock_handler, simple_response):
        """Test repair cycle with invalid responses"""
        # Setup handler to fail validation twice then succeed
        mock_handler.process.side_effect = [ValidationError("First error"), ValidationError("Second error"), "success"]

        caller = Caller(client, "provider:model", conversation_template, mock_handler)
        client.chat.completions.create.return_value = simple_response

        result = caller(input="test")
        assert result == "success"
        assert mock_handler.process.call_count == 3
        assert mock_handler.repair.call_count == 2


# --- Integration Tests: Complete Flows ---
class TestCallerIntegration:
    """Full integration tests for different caller configurations"""

    def test_chat_flow(self, client, conversation_template, simple_response):
        """Test basic chat flow with PassthroughValidator"""
        caller = create_chat_caller(client, "provider:model", conversation_template)
        client.chat.completions.create.return_value = simple_response

        result = caller(input="test")
        assert result == "test response"

    def test_structured_flow(self, client, conversation_template):
        """Test structured response flow with PydanticValidator"""

        class Response(BaseModel):
            message: str

        caller = create_structured_caller(client, "provider:model", conversation_template, Response)
        client.chat.completions.create.return_value = ChatCompletion(
            choices=[
                ChatCompletionChoice(
                    finish_reason="stop",
                    message=ChatCompletionMessage(role="assistant", content='{"message": "test"}'),
                )
            ]
        )

        result = caller(input="test")
        assert isinstance(result, Response)
        assert result.message == "test"

    def test_tool_flow(self, client, conversation_template):
        """Test tool calling flow"""

        @tool
        def test_tool(x: int) -> int:
            """Test tool"""
            return x + 1

        caller = create_tool_caller(client, "provider:model", conversation_template, [test_tool], auto_invoke=True)
        client.chat.completions.create.return_value = ChatCompletion(
            choices=[
                ChatCompletionChoice(
                    finish_reason="stop",
                    message=ChatCompletionMessage(
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="test1",
                                function=ChatCompletionMessageToolCallFunction(name="test_tool", arguments='{"x": 1}'),
                            )
                        ],
                    ),
                )
            ]
        )

        result = caller(input="test")
        assert result == "2"  # String because of JSON serialization

    def test_structured_validation_failure(self, client, conversation_template):
        """Test structured response with invalid JSON"""

        class Response(BaseModel):
            value: int

        caller = create_structured_caller(client, "provider:model", conversation_template, Response)
        client.chat.completions.create.return_value = ChatCompletion(
            choices=[
                ChatCompletionChoice(
                    finish_reason="stop",
                    message=ChatCompletionMessage(
                        role="assistant",
                        content='{"value": "not_an_int"}',  # Invalid type
                    ),
                )
            ]
        )

        with pytest.raises(ValidationError):
            caller(input="test")

    def test_tool_validation_failure(self, client, conversation_template):
        """Test tool call with invalid arguments"""

        @tool
        def test_tool(x: int, y: int) -> int:
            """Test tool requiring two arguments"""
            return x + y

        caller = create_tool_caller(client, "provider:model", conversation_template, [test_tool])
        client.chat.completions.create.return_value = ChatCompletion(
            choices=[
                ChatCompletionChoice(
                    finish_reason="stop",
                    message=ChatCompletionMessage(
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="test1",
                                function=ChatCompletionMessageToolCallFunction(
                                    name="test_tool",
                                    arguments='{"x": 1}',  # Missing required argument y
                                ),
                            )
                        ],
                    ),
                )
            ]
        )

        with pytest.raises(ValidationError):
            caller(input="test")

    def test_conversation_history_merging(self, client, conversation_template, simple_response):
        """Test merging of conversation history"""
        caller = create_chat_caller(client, "provider:model", conversation_template)
        client.chat.completions.create.return_value = simple_response

        # Create existing conversation
        history = Conversation(messages=[UserMessage(content="previous message")])

        caller(conversation_history=history, input="test")

        # Verify API call included both history and new messages
        call_args = client.chat.completions.create.call_args[1]
        messages = call_args["messages"]
        assert len(messages) > 2  # Should include history + new messages
        assert messages[0]["content"] == "previous message"
