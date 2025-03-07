from __future__ import annotations

import json
import logging
from typing import Any, Callable

from pydantic import BaseModel, create_model
import pytest
from typing_extensions import override

from yaaal.core.base import Validator
from yaaal.core.exceptions import ResponseError, ValidationError
from yaaal.core.handler import ResponseHandler, ToolHandler
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
def text_response():
    """Basic text response fixture"""
    return ChatCompletion(
        choices=[
            ChatCompletionChoice(
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="test content"),
            )
        ]
    )


@pytest.fixture
def mock_callable():
    """Create a mock callable with signature"""

    @tool
    def mock_tool(x: int = 1) -> None:
        """Mock tool for testing"""
        pass

    return mock_tool


@pytest.fixture
def tool_response(mock_callable):
    """Basic tool call response fixture using mock_tool name"""
    return ChatCompletion(
        choices=[
            ChatCompletionChoice(
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="test123",
                            function=ChatCompletionMessageToolCallFunction(
                                name="mock_tool",
                                arguments=json.dumps({"x": 42}),
                            ),
                        )
                    ],
                ),
            )
        ]
    )


@pytest.fixture
def mock_validator():
    """Mock validator for testing handler behavior"""

    class MockValidator(Validator):
        def __init__(self):
            self.validate_called = False
            self.repair_called = False

        @override
        def validate(self, completion: Any) -> str:
            self.validate_called = True
            self.last_completion = completion
            return "validated"

        @override
        def repair_instructions(self, failed_content: Any, error: str) -> Conversation:
            self.repair_called = True
            self.last_error = error
            return Conversation(messages=[AssistantMessage(content="mock repair")])

    return MockValidator()


@pytest.fixture
def calc_tool():
    """Sample calculation tool"""

    @tool
    def add3(x: int, y: int) -> int:
        return x + y + 3

    return add3


# --- Unit Tests: ResponseHandler ---
class TestResponseHandlerUnit:
    """Unit tests for ResponseHandler focusing on handler logic"""

    def test_handler_calls_validator(self, text_response, mock_validator):
        """Verify handler properly delegates to validator"""
        handler = ResponseHandler(validator=mock_validator)
        result = handler.process(text_response)

        assert mock_validator.validate_called
        assert mock_validator.last_completion == text_response.choices[0].message.content
        assert result == "validated"

    def test_handler_propagates_validation_error(self, text_response, mock_validator):
        """Verify validation errors are properly propagated"""

        def raise_error(_):
            raise ValidationError("test error")

        mock_validator.validate = raise_error
        handler = ResponseHandler(validator=mock_validator)

        with pytest.raises(ValidationError):
            handler.process(text_response)

    def test_repair_delegates_to_validator(self, mock_validator):
        """Verify repair properly delegates to validator"""
        handler = ResponseHandler(validator=mock_validator)
        msg = ChatCompletionMessage(role="assistant", content="test")
        handler.repair(msg, "test error")

        assert mock_validator.repair_called
        assert mock_validator.last_error == "test error"

    def test_content_filter_handling(self, text_response):
        """Test handling of filtered/refused content"""
        handler = ResponseHandler(validator=PassthroughValidator())

        filtered_response = ChatCompletion(
            choices=[
                ChatCompletionChoice(
                    finish_reason="content_filter",
                    message=ChatCompletionMessage(
                        role="assistant", content=None, tool_calls=None, refusal="Content was filtered"
                    ),
                )
            ]
        )

        with pytest.raises(ValueError, match="Expected content response but received none"):
            handler.process(filtered_response)


# --- Unit Tests: ToolHandler ---
class TestToolHandlerUnit:
    """Unit tests for ToolHandler focusing on handler logic"""

    @pytest.fixture
    def mock_tool_validator(self, mock_callable):
        """Mock tool validator that tracks calls"""

        class MockToolValidator(ToolValidator):
            def __init__(self):
                self.validate_called = False
                super().__init__([mock_callable])  # Initialize with valid tool

            @override
            def validate(self, completion: Any) -> BaseModel:
                self.validate_called = True
                self.last_completion = completion
                return create_model("MockArgs", x=(int, 1))()

        return MockToolValidator()

    def test_handler_calls_validator(self, tool_response, mock_tool_validator):
        """Verify handler properly delegates to tool validator"""
        handler = ToolHandler(validator=mock_tool_validator)
        handler.process(tool_response)

        assert mock_tool_validator.validate_called
        assert mock_tool_validator.last_completion == tool_response.choices[0].message.tool_calls[0]

    def test_auto_invoke_respects_flag(self, tool_response, mock_tool_validator):
        """Verify auto_invoke behavior"""
        handler = ToolHandler(validator=mock_tool_validator, auto_invoke=False)
        result = handler.process(tool_response)
        assert isinstance(result, BaseModel)

        handler.auto_invoke = True
        result = handler.process(tool_response)
        assert not isinstance(result, BaseModel)


# --- Integration Tests ---
class TestHandlerValidatorIntegration:
    """Integration tests for handlers with real validators"""

    def test_response_handler_with_pydantic(self):
        """Test ResponseHandler with PydanticValidator"""

        class Response(BaseModel):
            message: str
            count: int

        handler = ResponseHandler(validator=PydanticValidator(Response))
        response = ChatCompletion(
            choices=[
                ChatCompletionChoice(
                    finish_reason="stop",
                    message=ChatCompletionMessage(role="assistant", content='{"message": "test", "count": 42}'),
                )
            ]
        )

        result = handler.process(response)
        assert isinstance(result, Response)
        assert result.message == "test"
        assert result.count == 42

    def test_tool_handler_with_real_tool(self):
        """Test ToolHandler with actual tool implementation"""

        @tool
        def add(x: int, y: int) -> dict[str, int]:
            """Add two numbers"""
            return {"sum": x + y}

        handler = ToolHandler(validator=ToolValidator(toolbox=[add]), auto_invoke=True)
        response = ChatCompletion(
            choices=[
                ChatCompletionChoice(
                    finish_reason="stop",
                    message=ChatCompletionMessage(
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="test1",
                                function=ChatCompletionMessageToolCallFunction(
                                    name="add", arguments='{"x": 2, "y": 3}'
                                ),
                            )
                        ],
                    ),
                )
            ]
        )

        result = handler.process(response)
        assert result == '{"sum": 5}'

    def test_error_repair_cycle(self):
        """Test full validation error and repair cycle"""

        class StrictValidator(PydanticValidator):
            def validate(self, completion: str) -> BaseModel:
                if "error" in completion.lower():
                    raise ValidationError("Found error in response")
                return super().validate(completion)

        class Response(BaseModel):
            status: str

        handler = ResponseHandler(validator=StrictValidator(Response))
        error_response = ChatCompletion(
            choices=[
                ChatCompletionChoice(
                    finish_reason="stop",
                    message=ChatCompletionMessage(role="assistant", content='{"status": "error occurred"}'),
                )
            ]
        )

        # Verify error raised and repair instructions generated
        with pytest.raises(ValidationError):
            handler.process(error_response)

        repair = handler.repair(error_response.choices[0].message, "Found error in response")
        assert isinstance(repair, Conversation)
        assert len(repair.messages) == 2
        assert "schema" in repair.messages[1].content
