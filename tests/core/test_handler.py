import json

from pydantic import BaseModel
import pytest
from typing_extensions import override

from yaaal.core.base import BaseValidator, ResponseError, ValidationError
from yaaal.core.handler import CompositeHandler, ResponseHandler, ToolHandler
from yaaal.core.tools import tool
from yaaal.core.validator import PassthroughValidator, ToolValidator
from yaaal.types.base import JSON
from yaaal.types.core import (
    APIHandlerResult,
    AssistantMessage,
    Conversation,
    ToolResultMessage,
    UserMessage,
    ValidatorResult,
)
from yaaal.types.openai_compat import (
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallFunction,
    convert_response,
)


@pytest.fixture
def test_tool():
    @tool
    def test_tool(name: str, age: int) -> tuple:
        return (name, age)

    return test_tool


@pytest.fixture
def fail_content_validator():
    class FailValidator(BaseValidator):
        @override
        def validate(self, completion: str | ChatCompletionMessageToolCall) -> str:
            raise ValidationError

        @override
        def repair_instructions(self, failed_content: str, error: str) -> Conversation:
            return Conversation(
                messages=[
                    AssistantMessage(content=failed_content),
                    UserMessage(
                        content=f"""Validation failed: {error}""".strip(),
                    ),
                ]
            )

    return FailValidator()


@pytest.fixture
def fail_tool_validator(test_tool):
    fail_validator = ToolValidator(toolbox=[test_tool])

    def validate(completion: str | ChatCompletionMessageToolCall) -> BaseModel:
        raise ValidationError

    fail_validator.validate = validate

    return fail_validator


@pytest.fixture
def content() -> str:
    return "test content"


@pytest.fixture
def chat_completion_content(content) -> ChatCompletion:
    return ChatCompletion(
        choices=[
            ChatCompletionChoice(
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    content=content,
                ),
            )
        ]
    )


@pytest.fixture
def function() -> ChatCompletionMessageToolCallFunction:
    return ChatCompletionMessageToolCallFunction(
        name="test_tool",
        arguments=json.dumps({"name": "Bob", "age": 42}),
    )


@pytest.fixture
def tool_call(function) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id="testing123",
        function=function,
    )


@pytest.fixture
def chat_completion_tool(tool_call) -> ChatCompletion:
    return ChatCompletion(
        choices=[
            ChatCompletionChoice(
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    tool_calls=[tool_call],
                ),
            )
        ]
    )


class TestResponseHandler:
    @pytest.fixture
    def pass_handler(self):
        return ResponseHandler(validator=PassthroughValidator())

    @pytest.fixture
    def fail_handler(self, fail_content_validator):
        return ResponseHandler(validator=fail_content_validator)

    def test_init(self, pass_handler):
        assert isinstance(pass_handler.validator, PassthroughValidator)

    def test_content_passes(self, pass_handler, content, chat_completion_content):
        result = pass_handler(chat_completion_content)

        assert isinstance(result, AssistantMessage)
        assert result.content == content

    def test_validation_fails(self, fail_handler, chat_completion_content):
        with pytest.raises(ValidationError):
            fail_handler(chat_completion_content)

    def test_content_repair(self, fail_handler, chat_completion_content):
        repair_instructions = fail_handler.repair(chat_completion_content.choices[0].message, "Error message")

        assert isinstance(repair_instructions, Conversation)

    def test_toolcall_fails(self, pass_handler, chat_completion_tool):
        with pytest.raises(ValueError):
            pass_handler(chat_completion_tool)

    def test_toolcall_repair(self, fail_handler, chat_completion_tool):
        repair_instructions = fail_handler.repair(chat_completion_tool.choices[0].message, "Error message")

        assert repair_instructions is None


class TestToolHandler:
    @pytest.fixture
    def pass_handler(self, test_tool):
        return ToolHandler(validator=ToolValidator(toolbox=[test_tool]))

    @pytest.fixture
    def fail_handler(self, fail_tool_validator):
        return ToolHandler(validator=fail_tool_validator)

    def test_init(self, pass_handler):
        """Test that ToolHandler initializes correctly."""
        assert isinstance(pass_handler.validator, ToolValidator)
        assert pass_handler.auto_invoke is False

    def test_content_fails(self, pass_handler, chat_completion_content):
        with pytest.raises(ValueError):
            pass_handler(chat_completion_content)

    def test_toolcall_passes(self, pass_handler, chat_completion_tool):
        result = pass_handler(chat_completion_tool)

        assert isinstance(result, AssistantMessage)

        content = json.loads(result.content)
        assert content["name"] == "Bob"
        assert content["age"] == 42

    def test_toolcall_invoke(self, pass_handler, chat_completion_tool, tool_call):
        pass_handler.auto_invoke = True
        assert pass_handler.auto_invoke

        result = pass_handler(chat_completion_tool)
        assert isinstance(result, ToolResultMessage)
        assert result.tool_call_id == tool_call.id
        assert result.content == json.dumps(("Bob", 42))

    def test_validation_fails(self, fail_handler, chat_completion_tool):
        with pytest.raises(ValidationError):
            fail_handler(chat_completion_tool)

    def test_content_repair(self, fail_handler, chat_completion_content):
        repair_instructions = fail_handler.repair(chat_completion_content.choices[0].message, "Error message")

        assert repair_instructions is None

    def test_toolcall_repair(self, fail_handler, chat_completion_tool):
        repair_instructions = fail_handler.repair(chat_completion_tool.choices[0].message, "Error message")

        assert isinstance(repair_instructions, Conversation)


class TestCompositeHandler:
    @pytest.fixture
    def test_tool(self):
        @tool
        def test_tool(name: str, age: int) -> tuple:
            return (name, age)

        return test_tool

    @pytest.fixture
    def pass_handler(self, test_tool):
        return CompositeHandler(
            content_handler=ResponseHandler(validator=PassthroughValidator()),
            tool_handler=ToolHandler(validator=ToolValidator(toolbox=[test_tool])),
        )

    @pytest.fixture
    def fail_handler(self, fail_content_validator, fail_tool_validator):
        return CompositeHandler(
            content_handler=ResponseHandler(validator=fail_content_validator),
            tool_handler=ToolHandler(validator=fail_tool_validator),
        )

    def test_init(self, pass_handler):
        assert isinstance(pass_handler.content_handler, ResponseHandler)
        assert isinstance(pass_handler.tool_handler, ToolHandler)

    def test_content_passes(self, pass_handler, content, chat_completion_content):
        result = pass_handler(chat_completion_content)

        assert isinstance(result, AssistantMessage)
        assert result.content == content

    def test_content_validation_fails(self, fail_handler, chat_completion_content):
        with pytest.raises(ValidationError):
            fail_handler(chat_completion_content)

    def test_toolcall_passes(self, pass_handler, chat_completion_tool):
        result = pass_handler(chat_completion_tool)

        assert isinstance(result, AssistantMessage)

        content = json.loads(result.content)
        assert content["name"] == "Bob"
        assert content["age"] == 42

    def test_toolcall_invoke(self, pass_handler, chat_completion_tool, tool_call):
        pass_handler.tool_handler.auto_invoke = True
        assert pass_handler.tool_handler.auto_invoke

        result = pass_handler(chat_completion_tool)
        assert isinstance(result, ToolResultMessage)
        assert result.tool_call_id == tool_call.id
        assert result.content == json.dumps(("Bob", 42))

    def test_toolcallvalidation_fails(self, fail_handler, chat_completion_tool):
        with pytest.raises(ValidationError):
            fail_handler(chat_completion_tool)

    def test_refusal(self, pass_handler):
        with pytest.raises(ResponseError):
            pass_handler(
                ChatCompletion(
                    choices=[
                        ChatCompletionChoice(
                            finish_reason="content_filter",
                            message=ChatCompletionMessage(
                                role="assistant",
                                refusal="Error message",
                            ),
                        )
                    ]
                )
            )

    def test_message_error(self, pass_handler):
        with pytest.raises(ResponseError):
            pass_handler(
                ChatCompletion(
                    choices=[
                        ChatCompletionChoice(
                            finish_reason="content_filter",
                            message=ChatCompletionMessage(
                                role="assistant",
                                content=None,
                                tool_calls=None,
                                refusal=None,
                            ),
                        )
                    ]
                )
            )

    def test_content_repair(self, fail_handler, chat_completion_content):
        repair_instructions = fail_handler.repair(chat_completion_content.choices[0].message, "Error message")

        assert isinstance(repair_instructions, Conversation)

    def test_toolcall_repair(self, fail_handler, chat_completion_tool):
        repair_instructions = fail_handler.repair(chat_completion_tool.choices[0].message, "Error message")

        assert isinstance(repair_instructions, Conversation)
