import json
import re
from typing import Any, Optional

from pydantic import BaseModel, ValidationError as PydanticValidationError
import pytest

from yaaal.core.base import CallableWithSchema
from yaaal.core.exceptions import ValidationError
from yaaal.core.tool import tool
from yaaal.core.validator import (
    PassthroughValidator,
    PydanticValidator,
    RegexValidator,
    ToolValidator,
)
from yaaal.types_.core import Conversation, Message
from yaaal.types_.openai_compat import ChatCompletionMessageToolCall, ChatCompletionMessageToolCallFunction


class TestPassthroughValidator:
    def test_pass(self):
        validator = PassthroughValidator()
        completion = "test content"
        assert validator.validate(completion) == completion

    def test_repair(self):
        validator = PassthroughValidator()
        assert validator.repair_instructions("failed", "error") is None

    def test_validation_error_non_string(self):
        validator = PassthroughValidator()
        with pytest.raises(TypeError):
            validator.validate(123)

    def test_validation_none(self):
        validator = PassthroughValidator()
        with pytest.raises(TypeError):
            validator.validate(None)


class TestPydanticValidator:
    @pytest.fixture
    def sample_model(self):
        class SampleModel(BaseModel):
            name: str
            age: int

        return SampleModel

    @pytest.fixture
    def validator(self, sample_model):
        return PydanticValidator(sample_model)

    def test_pass(self, sample_model, validator):
        valid_completion = json.dumps({"name": "Bob", "age": 42})

        result = validator.validate(valid_completion)
        assert isinstance(result, sample_model)
        assert result.name == "Bob"
        assert result.age == 42

    def test_fail(self, validator):
        invalid_completion = '{"name": "Bob"}'  # missing required field
        with pytest.raises(PydanticValidationError):
            validator.validate(invalid_completion)

    def test_repair(self, sample_model, validator):
        repair = validator.repair_instructions('{"invalid": true}', "validation error")
        assert isinstance(repair, Conversation)
        assert len(repair.messages) == 2
        assert json.dumps(sample_model.model_json_schema()) in repair.messages[1].content

    def test_validate_valid_json(self, sample_model):
        validator = PydanticValidator(sample_model)
        result = validator.validate('{"name": "Bob", "age": 42}')
        assert isinstance(result, sample_model)
        assert result.name == "Bob"
        assert result.age == 42

    def test_validate_invalid_json(self, sample_model):
        validator = PydanticValidator(sample_model)
        with pytest.raises(PydanticValidationError):
            validator.validate('{"name": "Bob"}')  # missing age

    def test_validate_malformed_json(self, sample_model):
        validator = PydanticValidator(sample_model)

        # malformed JSON should be handled with json_repair
        result = validator.validate('{"name": "Bob", age: 42}')
        assert isinstance(result, sample_model)
        assert result.name == "Bob"
        assert result.age == 42

    def test_repair_instructions(self, sample_model):
        validator = PydanticValidator(sample_model)
        conversation = validator.repair_instructions('{"name": "Bob"}', "Field 'age' is required")
        assert isinstance(conversation, Conversation)
        assert len(conversation.messages) == 2
        assert "Validation failed" in conversation.messages[1].content
        assert "schema" in conversation.messages[1].content


class TestRegexValidator:
    @pytest.fixture
    def regex(self):
        return r"test\d+"

    @pytest.fixture
    def pattern(self, regex):
        return re.compile(regex)

    @pytest.fixture
    def validator(self, pattern):
        return RegexValidator(pattern)

    def test_pass(self, validator):
        result = validator.validate("test123")
        assert result == "test123"

    def test_fail(self, validator):
        with pytest.raises(ValidationError):
            validator.validate("invalid")

    def test_repair(self, regex, validator):
        repair = validator.repair_instructions("invalid", "no match")
        assert isinstance(repair, Conversation)
        assert len(repair.messages) == 2
        assert regex in repair.messages[1].content

    @pytest.fixture
    def email_pattern(self):
        return re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")

    def test_validate_match(self, email_pattern):
        validator = RegexValidator(email_pattern)
        result = validator.validate("Contact me at user@example.com for details")
        assert result == "user@example.com"

    def test_validate_no_match(self, email_pattern):
        validator = RegexValidator(email_pattern)
        with pytest.raises(ValidationError):
            validator.validate("No email address here")

    def test_repair_instructions(self, email_pattern):
        validator = RegexValidator(email_pattern)
        conversation = validator.repair_instructions("Invalid format", "No match found")
        assert isinstance(conversation, Conversation)
        assert len(conversation.messages) == 2
        assert "regex pattern" in conversation.messages[1].content


class TestToolValidator:
    @pytest.fixture
    def sample_tool(self):
        @tool
        def sample_tool(arg1: str, arg2: int) -> str:
            """The concatenator."""
            return arg1 + str(arg2)

        return sample_tool

    @pytest.fixture
    def validator(self, sample_tool):
        return ToolValidator(toolbox=[sample_tool])

    def test_pass(self, sample_tool, validator):
        tool_call = ChatCompletionMessageToolCall(
            id="test123",
            function=ChatCompletionMessageToolCallFunction(
                name="sample_tool",
                arguments='{"arg1": "test", "arg2": 3}',
            ),
        )
        result = validator.validate(tool_call)
        assert isinstance(result, BaseModel)
        assert result.arg1 == "test"

    def test_missing_tool(self, validator):
        with pytest.raises(ValidationError):
            invalid_function = ChatCompletionMessageToolCall(
                id="test123",
                function=ChatCompletionMessageToolCallFunction(
                    name="nonexistent_tool",
                    arguments="{}",
                ),
            )
            validator.validate(invalid_function)

    def test_validation_fail(self, validator):
        with pytest.raises(PydanticValidationError):
            invalid_function = ChatCompletionMessageToolCall(
                id="test123",
                function=ChatCompletionMessageToolCallFunction(
                    name="sample_tool",
                    arguments="{}",
                ),
            )
            validator.validate(invalid_function)

    def test_repair(self, sample_tool, validator):
        tool_call = ChatCompletionMessageToolCall(
            id="testcall",
            function=ChatCompletionMessageToolCallFunction(
                name="sample_tool",
                arguments='{"invalid": true}',
            ),
        )
        repair = validator.repair_instructions(tool_call, "validation error")
        assert isinstance(repair, Conversation)
        assert len(repair.messages) == 2
        assert json.dumps(sample_tool.function_schema.model_json_schema()) in repair.messages[1].content

    def test_validate_valid_call(self, sample_tool):
        validator = ToolValidator(toolbox=[sample_tool])
        print(validator.toolbox)
        result = validator.validate(
            ChatCompletionMessageToolCall(
                id="call_123",
                function=ChatCompletionMessageToolCallFunction(
                    name="sample_tool", arguments='{"arg1": "test", "arg2": "2"}'
                ),
                type="function",
            )
        )
        assert isinstance(result, sample_tool.function_schema.pydantic_model)

    def test_validate_invalid_tool(self, sample_tool):
        validator = ToolValidator([sample_tool])
        with pytest.raises(ValidationError):
            validator.validate(
                ChatCompletionMessageToolCall(
                    id="call_123",
                    function=ChatCompletionMessageToolCallFunction(name="unknown_tool", arguments='{"a": 1, "b": 2}'),
                    type="function",
                )
            )

    def test_validate_invalid_args(self, sample_tool):
        validator = ToolValidator([sample_tool])
        with pytest.raises(ValidationError):
            validator.validate(
                ChatCompletionMessageToolCall(
                    id="call_123",
                    function=ChatCompletionMessageToolCallFunction(
                        name="add", arguments='{"a": "not_a_number", "b": 2}'
                    ),
                    type="function",
                )
            )

    def test_invalid_tool_registration(self):
        with pytest.raises(TypeError):
            ToolValidator([lambda x: x])  # Plain callable without signature
