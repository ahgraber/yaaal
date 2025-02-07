import json
import re
from typing import Any, Optional

from pydantic import BaseModel, ValidationError as PydanticValidationError
import pytest

from yaaal.core.base import BaseCaller, ValidationError
from yaaal.core.tools import tool
from yaaal.core.validator import (
    PassthroughValidator,
    PydanticValidator,
    RegexValidator,
    ToolValidator,
)
from yaaal.types.core import Conversation, Message
from yaaal.types.openai_compat import ChatCompletionMessageToolCall, ChatCompletionMessageToolCallFunction


class TestPassthroughValidator:
    def test_pass(self):
        validator = PassthroughValidator()
        completion = "test content"
        assert validator.validate(completion) == completion

    def test_repair(self):
        validator = PassthroughValidator()
        assert validator.repair_instructions("failed", "error") is None


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

    def test_missing_description_warns(self, caplog):
        import logging

        @tool
        def undocumented_tool(n: int):
            return f"This is test {n}"

        caplog.set_level(logging.INFO, logger="yaaal.core.validator")

        _ = ToolValidator(toolbox=[undocumented_tool])
        logs = caplog.record_tuples

        # there should be only one log and it should warn about missing docstrings
        assert logs == [
            (
                "yaaal.core.validator",
                logging.WARNING,
                f"Did not find a 'description' for {undocumented_tool.signature().__name__}.  Does the function need docstrings?",
            )
        ]

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
        assert json.dumps(sample_tool.signature().model_json_schema()) in repair.messages[1].content
