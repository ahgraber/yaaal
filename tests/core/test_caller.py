import json
import re

from pydantic import BaseModel, Field, ValidationError
import pytest

from yaaal.core.caller import (
    PydanticResponseValidatorMixin,
    RegexResponseValidatorMixin,
)


class TestCaller:
    # There's not much in Caller to test at this point
    pass


class TestPydanticResponseValidatorMixin:
    @pytest.fixture
    def person(self):
        class Person(BaseModel):
            name: str
            age: int
            likes: list[str] = Field(min_length=2)

        return Person

    @pytest.fixture
    def valid_json(self):
        return {"name": "Bob", "age": 42, "likes": ["pizza", "ice cream"]}

    @pytest.fixture
    def mixin(self, person):
        class TestClass(PydanticResponseValidatorMixin):
            def __init__(self):
                self.response_validator = person

        return TestClass()

    def test_validate_with_valid_jsonstr(self, mixin, person, valid_json):
        testcase = json.dumps(valid_json)
        assert json.loads(testcase)  # ensure valid json

        result = mixin.validate(testcase)
        assert isinstance(result, person)

    def test_validate_with_invalid_jsonstr_quotes(self, mixin, person, valid_json):
        testcase = str(valid_json)
        # ensure testcase is invalid json
        with pytest.raises(json.JSONDecodeError):
            _ = json.loads(testcase)

        result = mixin.validate(testcase)
        assert isinstance(result, person)

    def test_validate_with_invalid(self, mixin, person):
        testcase = json.dumps({"name": "Bob", "age": 42, "likes": []})

        with pytest.raises(ValidationError):
            _ = mixin.validate(testcase)


class TestRegexResponseValidatorMixin:
    @pytest.fixture
    def response_type(self):
        return re.compile(r"^(\w+),\s*(\d+)$")  # matches "name, age" format

    @pytest.fixture
    def valid_str(self) -> str:
        return "Bob, 42"

    @pytest.fixture
    def mixin(self, response_type):
        class TestClass(RegexResponseValidatorMixin):
            def __init__(self):
                self.response_validator = response_type

        return TestClass()

    def test_validate_with_valid_string(self, mixin, valid_str):
        result = mixin.validate(valid_str)
        assert isinstance(result, re.Match)
        assert result.groups() == ("Bob", "42")

    def test_validate_with_invalid_string(self, mixin):
        invalid_str = "Invalid Format!"
        with pytest.raises(ValueError):
            mixin.validate(invalid_str)

    def test_match_groups_extraction(self, mixin, valid_str):
        result = mixin.validate(valid_str)
        name, age = result.groups()
        assert name == "Bob"
        assert age == "42"

    def test_validate_with_none(self, mixin):
        with pytest.raises(TypeError):
            mixin.validate(None)
