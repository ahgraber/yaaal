import json
import re
from unittest.mock import Mock, create_autospec

from pydantic import BaseModel, Field, ValidationError
import pytest

from aisuite import Client

from yaaal.core._types import (
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionResponse,
    ChatCompletionToolCallFunction,
    Conversation,
    ToolMessage,
)
from yaaal.core.caller import (
    Caller,
    CallerValidationError,
    PydanticResponseValidatorMixin,
    RegexResponseValidatorMixin,
)
from yaaal.core.prompt import PassthroughMessageTemplate, Prompt, StaticMessageTemplate
from yaaal.core.tools import tool


class TestCaller:
    @pytest.fixture
    def mock_client(self):
        return create_autospec(Client)

    @pytest.fixture
    def simple_prompt(self):
        return Prompt(
            name="Prompt",
            description="Prompt for unit tests",
            system_template=StaticMessageTemplate(role="system", template="This is a test"),
            user_template=PassthroughMessageTemplate(),
        )

    @pytest.fixture
    def mock_response(self):
        return create_autospec(ChatCompletionResponse)

    @pytest.fixture
    def caller(self, mock_client, simple_prompt):
        class TestCaller(Caller):
            def __init__(self):
                self.client = mock_client
                self.model = "modelname"
                self.prompt = simple_prompt

        caller = TestCaller()
        return caller

    @pytest.fixture
    def add3(self):
        @tool
        def add3(a: int, b: int) -> int:
            return a + b + 3

        return add3

    def test_client_property(self, caller, mock_client):
        assert caller.client == mock_client

    def test_model_property(self, caller):
        assert caller.model == "modelname"

    def test_prompt_property(self, caller, simple_prompt):
        assert caller.prompt == simple_prompt

    def test_toolbox_property(self, caller):
        assert caller.toolbox == {}

    def test_auto_invoke_property(self, caller):
        assert caller.auto_invoke is False

    def test_request_params_property(self, caller):
        assert caller.request_params == {}

    def test_signature_method(self, caller, simple_prompt):
        assert caller.signature().model_json_schema() == simple_prompt.signature().model_json_schema()

    def test_call_method(self, caller, mock_response):
        pass

    def test_handle_response_with_content(self, caller, mock_response):
        conversation = caller.prompt.render(user_vars={"content": "testing 1, 2, 3"})

        content = "response message content"
        mock_response.choices = [Mock(message=Mock(content=content, tool_calls=None))]

        # given the response contains a tool call, we should just check that _handle_tool_call is called
        caller._handle_content = Mock(return_value="content handled")

        result = caller._handle_response(conversation=conversation, response=mock_response)
        assert result == "content handled"

    def test_handle_response_with_tool_call(self, caller, add3, mock_response):
        # add tool to toolbox
        caller.toolbox = [add3]

        conversation = caller.prompt.render(user_vars={"content": "testing 1, 2, 3"})

        kwargs = {"a": 1, "b": 2}
        tool_call = ChatCompletionMessageToolCall(
            id="tc_123",
            function=ChatCompletionToolCallFunction(
                name="add3",
                arguments=json.dumps(kwargs),
            ),
        )
        mock_response.choices = [Mock(message=Mock(content=None, tool_calls=[tool_call]))]

        # given the response contains a tool call, we should just check that _handle_tool_call is called
        caller._handle_tool_call = Mock(return_value="tool called")

        result = caller._handle_response(conversation=conversation, response=mock_response)
        assert result == "tool called"

    def test_handle_response_with_unexpected(self, caller, mock_response):
        conversation = caller.prompt.render(user_vars={"content": "testing 1, 2, 3"})
        mock_response.choices = [Mock(message=Mock(content=None, tool_calls=None))]
        with pytest.raises(ValueError):
            caller._handle_response(conversation=conversation, response=mock_response)

    def test_handle_content_with_valid_content(self, caller):
        conversation = caller.prompt.render(user_vars={"content": "testing 1, 2, 3"})

        caller._validate_content = Mock(return_value="validated_content")

        result = caller._handle_content(conversation=conversation, content="content")
        assert result == "validated_content"

    def test_handle_content_with_invalid_content(self, caller):
        conversation = caller.prompt.render(user_vars={"content": "testing 1, 2, 3"})

        caller._validate_content = Mock(side_effect=Exception("validation error"))
        caller._render_repair = Mock(return_value=None)

        with pytest.raises(CallerValidationError):
            caller._handle_content(conversation=conversation, content="content")

    def test_handle_tool_call_with_valid_tool(self, caller, add3):
        #        add tool to toolbox
        caller.toolbox = [add3]
        caller.auto_invoke = False

        conversation = caller.prompt.render(user_vars={"content": "testing 1, 2, 3"})

        kwargs = {"a": 1, "b": 2}
        tool_call = ChatCompletionMessageToolCall(
            id="tc_123",
            function=ChatCompletionToolCallFunction(
                name="add3",
                arguments=json.dumps(kwargs),
            ),
        )
        result = caller._handle_tool_call(conversation=conversation, tool_call=tool_call)

        # since auto_invoke is False, the result should be a BaseModel that defines the tool call
        assert isinstance(result, BaseModel)
        assert result.__class__.__name__ == "add3"
        assert result.a == 1
        assert result.b == 2

    def test_handle_tool_call_with_valid_tool_invoked(self, caller, add3):
        #        add tool to toolbox
        caller.toolbox = [add3]
        caller.auto_invoke = True

        conversation = caller.prompt.render(user_vars={"content": "testing 1, 2, 3"})

        kwargs = {"a": 1, "b": 2}
        tool_call = ChatCompletionMessageToolCall(
            id="tc_123",
            function=ChatCompletionToolCallFunction(
                name="add3",
                arguments=json.dumps(kwargs),
            ),
        )
        result = caller._handle_tool_call(conversation=conversation, tool_call=tool_call)

        # since auto_invoke is True, the result should be a ToolMessage containing the function result
        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == "tc_123"
        assert result.content == str(add3(**kwargs)) == str(6)

    def test_handle_tool_call_with_invalid_tool(self, caller, add3):
        #        add tool to toolbox
        caller.toolbox = [add3]
        caller.auto_invoke = False

        conversation = caller.prompt.render(user_vars={"content": "testing 1, 2, 3"})

        kwargs = {"a": 1, "b": 2}
        tool_call = ChatCompletionMessageToolCall(
            id="tc_123",
            function=ChatCompletionToolCallFunction(
                name="not_a_tool",
                arguments=json.dumps(kwargs),
            ),
        )
        with pytest.raises(KeyError):
            caller._handle_tool_call(conversation=conversation, tool_call=tool_call)


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

        result = mixin._validate_content(testcase)
        assert isinstance(result, person)

    def test_validate_with_invalid_jsonstr_quotes(self, mixin, person, valid_json):
        testcase = str(valid_json)
        # ensure testcase is invalid json
        with pytest.raises(json.JSONDecodeError):
            _ = json.loads(testcase)

        result = mixin._validate_content(testcase)
        assert isinstance(result, person)

    def test_validate_with_invalid(self, mixin, person):
        testcase = json.dumps({"name": "Bob", "age": 42, "likes": []})

        with pytest.raises(ValidationError):
            _ = mixin._validate_content(testcase)


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
        result = mixin._validate_content(valid_str)
        assert isinstance(result, re.Match)
        assert result.groups() == ("Bob", "42")

    def test_validate_with_invalid_string(self, mixin):
        invalid_str = "Invalid Format!"
        with pytest.raises(ValueError):
            mixin._validate_content(invalid_str)

    def test_match_groups_extraction(self, mixin, valid_str):
        result = mixin._validate_content(valid_str)
        name, age = result.groups()
        assert name == "Bob"
        assert age == "42"

    def test_validate_with_none(self, mixin):
        with pytest.raises(TypeError):
            mixin._validate_content(None)
