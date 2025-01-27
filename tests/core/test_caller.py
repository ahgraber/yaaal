import json
import re
from typing import Pattern, cast
from unittest.mock import Mock, create_autospec

from pydantic import BaseModel, Field, ValidationError
import pytest

from aisuite import Client

from yaaal.core.caller import (
    BaseCaller,
    CallerValidationError,
    ChatCaller,
    RegexCaller,
    StructuredCaller,
    ToolCaller,
)
from yaaal.core.prompt import PassthroughMessageTemplate, Prompt, StaticMessageTemplate
from yaaal.core.tools import tool
from yaaal.types.core import (
    Conversation,
    Message,
    ToolMessage,
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
def response_with_content():
    response = ChatCompletion(
        choices=[
            ChatCompletionChoice(
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    content="Hello world!",
                    tool_calls=None,
                    refusal=None,
                ),
            )
        ]
    )
    return response


@pytest.fixture
def add3():
    def add3(x: int, y: int) -> int:
        "A fancy way to add."
        return x + y + 3

    return tool(add3)


@pytest.fixture
def response_with_tool_call():
    response = ChatCompletion(
        choices=[
            ChatCompletionChoice(
                finish_reason="tool_calls",
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="tool42",
                            function=ChatCompletionMessageToolCallFunction(
                                name="add3",
                                arguments=json.dumps({"x": 1, "y": 2}),
                            ),
                        )
                    ],
                    refusal=None,
                ),
            )
        ]
    )
    return response


class TestBaseCaller:
    @pytest.fixture
    def response_with_content(self):
        response = ChatCompletion(
            choices=[
                ChatCompletionChoice(
                    finish_reason="stop",
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="Hello world!",
                        tool_calls=None,
                        refusal=None,
                    ),
                )
            ]
        )
        return response

    @pytest.fixture
    def response_with_tool_call(self):
        response = ChatCompletion(
            choices=[
                ChatCompletionChoice(
                    finish_reason="tool_calls",
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="tool42",
                                function=ChatCompletionMessageToolCallFunction(
                                    name="testname",
                                    arguments="testargs",
                                ),
                            )
                        ],
                        refusal=None,
                    ),
                )
            ]
        )
        return response

    def test_client_property(self):
        caller = BaseCaller()

        client = create_autospec(Client)
        caller.client = client
        assert caller.client == client

    def test_model_property(self):
        caller = BaseCaller()

        model = "provider:model"
        caller.model = model
        assert caller.model == model

    def test_prompt_property(self):
        caller = BaseCaller()

        prompt = create_autospec(Prompt)
        caller.prompt = prompt
        assert caller.prompt == prompt

    def test_request_params_property(self):
        caller = BaseCaller()

        params = {"param1": "value1"}
        caller.request_params = params
        assert caller.request_params == params

    def test_max_repair_attempts_property(self):
        caller = BaseCaller()

        attempts = 3
        caller.max_repair_attempts = attempts
        assert caller.max_repair_attempts == attempts

    def test_signature(self):
        caller = BaseCaller()

        prompt = create_autospec(Prompt)
        caller.prompt = prompt
        caller.prompt.signature.return_value = BaseModel
        assert caller.signature() == BaseModel

    def test_call(self):
        caller = BaseCaller()

        caller.client = create_autospec(Client)
        caller.model = "provider:model"
        caller.prompt = create_autospec(Prompt)
        caller.request_params = {"param1": "value1"}

        caller._chat_completions_create = Mock(return_value=create_autospec(ChatCompletion))
        caller._handle_response = Mock(return_value="response")

        result = caller(system_vars={}, user_vars={})
        assert result == "response"

    def test_chat_completions_create(self):
        from aisuite.framework import ChatCompletionResponse as AISuiteChatCompletionResponse

        caller = BaseCaller()
        caller.client = create_autospec(Client)
        caller.model = "provider:model"
        caller.prompt = create_autospec(Prompt)
        caller.request_params = {"param1": "value1"}

        conversation = create_autospec(Conversation)
        conversation.model_dump.return_value = {"messages": []}
        caller.client.chat.completions.create.return_value = AISuiteChatCompletionResponse()

        result = caller._chat_completions_create(conversation)
        assert isinstance(result, ChatCompletion)

    def test_handle_response_with_content(self, response_with_content):
        caller = BaseCaller()

        caller._handle_content = Mock(return_value="content")

        result = caller._handle_response(
            conversation=create_autospec(Conversation),
            response=response_with_content,
        )
        assert result == "content"

    def test_handle_response_with_tool_call(self, response_with_tool_call):
        caller = BaseCaller()

        caller._handle_tool_call = Mock(return_value="tool_call")

        result = caller._handle_response(
            conversation=create_autospec(Conversation),
            response=response_with_tool_call,
        )
        assert result == "tool_call"

    def test_handle_response_with_unexpected_response(self, response_with_content):
        caller = BaseCaller()

        response_with_content.choices[0].message.content = None
        response_with_content.choices[0].message.tool_calls = None

        with pytest.raises(ValueError):
            caller._handle_response(
                conversation=create_autospec(Conversation),
                response=response_with_content,
            )

    def test_handle_content(self):
        caller = BaseCaller()

        caller._validate_content = Mock(return_value="validated_content")
        # caller._repair_response = Mock(return_value=None)

        result = caller._handle_content(
            conversation=create_autospec(Conversation),
            content="content",
        )
        assert result == "validated_content"

    def test_handle_content_with_repair(self, response_with_content):
        caller = BaseCaller()
        caller.max_repair_attempts = 3

        # induce error in _validate_content
        caller._validate_content = Mock(side_effect=Exception("error"))
        # triggering _repair_response
        caller._repair_response = Mock(
            return_value=Conversation(
                messages=[
                    Message(role="assistant", content="invalid"),
                    Message(role="system", content="fix this"),
                ]
            )
        )
        # the repair call to _chat_completions_create should return a valid response object
        caller._chat_completions_create = Mock(return_value=response_with_content)
        caller._handle_response = Mock(return_value="repaired_content")

        result = caller._handle_content(
            conversation=Conversation(
                messages=[
                    Message(role="system", content="original instructions"),
                    Message(role="assistant", content="invalid"),
                    Message(role="system", content="fix this"),
                ]
            ),
            content="bad_content",
        )
        assert result == "repaired_content"

    def test_handle_content_with_max_repair_attempts(self, response_with_content):
        caller = BaseCaller()
        caller.max_repair_attempts = 1

        caller._validate_content = Mock(side_effect=Exception("error"))
        caller._repair_response = Mock(
            return_value=Conversation(
                messages=[
                    Message(role="assistant", content="invalid"),
                    Message(role="system", content="fix this"),
                ]
            )
        )

        caller._chat_completions_create = Mock(return_value=response_with_content)

        with pytest.raises(CallerValidationError):
            caller._handle_content(
                conversation=Conversation(
                    messages=[
                        Message(role="system", content="original instructions"),
                        Message(role="assistant", content="invalid"),
                        Message(role="system", content="fix this"),
                    ]
                ),
                content="content",
            )

    def test_handle_tool_call(self):
        caller = BaseCaller()
        with pytest.raises(NotImplementedError):
            caller._handle_tool_call(conversation=create_autospec(Conversation), tool_call=Mock())

    def test_validate_tool(self):
        caller = BaseCaller()
        with pytest.raises(NotImplementedError):
            caller._validate_tool(name="name", arguments="arguments")

    def test_repair_tool(self):
        caller = BaseCaller()
        with pytest.raises(NotImplementedError):
            caller._repair_tool(tool_call=Mock(), exception="exception")


class TestChatCaller:
    # This just adds a nice init to BaseCaller
    pass


class TestRegexCaller:
    @pytest.fixture
    def regex_caller(self):
        client = create_autospec(Client)
        model = "provider:model"
        prompt = create_autospec(Prompt)
        response_validator = re.compile(r"Hello\s\w+!")
        return RegexCaller(
            client=client,
            model=model,
            prompt=prompt,
            response_validator=response_validator,
        )

    def test_response_validator_property(self, regex_caller):
        pattern = re.compile(r"New\sPattern")
        regex_caller.response_validator = pattern

        assert isinstance(regex_caller.response_validator, Pattern)
        assert regex_caller.response_validator == pattern

    def test_validate_content(self, regex_caller):
        response = "Hello world!"
        result = regex_caller._validate_content(response)

        assert isinstance(result, str)
        assert result == "Hello world!"

    def test_validate_content_no_match(self, regex_caller):
        response = "Goodbye world!"
        with pytest.raises(ValueError, match="Response did not match expected pattern"):
            regex_caller._validate_content(response)

    def test_repair_response(self, regex_caller):
        response_content = "Invalid response"
        exception = "Response did not match expected pattern"

        conversation = regex_caller._repair_response(response_content, exception)

        assert isinstance(conversation, Conversation)
        assert len(conversation.messages) == 2
        assert conversation.messages[0].role == "assistant"
        assert conversation.messages[0].content == response_content
        assert conversation.messages[1].role == "user"
        assert "Response must match the following regex pattern" in conversation.messages[1].content
        assert regex_caller.response_validator.pattern in conversation.messages[1].content


class TestStructuredCaller:
    @pytest.fixture
    def structured_caller(self, add3):
        client = create_autospec(Client)
        model = "provider:model"
        prompt = create_autospec(Prompt)
        response_validator = add3.signature()

        return StructuredCaller(
            client=client,
            model=model,
            prompt=prompt,
            response_validator=response_validator,
        )

    def test_make_request_params(self, structured_caller):
        request_params = {"param1": "value1"}
        result = structured_caller._make_request_params(request_params)
        assert request_params["param1"] == "value1"
        assert isinstance(request_params, dict)
        assert "tools" in result
        assert "tool_choice" in result

        request_params = {"model": "this should fail"}
        with pytest.raises(ValueError):
            structured_caller._make_request_params(request_params)

    def test_response_validator_property(self, structured_caller):
        new_validator = create_autospec(BaseModel)
        structured_caller.response_validator = new_validator

        assert isinstance(structured_caller.response_validator, BaseModel)
        assert structured_caller.response_validator == new_validator

    def test_validate_content(self, structured_caller):
        response = json.dumps({"x": 1, "y": 2})

        result = structured_caller._validate_content(response)

        assert isinstance(result, BaseModel)
        assert result.x == 1
        assert result.y == 2

    def test_validate_content_with_exception(self, structured_caller):
        response = '{"key": "value"}'
        structured_caller.response_validator.model_validate = Mock(
            side_effect=ValidationError("mock validation error", [])
        )
        with pytest.raises(ValidationError):
            structured_caller._validate_content(response)

    def test_repair_response(self, structured_caller):
        response_content = '{"key": "value"}'
        exception = "Validation error"

        conversation = structured_caller._repair_response(response_content, exception)

        assert isinstance(conversation, Conversation)
        assert len(conversation.messages) == 2
        assert conversation.messages[0].role == "assistant"
        assert conversation.messages[0].content == response_content
        assert conversation.messages[1].role == "user"
        assert exception in conversation.messages[1].content

    def test_handle_tool_call(self, structured_caller, add3, response_with_tool_call):
        arguments = {"x": 1, "y": 2}
        tool_call = response_with_tool_call.choices[0].message.tool_calls[0]
        tool_call.function.arguments = json.dumps(arguments)

        result = structured_caller._handle_tool_call(conversation=create_autospec(Conversation), tool_call=tool_call)

        assert isinstance(result, BaseModel)
        assert result.x == arguments["x"]
        assert result.y == arguments["y"]
        assert result.model_dump() == add3.signature().model_validate(arguments).model_dump()

    def test_handle_tool_call_with_validation_error(self, structured_caller, add3, response_with_tool_call):
        arguments = {"x": 1, "z": None}
        tool_call = response_with_tool_call.choices[0].message.tool_calls[0]
        tool_call.function.arguments = json.dumps(arguments)

        structured_caller._chat_completions_create = Mock(return_value=create_autospec(ChatCompletion))
        structured_caller._handle_response = Mock(return_value="repaired_tool")

        result = structured_caller._handle_tool_call(
            conversation=Conversation(
                messages=[
                    Message(role="system", content="original instructions"),
                    Message(role="assistant", content=json.dumps({"name": "add3", "arguments": arguments})),
                    Message(role="system", content="fix this"),
                ]
            ),
            tool_call=tool_call,
        )
        assert result == "repaired_tool"

    def test_handle_tool_call_with_max_repair_attempts(self, structured_caller, add3, response_with_tool_call):
        arguments = {"x": 1, "z": None}
        tool_call = response_with_tool_call.choices[0].message.tool_calls[0]
        tool_call.function.arguments = json.dumps(arguments)

        structured_caller.max_repair_attempts = 1

        structured_caller._repair_tool = Mock(
            return_value=Conversation(
                messages=[
                    Message(role="assistant", content=json.dumps({"name": "add3", "arguments": arguments})),
                    Message(role="system", content="fix this"),
                ]
            )
        )
        structured_caller._chat_completions_create = Mock(return_value=response_with_tool_call)

        with pytest.raises(CallerValidationError):
            structured_caller._handle_tool_call(
                conversation=Conversation(
                    messages=[
                        Message(role="system", content="original instructions"),
                        Message(role="assistant", content=json.dumps({"name": "add3", "arguments": arguments})),
                        Message(role="system", content="fix this"),
                    ]
                ),
                tool_call=tool_call,
            )

    def test_validate_tool(self, structured_caller, add3, response_with_tool_call):
        arguments = {"x": 1, "y": 2}
        tool_call = response_with_tool_call.choices[0].message.tool_calls[0]
        tool_call.function.arguments = json.dumps(arguments)

        result = structured_caller._validate_tool(name="add3", arguments=json.dumps(arguments))

        assert isinstance(result, BaseModel)
        assert result.model_dump() == add3.signature().model_validate(arguments).model_dump()

    def test_repair_tool(self, structured_caller, response_with_tool_call):
        arguments = {"x": 1, "z": None}
        tool_call = response_with_tool_call.choices[0].message.tool_calls[0]
        tool_call.function.arguments = json.dumps(arguments)

        exception = "Validation error"
        conversation = structured_caller._repair_tool(tool_call, exception)

        assert isinstance(conversation, Conversation)
        assert len(conversation.messages) == 2
        assert conversation.messages[0].role == "assistant"
        assert conversation.messages[0].content == tool_call.function.model_dump_json()
        assert conversation.messages[1].role == "user"
        assert exception in conversation.messages[1].content


class TestToolCaller:
    @pytest.fixture
    def caller(self):
        client = create_autospec(Client)
        model = "provider:model"
        prompt = Prompt(
            name="caller-as-a-tool",
            description="A caller to test tool-calling functionality",
            system_template=StaticMessageTemplate(role="system", template="The system is down. The system is down."),
            user_template=PassthroughMessageTemplate(),
        )

        return ChatCaller(
            client=client,
            model=model,
            prompt=prompt,
        )

    @pytest.fixture
    def tool_caller(self, add3, caller):
        client = create_autospec(Client)
        model = "provider:model"
        prompt = create_autospec(Prompt)
        return ToolCaller(
            client=client,
            model=model,
            prompt=prompt,
            toolbox=[add3, caller],
            auto_invoke=False,
        )

    def test_toolbox_property(self, tool_caller, add3, caller):
        assert {"add3", "caller_as_a_tool"} == set(tool_caller.toolbox)
        assert tool_caller.toolbox["add3"] == add3
        assert tool_caller.toolbox["caller_as_a_tool"] == caller

        with pytest.raises(TypeError):
            tool_caller.toolbox = ["failme"]

    def test_auto_invoke_property(self, tool_caller):
        assert tool_caller.auto_invoke is False

        tool_caller.auto_invoke = True
        assert tool_caller.auto_invoke is True

    def test_make_request_params(self, tool_caller):
        request_params = {"param1": "value1"}
        result = tool_caller._make_request_params(request_params)

        assert isinstance(request_params, dict)
        assert request_params["param1"] == "value1"
        assert "tools" in result
        assert "tool_choice" in result

        request_params = {"model": "this should fail"}
        with pytest.raises(ValueError):
            tool_caller._make_request_params(request_params)

    def test_handle_tool_call_no_invoke(self, tool_caller, add3, response_with_tool_call):
        arguments = {"x": 1, "y": 2}
        tool_call = response_with_tool_call.choices[0].message.tool_calls[0]
        tool_call.function.arguments = json.dumps(arguments)

        result = tool_caller._handle_tool_call(conversation=create_autospec(Conversation), tool_call=tool_call)

        assert isinstance(result, BaseModel)
        assert result.x == arguments["x"]
        assert result.y == arguments["y"]
        assert result.model_dump() == add3.signature().model_validate(arguments).model_dump()

    def test_handle_tool_call_with_invoke(self, tool_caller, add3, response_with_tool_call):
        arguments = {"x": 1, "y": 2}
        tool_call = response_with_tool_call.choices[0].message.tool_calls[0]
        tool_call.function.arguments = json.dumps(arguments)

        tool_caller.auto_invoke = True
        result = tool_caller._handle_tool_call(conversation=create_autospec(Conversation), tool_call=tool_call)

        assert isinstance(result, ToolMessage)
        assert result.content == "6"

    def test_handle_tool_call_with_validation_error(self, tool_caller, add3, response_with_tool_call):
        arguments = {"x": 1, "z": None}
        tool_call = response_with_tool_call.choices[0].message.tool_calls[0]
        tool_call.function.arguments = json.dumps(arguments)

        tool_caller._chat_completions_create = Mock(return_value=create_autospec(ChatCompletion))
        tool_caller._handle_response = Mock(return_value="repaired_tool")

        result = tool_caller._handle_tool_call(
            conversation=Conversation(
                messages=[
                    Message(role="system", content="original instructions"),
                    Message(role="assistant", content=json.dumps({"name": "add3", "arguments": arguments})),
                    Message(role="system", content="fix this"),
                ]
            ),
            tool_call=tool_call,
        )
        assert result == "repaired_tool"

    def test_handle_tool_call_with_max_repair_attempts(self, tool_caller, add3, response_with_tool_call):
        arguments = {"x": 1, "z": None}
        tool_call = response_with_tool_call.choices[0].message.tool_calls[0]
        tool_call.function.arguments = json.dumps(arguments)

        tool_caller.max_repair_attempts = 1

        tool_caller._repair_tool = Mock(
            return_value=Conversation(
                messages=[
                    Message(role="assistant", content=json.dumps({"name": "add3", "arguments": arguments})),
                    Message(role="system", content="fix this"),
                ]
            )
        )
        tool_caller._chat_completions_create = Mock(return_value=response_with_tool_call)

        with pytest.raises(CallerValidationError):
            tool_caller._handle_tool_call(
                conversation=Conversation(
                    messages=[
                        Message(role="system", content="original instructions"),
                        Message(role="assistant", content=json.dumps({"name": "add3", "arguments": arguments})),
                        Message(role="system", content="fix this"),
                    ]
                ),
                tool_call=tool_call,
            )

    def test_validate_tool(self, tool_caller, add3, response_with_tool_call):
        arguments = {"x": 1, "y": 2}
        tool_call = response_with_tool_call.choices[0].message.tool_calls[0]
        tool_call.function.arguments = json.dumps(arguments)

        result = tool_caller._validate_tool(name="add3", arguments=json.dumps(arguments))

        assert isinstance(result, BaseModel)
        assert result.model_dump() == add3.signature().model_validate(arguments).model_dump()

    def test_repair_tool(self, tool_caller, response_with_tool_call):
        arguments = {"x": 1, "z": None}
        tool_call = response_with_tool_call.choices[0].message.tool_calls[0]
        tool_call.function.arguments = json.dumps(arguments)

        exception = "Validation error"
        conversation = tool_caller._repair_tool(tool_call, exception)

        assert isinstance(conversation, Conversation)
        assert len(conversation.messages) == 2
        assert conversation.messages[0].role == "assistant"
        assert conversation.messages[0].content == tool_call.function.model_dump_json()
        assert conversation.messages[1].role == "user"
        assert exception in conversation.messages[1].content
