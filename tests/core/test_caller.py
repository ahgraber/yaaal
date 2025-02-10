from __future__ import annotations

import json
import re
from typing import Pattern, Type
from unittest.mock import Mock, create_autospec, patch

from pydantic import BaseModel, create_model
import pytest

from aisuite import Client
from aisuite.framework import ChatCompletionResponse

from yaaal.core.base import CallableWithSignature
from yaaal.core.caller import (
    Caller,
    _make_structured_params,
    _make_tool_params,
    create_chat_caller,
    create_structured_caller,
    create_tool_caller,
)
from yaaal.core.exceptions import ValidationError
from yaaal.core.handler import CompositeHandler, ResponseHandler, ToolHandler
from yaaal.core.prompt import Prompt, StaticMessageTemplate, StringMessageTemplate
from yaaal.core.tool import tool
from yaaal.core.validator import PassthroughValidator, PydanticValidator, ToolValidator
from yaaal.types.core import AssistantMessage, Conversation, ToolResultMessage, UserMessage
from yaaal.types.openai_compat import (
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallFunction,
)


@pytest.fixture
def client():
    return create_autospec(Client)  # Mock external API


@pytest.fixture
def model():
    return "provider:gpt"


@pytest.fixture
def prompt():
    return Prompt(
        name="Simple prompt",
        description="A simple assistant",
        system_template=StaticMessageTemplate(role="system", template="You are a test assistant"),
        user_template=StringMessageTemplate(
            role="user",
            template="Quiz me on $topic",
            template_vars_model=create_model("user_vars", topic=(str, ...)),
        ),
    )


@pytest.fixture
def passthrough_handler(response_model):
    return ResponseHandler(validator=PassthroughValidator())


@pytest.fixture
def chat_response():
    return ChatCompletion(
        choices=[
            ChatCompletionChoice(
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    content="This is a test",
                    tool_calls=None,
                    refusal=None,
                ),
            )
        ]
    )


@pytest.fixture
def response_model():
    class TestModel(BaseModel):
        name: str
        age: int

    return TestModel


@pytest.fixture
def pydantic_handler(response_model):
    return ResponseHandler(validator=PydanticValidator(model=response_model))


@pytest.fixture
def structured_response():
    return ChatCompletion(
        choices=[
            ChatCompletionChoice(
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    content=json.dumps({"name": "Bob", "age": 42}),
                    tool_calls=None,
                    refusal=None,
                ),
            )
        ]
    )


@pytest.fixture
def add_tool():
    @tool
    def add3(x: int, y: int) -> int:
        """A fancy way to add."""
        return x + y + 3

    return add3


@pytest.fixture
def tool_handler(add_tool):
    return ToolHandler(validator=ToolValidator(toolbox=[add_tool]))


@pytest.fixture
def tool_response():
    return ChatCompletion(
        choices=[
            ChatCompletionChoice(
                finish_reason="tool_calls",
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call1",
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


class TestCallerInitialization:
    def test_init(self, client, model, prompt):
        caller = Caller(client=client, model=model, prompt=prompt, handler=Mock())
        assert caller.client == client
        assert caller.model == model
        assert caller.prompt == prompt
        assert caller.request_params == {}

    def test_make_request_params(self, client, model, prompt):
        params = {"temperature": 0.8}
        caller = Caller(
            client=client,
            model=model,
            prompt=prompt,
            handler=Mock(),
            request_params=params,
        )
        assert caller.request_params == params

    def test_make_request_params_with_model(self, client, model, prompt):
        with pytest.raises(ValueError):
            Caller(
                client=client,
                model=model,
                prompt=prompt,
                handler=Mock(),
                request_params={"model": model},
            )

    def test_signature(self, client, model, prompt, passthrough_handler):
        caller = Caller(
            client=client,
            model=model,
            prompt=prompt,
            handler=passthrough_handler,
        )
        assert caller.signature.model_json_schema() == caller.prompt.signature.model_json_schema()


class TestCallerRepairMechanism:
    def test_handle_with_repair_success(self, client, model, prompt, chat_response):
        mock_handler = Mock()
        mock_handler.process.return_value = "Success!"
        caller = Caller(
            client=client,
            model=model,
            prompt=prompt,
            handler=mock_handler,
        )
        conversation = prompt.render(user_vars={"topic": "rhymes with 'T'"})

        result = caller._handle_with_repair(conversation, chat_response)
        assert result == "Success!"
        mock_handler.process.assert_called_once_with(chat_response)

    def test_handle_with_repair_retry_success(self, client, model, prompt, chat_response):
        mock_handler = Mock()
        mock_handler.process.side_effect = [ValueError("Bad response"), "Fixed!"]
        mock_handler.repair = Mock(
            return_value=Conversation(
                messages=[AssistantMessage(content="invalid"), UserMessage(content="Please fix")]
            )
        )
        caller = Caller(
            client=client,
            model=model,
            prompt=prompt,
            handler=mock_handler,
        )
        conversation = prompt.render(user_vars={"topic": "rhymes with 'T'"})
        client.chat.completions.create.return_value = chat_response

        result = caller._handle_with_repair(conversation, chat_response)
        assert result == "Fixed!"
        assert len(conversation.messages) == 4  # initial + repair messages
        assert conversation.messages[-1].content == "Please fix"
        assert mock_handler.process.call_count == 2

    def test_handle_with_repair_max_attempts(self, client, model, prompt, chat_response):
        mock_handler = Mock()
        mock_handler.process.side_effect = [ValueError("Bad response")]
        mock_handler.repair = Mock(
            return_value=Conversation(
                messages=[AssistantMessage(content="invalid"), UserMessage(content="Please fix")]
            )
        )
        caller = Caller(
            client=client,
            model=model,
            prompt=prompt,
            handler=mock_handler,
        )
        conversation = prompt.render(user_vars={"topic": "rhymes with 'T'"})
        client.chat.completions.create.return_value = chat_response

        with pytest.raises(ValidationError, match="Max repair attempts reached"):
            caller._handle_with_repair(conversation, chat_response)

        assert mock_handler.process.call_count == caller.max_repair_attempts + 1

    def test_handle_with_repair_no_instructions(self, client, model, prompt, chat_response):
        mock_handler = Mock()
        mock_handler.process.side_effect = [ValueError("Bad response")]
        mock_handler.repair = Mock(return_value=None)
        caller = Caller(
            client=client,
            model=model,
            prompt=prompt,
            handler=mock_handler,
        )
        conversation = prompt.render(user_vars={"topic": "rhymes with 'T'"})

        with pytest.raises(ValidationError, match="No repair instructions available"):
            caller._handle_with_repair(conversation, chat_response)

        mock_handler.process.assert_called_once_with(chat_response)
        mock_handler.repair.assert_called_once()


class TestCallerRun:
    def test_with_passthrough_handler(self, client, model, prompt, passthrough_handler, chat_response):
        client.chat.completions.create.return_value = chat_response
        caller = Caller(
            client=client,
            model=model,
            prompt=prompt,
            handler=passthrough_handler,
        )

        result = caller(user_vars={"topic": "test input"})
        assert isinstance(result, str)
        assert result == "This is a test"

    def test_with_pydantic_handler(self, client, model, prompt, pydantic_handler, structured_response):
        client.chat.completions.create.return_value = structured_response
        caller = Caller(client=client, model=model, prompt=prompt, handler=pydantic_handler)

        result = caller(user_vars={"topic": "test input"})
        assert isinstance(result, BaseModel)
        assert result.model_dump() == {"name": "Bob", "age": 42}

    def test_with_tool_handler(self, client, model, prompt, tool_handler, tool_response):
        client.chat.completions.create.return_value = tool_response
        caller = Caller(
            client=client,
            model=model,
            prompt=prompt,
            handler=tool_handler,
        )

        result = caller(user_vars={"topic": "test input"})
        assert isinstance(result, BaseModel)
        assert result.model_dump() == {"x": 1, "y": 2}

    def test_with_tool_handler_invoke(self, client, model, prompt, tool_handler, tool_response):
        client.chat.completions.create.return_value = tool_response
        tool_handler.auto_invoke = True
        caller = Caller(
            client=client,
            model=model,
            prompt=prompt,
            handler=tool_handler,
        )

        result = caller(user_vars={"topic": "test input"})
        assert isinstance(result, str)
        assert result == str(6)  # Changed from ToolResultMessage check

    def test_call_with_existing_conversation(self, client, model, prompt, passthrough_handler, chat_response):
        conversation = prompt.render(user_vars={"topic": "initial topic"})

        client.chat.completions.create.return_value = chat_response

        caller = Caller(
            client=client,
            model=model,
            prompt=prompt,
            handler=passthrough_handler,
        )

        result = caller(conversation=conversation, user_vars={"topic": "second topic"})
        assert len(conversation.messages) == 4
        msgs = [m.content for m in conversation.messages]
        assert "initial topic" in msgs[1]  # 0, system; 1, user
        assert "second topic" in msgs[3]  # 2, system; 3, user
        assert result == "This is a test"


class TestCallerFactories:
    def test_create_chat_caller(self, client, model, prompt):
        caller = create_chat_caller(client, model, prompt)
        assert isinstance(caller.handler, ResponseHandler)
        assert isinstance(caller.handler.validator, PassthroughValidator)

    def test_create_structured_caller(self, client, model, prompt, response_model):
        caller = create_structured_caller(client, model, prompt, response_model)
        assert isinstance(caller.handler, ResponseHandler)
        assert isinstance(caller.handler.validator, PydanticValidator)

    def test_create_tool_caller(self, client, model, prompt, add_tool):
        caller = create_tool_caller(client, model, prompt, [add_tool])
        assert isinstance(caller.handler, ToolHandler)
        assert isinstance(caller.handler.validator, ToolValidator)


class TestCallerHelpers:
    def test_make_structured_params_anthropic(self):
        params = _make_structured_params(model="anthropic:baube")
        assert params == {"response_format": {"type": "json"}}

    def test_make_structured_params_openai(self):
        params = _make_structured_params("openai:mdl")
        assert params == {"response_format": {"type": "json_object"}}

    def test_make_tool_params_anthropic(self, add_tool):
        params = _make_tool_params("anthropic:baube", [add_tool])
        assert "tools" in params
        assert params["tool_choice"] == {"type": "auto"}

    def test_make_tool_params_openai(self, add_tool):
        params = _make_tool_params("openai:mdl", [add_tool])
        assert "tools" in params
        assert params["tool_choice"] == "auto"
