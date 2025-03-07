import json
from typing import Literal

from pydantic import BaseModel
import pytest

from aisuite.framework.chat_completion_response import ChatCompletionResponse as AISuiteChatCompletion
from aisuite.framework.choice import Choice as AISuiteChatCompletionChoice
from aisuite.framework.message import (
    ChatCompletionMessageToolCall as AISuiteChatCompletionMessageToolCall,
    Function as AISuiteChatCompletionMessageToolCallFunction,
    Message as AISuiteChatCompletionMessage,
)
from openai.types.chat.chat_completion import (
    ChatCompletion as OpenAIChatCompletion,
    Choice as OpenAIChatCompletionChoice,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage as OpenAIChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall as OpenAIChatCompletionMessageToolCall,
    Function as OpenAIChatCompletionMessageToolCallFunction,
)

from yaaal.types_.openai_compat import (
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallFunction,
    convert_response,
)


@pytest.fixture
def content():
    return ["raindrops on roses", "whiskers on kittens", "warm woolen mittens"]


@pytest.fixture
def tool_calls():
    return [
        {"name": "add1", "arguments": json.dumps({"x": 1})},
        {"name": "add2", "arguments": json.dumps({"x": 1, "y": 2})},
        {"name": "add3", "arguments": json.dumps({"x": 1, "y": 2, "z": 3})},
    ]


def test_openai_content_completion(content):
    completion = OpenAIChatCompletion(
        id="test123",
        created=1234567,
        model="openai:gpt-fake",
        object="chat.completion",
        choices=[
            OpenAIChatCompletionChoice(
                finish_reason="stop",
                index=0,
                message=OpenAIChatCompletionMessage(role="assistant", content=content[0]),
            )
        ],
    )

    converted = convert_response(completion)
    assert isinstance(converted, ChatCompletion)
    assert isinstance(converted.choices[0], ChatCompletionChoice)
    assert isinstance(converted.choices[0].message, ChatCompletionMessage)

    assert converted.choices[0].finish_reason == "stop"
    assert converted.choices[0].message.role == "assistant"
    assert converted.choices[0].message.content == content[0]


def test_openai_multicontent_completion(content):
    completion = OpenAIChatCompletion(
        id="test123",
        created=1234567,
        model="openai:gpt-fake",
        object="chat.completion",
        choices=[
            OpenAIChatCompletionChoice(
                finish_reason="stop",
                index=0,
                message=OpenAIChatCompletionMessage(role="assistant", content=content[0]),
            ),
            OpenAIChatCompletionChoice(
                finish_reason="stop",
                index=1,
                message=OpenAIChatCompletionMessage(role="assistant", content=content[1]),
            ),
            OpenAIChatCompletionChoice(
                finish_reason="stop",
                index=2,
                message=OpenAIChatCompletionMessage(role="assistant", content=content[2]),
            ),
        ],
    )

    converted = convert_response(completion)
    assert isinstance(converted, ChatCompletion)

    for i, c in enumerate(content):
        assert isinstance(converted.choices[i], ChatCompletionChoice)
        assert isinstance(converted.choices[i].message, ChatCompletionMessage)

        assert converted.choices[i].finish_reason == "stop"
        assert converted.choices[i].message.role == "assistant"
        assert converted.choices[i].message.content == c


def test_openai_tool_completion(tool_calls):
    completion = OpenAIChatCompletion(
        id="test123",
        created=1234567,
        model="openai:gpt-fake",
        object="chat.completion",
        choices=[
            OpenAIChatCompletionChoice(
                finish_reason="tool_calls",
                index=0,
                message=OpenAIChatCompletionMessage(
                    role="assistant",
                    tool_calls=[
                        OpenAIChatCompletionMessageToolCall(
                            id="tool42",
                            function=OpenAIChatCompletionMessageToolCallFunction(**tool_calls[0]),
                            type="function",
                        )
                    ],
                ),
            )
        ],
    )

    converted = convert_response(completion)
    assert isinstance(converted, ChatCompletion)
    assert isinstance(converted.choices[0], ChatCompletionChoice)
    assert isinstance(converted.choices[0].message, ChatCompletionMessage)
    assert isinstance(converted.choices[0].message.tool_calls[0], ChatCompletionMessageToolCall)
    assert isinstance(converted.choices[0].message.tool_calls[0].function, ChatCompletionMessageToolCallFunction)

    assert converted.choices[0].finish_reason == "tool_calls"
    assert converted.choices[0].message.role == "assistant"
    assert converted.choices[0].message.content is None
    assert converted.choices[0].message.tool_calls[0].function == ChatCompletionMessageToolCallFunction(
        **tool_calls[0]
    )


def test_openai_multitool_completion(tool_calls):
    completion = OpenAIChatCompletion(
        id="test123",
        created=1234567,
        model="openai:gpt-fake",
        object="chat.completion",
        choices=[
            OpenAIChatCompletionChoice(
                finish_reason="tool_calls",
                index=0,
                message=OpenAIChatCompletionMessage(
                    role="assistant",
                    tool_calls=[
                        OpenAIChatCompletionMessageToolCall(
                            id="tool42",
                            function=OpenAIChatCompletionMessageToolCallFunction(**tool_calls[0]),
                            type="function",
                        ),
                        OpenAIChatCompletionMessageToolCall(
                            id="order66",
                            function=OpenAIChatCompletionMessageToolCallFunction(**tool_calls[1]),
                            type="function",
                        ),
                        OpenAIChatCompletionMessageToolCall(
                            id="function1",
                            function=OpenAIChatCompletionMessageToolCallFunction(**tool_calls[2]),
                            type="function",
                        ),
                    ],
                ),
            )
        ],
    )

    converted = convert_response(completion)
    assert isinstance(converted, ChatCompletion)
    assert isinstance(converted.choices[0], ChatCompletionChoice)
    assert isinstance(converted.choices[0].message, ChatCompletionMessage)

    assert converted.choices[0].finish_reason == "tool_calls"
    assert converted.choices[0].message.role == "assistant"
    assert converted.choices[0].message.content is None

    for i, tc in enumerate(tool_calls):
        assert isinstance(converted.choices[0].message.tool_calls[i], ChatCompletionMessageToolCall)
        assert isinstance(converted.choices[0].message.tool_calls[i].function, ChatCompletionMessageToolCallFunction)
        assert converted.choices[0].message.tool_calls[i].function == ChatCompletionMessageToolCallFunction(**tc)


def test_aisuite_content_completion(content):
    choice = AISuiteChatCompletionChoice()
    choice.finish_reason = "stop"
    choice.message = AISuiteChatCompletionMessage(
        role="assistant",
        content=content[0],
        tool_calls=None,
        refusal=None,
    )
    completion = AISuiteChatCompletion()
    completion.choices = [choice]

    converted = convert_response(completion)
    assert isinstance(converted, ChatCompletion)
    assert isinstance(converted.choices[0], ChatCompletionChoice)
    assert isinstance(converted.choices[0].message, ChatCompletionMessage)

    assert converted.choices[0].finish_reason == "stop"
    assert converted.choices[0].message.role == "assistant"
    assert converted.choices[0].message.content == content[0]


def test_aisuite_multicontent_completion(content):
    choices = []
    for c in content:
        choice = AISuiteChatCompletionChoice()
        choice.finish_reason = "stop"
        choice.message = AISuiteChatCompletionMessage(
            role="assistant",
            content=c,
            tool_calls=None,
            refusal=None,
        )
        choices.append(choice)
    completion = AISuiteChatCompletion()
    completion.choices = choices

    converted = convert_response(completion)
    assert isinstance(converted, ChatCompletion)

    for i, c in enumerate(content):
        assert isinstance(converted.choices[i], ChatCompletionChoice)
        assert isinstance(converted.choices[i].message, ChatCompletionMessage)

        assert converted.choices[i].finish_reason == "stop"
        assert converted.choices[i].message.role == "assistant"
        assert converted.choices[i].message.content == c


def test_aisuite_tool_completion(tool_calls):
    choice = AISuiteChatCompletionChoice()
    choice.finish_reason = "tool_calls"
    choice.message = AISuiteChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[
            AISuiteChatCompletionMessageToolCall(
                id="tool42",
                function=AISuiteChatCompletionMessageToolCallFunction(**tool_calls[0]),
                type="function",
            )
        ],
        refusal=None,
    )
    completion = AISuiteChatCompletion()
    completion.choices = [choice]

    converted = convert_response(completion)
    assert isinstance(converted, ChatCompletion)
    assert isinstance(converted.choices[0], ChatCompletionChoice)
    assert isinstance(converted.choices[0].message, ChatCompletionMessage)
    assert isinstance(converted.choices[0].message.tool_calls[0], ChatCompletionMessageToolCall)
    assert isinstance(converted.choices[0].message.tool_calls[0].function, ChatCompletionMessageToolCallFunction)

    assert converted.choices[0].finish_reason == "tool_calls"
    assert converted.choices[0].message.role == "assistant"
    assert converted.choices[0].message.content is None
    assert converted.choices[0].message.tool_calls[0].function == ChatCompletionMessageToolCallFunction(
        **tool_calls[0]
    )


def test_aisuite_multitool_completion(tool_calls):
    choice = AISuiteChatCompletionChoice()
    choice.finish_reason = "tool_calls"
    choice.message = AISuiteChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[
            AISuiteChatCompletionMessageToolCall(
                id="tool42",
                function=AISuiteChatCompletionMessageToolCallFunction(**tc),
                type="function",
            )
            for tc in tool_calls
        ],
        refusal=None,
    )
    completion = AISuiteChatCompletion()
    completion.choices = [choice]

    converted = convert_response(completion)
    assert isinstance(converted, ChatCompletion)
    assert isinstance(converted.choices[0], ChatCompletionChoice)
    assert isinstance(converted.choices[0].message, ChatCompletionMessage)

    assert converted.choices[0].finish_reason == "tool_calls"
    assert converted.choices[0].message.role == "assistant"
    assert converted.choices[0].message.content is None

    for i, tc in enumerate(tool_calls):
        assert isinstance(converted.choices[0].message.tool_calls[i], ChatCompletionMessageToolCall)
        assert isinstance(converted.choices[0].message.tool_calls[i].function, ChatCompletionMessageToolCallFunction)
        assert converted.choices[0].message.tool_calls[i].function == ChatCompletionMessageToolCallFunction(**tc)
