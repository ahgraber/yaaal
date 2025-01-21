from pydantic import BaseModel, ValidationError, create_model
import pytest

from yaaal.core._types import JSON, Conversation, Message


class TestMessage:
    def test_valid_message(self):
        message = Message(role="user", content="Hello")
        assert message.role == "user"
        assert message.content == "Hello"

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError) as exc_info:
            Message(role="user")
        assert "content" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Message(content="Hello")
        assert "role" in str(exc_info.value)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError) as exc_info:
            Message(role="user", content="Hello", extra_field="not allowed")
        assert "extra inputs are not permitted" in str(exc_info.value).lower()

    def test_valid_role_values(self):
        roles = ["system", "user", "assistant", "function"]
        for role in roles:
            message = Message(role=role, content="test content")
            assert message.role == role
            assert message.content == "test content"

    def test_invalid_types(self):
        with pytest.raises(ValidationError):
            Message(role=123, content="Hello")  # role must be str

        with pytest.raises(ValidationError):
            Message(role="user", content=["not", "a", "string"])  # content must be str

    def test_empty_strings(self):
        with pytest.raises(ValidationError):
            Message(role="user", content="")

        with pytest.raises(ValidationError):
            Message(role="", content="Hello")


class TestConversation:
    def test_valid_conversation(self):
        message1 = Message(role="user", content="Hello")
        message2 = Message(role="assistant", content="Hi there!")

        conversation = Conversation(messages=[message1, message2])

        assert len(conversation.messages) == 2
        assert conversation.messages[0].role == "user"
        assert conversation.messages[0].content == "Hello"
        assert conversation.messages[1].role == "assistant"
        assert conversation.messages[1].content == "Hi there!"

    def test_empty_messages(self):
        with pytest.raises(ValidationError):
            Conversation(messages=[])

    def test_blank_init(self):
        with pytest.raises(ValidationError):
            Conversation()
