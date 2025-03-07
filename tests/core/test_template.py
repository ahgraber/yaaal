import logging

from jinja2 import StrictUndefined, Template as JinjaTemplate
from jinja2.exceptions import UndefinedError
from pydantic import BaseModel, ValidationError, create_model
import pytest

from yaaal.core.template import (
    ConversationTemplate,
    JinjaMessageTemplate,
    MessageTemplate,
    StaticMessageTemplate,
    StringMessageTemplate,
    UserMessageTemplate,
)
from yaaal.types_.base import JSON
from yaaal.types_.core import Conversation, Message


class TestMessageTemplate:
    @pytest.fixture
    def test_template(self):
        class TestTemplate(MessageTemplate):
            def __init__(self, role: str, template: str):
                self.role = role
                self.template = template
                self.validation_model = None

        return TestTemplate

    def test_init(self, test_template):
        t = test_template(
            role="system",
            template="This is a test",
        )
        assert t.role == "system"
        assert t.template == "This is a test"
        assert t.validation_model is None

        with pytest.raises(NotImplementedError):
            t.render({})


class TestStaticMessageTemplate:
    def test_valid(self):
        t = StaticMessageTemplate(
            role="system",
            template="This is a test",
        )
        assert t.role == "system"
        assert t.template == "This is a test"
        assert t.validation_model is None

        assert t.render() == Message(role="system", content="This is a test")


class TestStringMessageTemplate:
    class Person(BaseModel):
        name: str
        age: int

    @pytest.fixture
    def template(self):
        return "Hi, my name is $name and I'm $age years old."

    def test_valid_template(self, template):
        t = StringMessageTemplate(
            role="user",
            template=template,
            validation_model=self.Person,
        )
        assert t.role == "user"
        assert t.validation_model == self.Person

        template_vars = {"name": "Bob", "age": 42}
        message = t.render(template_vars=template_vars)
        assert message.role == "user"
        assert message.content == "Hi, my name is Bob and I'm 42 years old."

    def test_missing_template_vars(self, template):
        t = StringMessageTemplate(
            role="user",
            template=template,
            validation_model=self.Person,
        )
        missing_vars = {"name": "Bob"}
        with pytest.raises(ValidationError):
            t.render(template_vars=missing_vars)

    def test_invalid_template_vars(self, template):
        t = StringMessageTemplate(
            role="user",
            template=template,
            validation_model=self.Person,
        )
        invalid_vars = {"name": "Bob", "age": "forty-two"}
        with pytest.raises(ValidationError):
            t.render(template_vars=invalid_vars)

    def test_no_validation(self, template):
        t = StringMessageTemplate(
            role="user",
            template=template,
        )
        template_vars = {"name": "Bob", "age": 42}
        message = t.render(template_vars=template_vars)
        assert message.role == "user"
        assert message.content == "Hi, my name is Bob and I'm 42 years old."

    def test_template_with_escaped_variables(self):
        """Test handling of escaped $ characters."""
        t = StringMessageTemplate(
            role="user",
            template="Cost: $$ vs ${amount}",
            validation_model=create_model("Vars", amount=(int, ...)),
        )
        msg = t.render({"amount": 42})
        assert msg.content == "Cost: $ vs 42"

    def test_model_dump_handling(self):
        """Test handling of BaseModel inputs."""

        t = StringMessageTemplate(
            role="user",
            template="$name: $age",
            validation_model=self.Person,
        )
        msg = t.render(self.Person(name="Bob", age=42).model_dump())
        assert msg.content == "Bob: 42"


class TestUserMessageTemplate:
    def test_valid_template(self):
        t = UserMessageTemplate()

        template_vars = {"user": "Hi, my name is Bob and I'm 42 years old."}
        message = t.render(template_vars=template_vars)
        assert message.role == "user"
        assert message.content == "Hi, my name is Bob and I'm 42 years old."

    def test_static_role(self):
        """UserMessageTemplate can only have 'user' role."""
        t = UserMessageTemplate(role="system")
        assert t.role == "user"

        template_vars = {"user": "Hi, my name is Bob and I'm 42 years old."}
        message = t.render(template_vars=template_vars)
        assert message.role == "user"
        assert message.content == "Hi, my name is Bob and I'm 42 years old."

    def test_missing_template_vars(self):
        t = UserMessageTemplate()
        with pytest.raises(ValidationError):
            t.render(template_vars={})

        with pytest.raises(TypeError):
            t.render()

    def test_invalid_template_vars(self):
        t = UserMessageTemplate()
        with pytest.raises(ValidationError):
            t.render(template_vars={"user": 42})  # content must be str

    def test_extra_template_vars(self):
        t = UserMessageTemplate()
        template_vars = {"user": "Hi, my name is Bob and I'm 42 years old.", "extra": "not used"}
        message = t.render(template_vars=template_vars)
        assert message.role == "user"
        assert message.content == "Hi, my name is Bob and I'm 42 years old."

    def test_model_dump_handling(self):
        """Test handling of BaseModel inputs."""

        class UserContent(BaseModel):
            user: str

        t = UserMessageTemplate()
        msg = t.render(UserContent(user="test"))
        assert msg.content == "test"


class TestJinjaMessageTemplate:
    class Person(BaseModel):
        name: str
        age: int

    @pytest.fixture
    def template(self):
        return "Hi, my name is {{name}} and I'm {{age}} years old."

    def test_valid_template(self, template):
        t = JinjaMessageTemplate(
            role="user",
            template=template,
            validation_model=self.Person,
        )

        template_vars = {"name": "Bob", "age": 42}
        message = t.render(template_vars=template_vars)
        assert message.role == "user"
        assert message.content == "Hi, my name is Bob and I'm 42 years old."

    def test_missing_template_vars(self, template):
        t = JinjaMessageTemplate(
            role="user",
            template=template,
            validation_model=self.Person,
        )
        missing_vars = {"name": "Bob"}
        with pytest.raises(ValidationError):
            t.render(template_vars=missing_vars)

    def test_invalid_template_vars(self, template):
        t = JinjaMessageTemplate(
            role="user",
            template=template,
            validation_model=self.Person,
        )
        invalid_vars = {"name": "Bob", "age": "forty-two"}
        with pytest.raises(ValidationError):
            t.render(template_vars=invalid_vars)

    def test_no_validation(self, template):
        t = JinjaMessageTemplate(
            role="user",
            template=template,
        )
        template_vars = {"name": "Bob", "age": 42}
        message = t.render(template_vars=template_vars)
        assert message.role == "user"
        assert message.content == "Hi, my name is Bob and I'm 42 years old."

    def test_undefined_variable(self, template):
        # If there's no pydantic validation, jinja should raise an error if a variable
        # in the template does not have a corresponding value during render
        t = JinjaMessageTemplate(
            role="user",
            template=template,
        )
        template_vars = {"name": "Bob"}
        with pytest.raises(UndefinedError):
            t.render(template_vars=template_vars)

    def test_jinja_extra_template_vars(self, template):
        # If there's no pydantic validation, jinja should ignore extra variable keys
        t = JinjaMessageTemplate(
            role="user",
            template=template,
        )
        template_vars = {"name": "Bob", "age": 42, "bonus": "this is a test"}
        message = t.render(template_vars=template_vars)
        assert message.role == "user"
        assert message.content == "Hi, my name is Bob and I'm 42 years old."

    def test_init_with_jinja_template(self):
        template = JinjaTemplate("Hello {{name}}", undefined=StrictUndefined)
        t = JinjaMessageTemplate(role="user", template=template)
        assert t.template == template
        assert t.template.environment.undefined == StrictUndefined

    def test_init_with_string(self):
        t = JinjaMessageTemplate(role="user", template="Hello {{name}}")
        assert isinstance(t.template, JinjaTemplate)
        assert t.template.environment.undefined == StrictUndefined

    def test_complex_template_rendering(self):
        template = """
        {% for item in items %}
        - {{item.name}}: {{item.value}}
        {% endfor %}
        Total: {{total}}
        """

        class Item(BaseModel):
            name: str
            value: int

        class TemplateVars(BaseModel):
            items: list[Item]
            total: int

        t = JinjaMessageTemplate(role="user", template=template, validation_model=TemplateVars)

        template_vars = {"items": [{"name": "A", "value": 1}, {"name": "B", "value": 2}], "total": 3}
        msg = t.render(template_vars)
        assert "- A: 1" in msg.content
        assert "- B: 2" in msg.content
        assert "Total: 3" in msg.content


class TestConversationTemplate:
    """Tests for ConversationTemplate class."""

    class Person(BaseModel):
        name: str
        age: int

    @pytest.fixture
    def system_template(self):
        return StaticMessageTemplate(
            role="system",
            template="You are a helpful assistant",
        )

    @pytest.fixture
    def user_template(self):
        return StringMessageTemplate(
            role="user",
            template="Hi, my name is $name and I'm $age years old.",
            validation_model=self.Person,
        )

    @pytest.fixture
    def assistant_message(self):
        return Message(
            role="assistant",
            content="I understand.",
        )

    @pytest.fixture
    def user_message(self):
        return Message(
            role="user",
            content="I understand.",
        )

    def test_init_and_signature(self, system_template, user_template):
        ct = ConversationTemplate(
            name="Test Template",
            description="A test template",
            conversation_spec=[system_template, user_template],
        )
        assert ct.name == "test_template"
        assert ct.description == "A test template"
        assert len(ct.conversation_spec) == 2
        signature = ct.signature
        assert issubclass(signature, BaseModel)
        assert ct.schema == signature.model_json_schema()

    def test_name_validation(self, caplog, system_template, user_template):
        """Test name validation and conversion."""
        expected = "test_template"

        # Test basic snake_case name passes through unchanged
        ct = ConversationTemplate(
            name="test_template",
            description="Test template",
            conversation_spec=[system_template, user_template],
        )
        assert ct.name == expected

        # Test PascalCase gets converted to snake_case
        with caplog.at_level(logging.WARNING):
            ct = ConversationTemplate(
                name="TestTemplate",
                description="Test template",
                conversation_spec=[system_template, user_template],
            )
        assert ct.name == expected
        assert "Converted template name 'TestTemplate' to 'test_template'" in caplog.text

        # Test camelCase gets converted to snake_case
        with caplog.at_level(logging.WARNING):
            ct = ConversationTemplate(
                name="testTemplate",
                description="Test template",
                conversation_spec=[system_template, user_template],
            )
        assert ct.name == expected
        assert "Converted template name 'testTemplate' to 'test_template'" in caplog.text

        # Test spaces get converted to underscores
        with caplog.at_level(logging.WARNING):
            ct = ConversationTemplate(
                name="Test template",
                description="Test template",
                conversation_spec=[system_template, user_template],
            )
        assert ct.name == expected
        assert "Converted template name 'TestTemplate' to 'test_template'" in caplog.text

    def test_description_validation(self):
        with pytest.raises(ValueError, match="Description must be provided as string for use as docstring"):
            ConversationTemplate(
                name="TestTemplate",
                description=None,
                conversation_spec=[StaticMessageTemplate(role="system", template="Test")],
            )

        with pytest.raises(ValueError, match="Description must be provided as string for use as docstring"):
            ConversationTemplate(
                name="TestTemplate",
                description="",
                conversation_spec=[StaticMessageTemplate(role="system", template="Test")],
            )

    def test_empty_conversation_list(self):
        with pytest.raises(ValueError, match="Conversation list cannot be empty"):
            ConversationTemplate(
                name="Empty",
                description="Empty template list",
                conversation_spec=[],
            )

    def test_no_system_message(self, user_template, assistant_message):
        # None of these have a system role matching "system"
        with pytest.raises(ValueError, match="Template list must contain at least one system message"):
            ConversationTemplate(
                name="NoSystem",
                description="Missing system message",
                conversation_spec=[user_template, assistant_message],
            )

    def test_invalid_conversation_item(self, system_template):
        # Pass an invalid type in the conversation list
        with pytest.raises(TypeError, match="Conversation list must contain only MessageTemplates or Messages"):
            ct = ConversationTemplate(
                name="InvalidType",
                description="Contains invalid message type",
                conversation_spec=[system_template, 123],
            )
            ct.render({})

    def test_render_with_mixed_messages(self, system_template, user_template, assistant_message):
        # Include a pre-rendered Message alongside templates.
        ct = ConversationTemplate(
            name="Mixed",
            description="Mixed conversation types",
            conversation_spec=[system_template, user_template, assistant_message],
        )
        # Provide variables that validate for user_template; static messages ignore variables.
        variables = {"name": "Alice", "age": 30}
        conv = ct.render(variables)
        assert len(conv.messages) == 3
        # Verify static template renders normally.
        assert conv.messages[0].content == system_template.template
        # Verify user template renders with substitution.
        assert conv.messages[1].content == "Hi, my name is Alice and I'm 30 years old."
        # Verify pre-rendered Message is preserved.
        assert conv.messages[2].content == assistant_message.content

    def test_render_complex_conversation(self):
        class SystemVars(BaseModel):
            instructions: str

        class UserVars(BaseModel):
            query: str

        class AssistantVars(BaseModel):
            response: str

        templates = [
            JinjaMessageTemplate(role="system", template="{{instructions}}", validation_model=SystemVars),
            JinjaMessageTemplate(role="user", template="Query: {{query}}", validation_model=UserVars),
            JinjaMessageTemplate(role="assistant", template="{{response}}", validation_model=AssistantVars),
            JinjaMessageTemplate(role="user", template="Query: {{query}}", validation_model=UserVars),
        ]

        ct = ConversationTemplate(
            name="Complex",
            description="Test complex conversation",
            conversation_spec=templates,
        )

        variables = {"instructions": "You are a helpful assistant.", "query": "hello", "response": "Hi there!"}
        conv = ct.render(variables)
        # Expect four messages as provided in templates.
        assert len(conv.messages) == 4
        assert conv.messages[0].content == "You are a helpful assistant."
        assert conv.messages[1].content == "Query: hello"
        assert conv.messages[2].content == "Hi there!"
        assert conv.messages[3].content == "Query: hello"
