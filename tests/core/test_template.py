import json
import re
import textwrap

from jinja2 import StrictUndefined, Template as JinjaTemplate
from jinja2.exceptions import UndefinedError
from pydantic import BaseModel, ValidationError, create_model
import pytest

from yaaal.core.template import (
    ConversationTemplate,
    JinjaMessageTemplate,
    MessageTemplate,
    PassthroughMessageTemplate,
    StaticMessageTemplate,
    StringMessageTemplate,
)
from yaaal.types.base import JSON
from yaaal.types.core import Conversation, Message


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
        assert t.name is None

        assert t.render() == Message(role="system", content="This is a test")

    def test_name_normalization(self):
        """Test that template names are properly snake_cased."""
        t = StaticMessageTemplate(
            name="My Template",
            role="system",
            template="test",
        )
        assert t.name == "my_template"

    def test_none_name_handling(self):
        """Test that None name is preserved."""
        t = StaticMessageTemplate(
            name=None,
            role="system",
            template="test",
        )
        assert t.name is None


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
        assert t.name is None

        template_vars = {"name": "Bob", "age": 42}
        message = t.render(template_vars=template_vars)
        assert message.role == "user"
        assert message.content == "Hi, my name is Bob and I'm 42 years old."

    def test_name_normalization(self, template):
        """Test that template names are properly snake_cased."""
        t = StringMessageTemplate(
            name="My Template",
            role="system",
            template=template,
        )
        assert t.name == "my_template"

    def test_none_name_handling(self, template):
        """Test that None name is preserved."""
        t = StringMessageTemplate(
            name=None,
            role="system",
            template=template,
        )
        assert t.name is None

    def test_missing_template_vars(self, template):
        template = StringMessageTemplate(
            role="user",
            template=template,
            validation_model=self.Person,
        )
        missing_vars = {"name": "Bob"}
        with pytest.raises(ValidationError):
            template.render(template_vars=missing_vars)

    def test_invalid_template_vars(self, template):
        template = StringMessageTemplate(
            role="user",
            template=template,
            validation_model=self.Person,
        )
        invalid_vars = {"name": "Bob", "age": "forty-two"}
        with pytest.raises(ValidationError):
            template.render(template_vars=invalid_vars)

    def test_extra_template_vars(self, template):
        # whether extra vars are permitted is a validation_model setting in the pydantic BaseModel
        # TODO: In the future, consider whether a MessageTemplate should mandate "extra='forbid'"
        pass

    def test_no_validation(self, template):
        template = StringMessageTemplate(
            role="user",
            template=template,
        )
        template_vars = {"name": "Bob", "age": 42}
        message = template.render(template_vars=template_vars)
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


class TestPassthroughMessageTemplate:
    def test_valid_template(self):
        t = PassthroughMessageTemplate()
        assert t.name is None

        template_vars = {"content": "Hi, my name is Bob and I'm 42 years old."}
        message = t.render(template_vars=template_vars)
        assert message.role == "user"
        assert message.content == "Hi, my name is Bob and I'm 42 years old."

    def test_name_normalization(self):
        """Test that template names are properly snake_cased."""
        t = PassthroughMessageTemplate(name="My Template")
        assert t.name == "my_template"

    def test_none_name_handling(self):
        """Test that None name is preserved."""
        t = PassthroughMessageTemplate(name=None)
        assert t.name is None

    def test_static_role(self):
        """PassthroughMessageTemplate can only have 'user' role."""
        t = PassthroughMessageTemplate(role="system")
        assert t.role == "user"

        template_vars = {"content": "Hi, my name is Bob and I'm 42 years old."}
        message = t.render(template_vars=template_vars)
        assert message.role == "user"
        assert message.content == "Hi, my name is Bob and I'm 42 years old."

    def test_missing_template_vars(self):
        t = PassthroughMessageTemplate()
        with pytest.raises(ValidationError):
            t.render(template_vars={})

        with pytest.raises(TypeError):
            t.render()

    def test_invalid_template_vars(self):
        t = PassthroughMessageTemplate()
        with pytest.raises(ValidationError):
            t.render(template_vars={"content": 42})  # content must be str

    def test_extra_template_vars(self):
        t = PassthroughMessageTemplate()
        template_vars = {"content": "Hi, my name is Bob and I'm 42 years old.", "extra": "not used"}
        message = t.render(template_vars=template_vars)
        assert message.role == "user"
        assert message.content == "Hi, my name is Bob and I'm 42 years old."

    def test_name_preservation(self):
        """Test that name is properly handled."""
        t = PassthroughMessageTemplate(name="user_input")
        assert t.name == "user_input"
        assert t.validation_model.__name__ == "user_input"

        u = PassthroughMessageTemplate(name=None)
        assert u.name is None
        assert u.validation_model.__name__ == "PassthroughModel"

    def test_model_dump_handling(self):
        """Test handling of BaseModel inputs."""

        class Content(BaseModel):
            content: str

        t = PassthroughMessageTemplate()
        msg = t.render(Content(content="test"))
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
        assert t.name is None

        template_vars = {"name": "Bob", "age": 42}
        message = t.render(template_vars=template_vars)
        assert message.role == "user"
        assert message.content == "Hi, my name is Bob and I'm 42 years old."

    def test_name_normalization(self, template):
        """Test that template names are properly snake_cased."""
        t = StringMessageTemplate(
            name="My Template",
            role="system",
            template=template,
        )
        assert t.name == "my_template"

    def test_none_name_handling(self, template):
        """Test that None name is preserved."""
        t = StringMessageTemplate(
            name=None,
            role="system",
            template=template,
        )
        assert t.name is None

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

    def test_extra_template_vars(self, template):
        # whether extra vars are permitted is a validation_model setting in the pydantic BaseModel
        # TODO: In the future, consider whether a MessageTemplate should mandate "extra='forbid'"
        pass

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
            name="system_template",
            role="system",
            template="You are a helpful assistant",
        )

    @pytest.fixture
    def user_template(self):
        return StringMessageTemplate(
            name="user_template",
            role="user",
            template="Hi, my name is $name and I'm $age years old.",
            validation_model=self.Person,
        )

    @pytest.fixture
    def assistant_message(self):
        return StaticMessageTemplate(
            name="assistant message",  # converted to assistant_message
            role="assistant",
            template="I understand.",
        )

    def test_init(self, system_template, user_template):
        ct = ConversationTemplate(
            name="Test Template",
            description="A test template",
            templates=[system_template, user_template],
        )
        assert ct.name == "test_template"
        assert ct.description == "A test template"
        assert len(ct.templates) == 2
        assert "system_template" in ct.templates
        assert "user_template" in ct.templates

    def test_template_validation_transforms(self):
        """Test that validation returns transformed templates."""
        template = StaticMessageTemplate(name="system template", role="system", template="Hello")

        ct = ConversationTemplate(
            name="Test",
            description="Test",
            templates=[template],
        )

        assert isinstance(ct.templates, dict)
        # to_snake_case applied
        # templates are dict keyed by name
        assert ct.templates["system_template"] == template

    def test_empty_templates(self):
        """Test initialization with empty template list."""
        with pytest.raises(ValueError):
            ConversationTemplate(
                name="Empty",
                description="Empty template list",
                templates=[],
            )

    def test_invalid_template_without_name(self):
        with pytest.raises(ValueError):
            ConversationTemplate(
                name="Test",
                description="Test",
                templates=[
                    StaticMessageTemplate(
                        name=None,  # Missing name
                        role="system",
                        template="Static message",
                    )
                ],
            )
        with pytest.raises(ValueError):
            ConversationTemplate(
                name="Test",
                description="Test",
                templates=[
                    StringMessageTemplate(
                        name=None,  # Missing name
                        role="user",
                        template="Hello $name",
                        validation_model=self.Person,
                    )
                ],
            )

    def test_duplicate_template_names(self, user_template):
        """Test handling of multiple templates with same name."""
        templates = [
            user_template,
            PassthroughMessageTemplate(
                name="user template",  # Same name as user_template
                role="user",
                template="Another message",
                validation_model=self.Person,
            ),
        ]

        with pytest.raises(ValueError):
            ConversationTemplate(
                name="Test",
                description="Test",
                templates=templates,
            )

    def test_validate_templates_static(self):
        """Test that static templates don't need validation model."""
        ct = ConversationTemplate(
            name="Test",
            description="Test",
            templates=[
                StaticMessageTemplate(
                    name="static template",
                    role="system",
                    template="Static message",
                )
            ],
        )
        assert "static_template" in ct.templates

    def test_invalid_template_without_model(self):
        with pytest.raises(ValueError):
            ConversationTemplate(
                name="Test",
                description="Test",
                templates=[
                    StringMessageTemplate(
                        name="user",
                        role="user",
                        template="Hello $name",
                        validation_model=None,  # Missing model
                    )
                ],
            )

    def test_signature_generation(self, system_template, user_template):
        ct = ConversationTemplate(
            name="Test",
            description="Test",
            templates=[system_template, user_template],
        )

        signature = ct.signature
        assert issubclass(signature, BaseModel)

    def test_signature_schema_generation(self, system_template, user_template):
        """Test schema generation details."""
        ct = ConversationTemplate(
            name="Test",
            description="Test",
            templates=[system_template, user_template],
        )

        assert ct.schema == ct.signature.model_json_schema()

        # identify templates with validators
        with_validators = [
            template for name, template in ct.templates.items() if not isinstance(template, StaticMessageTemplate)
        ]
        # check that each template has a $def
        for template in with_validators:
            assert f"{template.name}_MessageSpec" in ct.schema["$defs"]

    def test_render(self, user_template):
        conversation_spec = [
            {"user_template": {"name": "Bob", "age": 42}},
        ]
        ct = ConversationTemplate(
            name="test",
            description="Test conversation",
            templates=[user_template],
        )
        conv = ct.render(conversation_spec)

        assert len(conv.messages) == 1
        assert conv.messages[0].content == "Hi, my name is Bob and I'm 42 years old."

    def test_render_with_static_message(self, system_template, assistant_message):
        conversation_spec = [{"system_template": {}}, {"assistant_message": None}]
        ct = ConversationTemplate(
            name="Test",
            description="Test",
            templates=[system_template, assistant_message],
        )

        conv = ct.render(conversation_spec)

        assert len(conv.messages) == len(conversation_spec)
        assert conv.messages[0].content == "You are a helpful assistant"
        assert conv.messages[1].content == "I understand."

    def test_render_complex_conversation(self):
        class SystemVars(BaseModel):
            instructions: str

        class UserVars(BaseModel):
            query: str

        class AssistantVars(BaseModel):
            response: str

        templates = [
            JinjaMessageTemplate(
                name="system_template", role="system", template="{{instructions}}", validation_model=SystemVars
            ),
            JinjaMessageTemplate(
                name="user_template", role="user", template="Query: {{query}}", validation_model=UserVars
            ),
            JinjaMessageTemplate(
                name="assistant_template", role="assistant", template="{{response}}", validation_model=AssistantVars
            ),
        ]

        ct = ConversationTemplate(name="Complex", description="Test complex conversation", templates=templates)

        conversation_spec = [
            {"system_template": {"instructions": "You are a helpful assistant."}},
            {"user_template": {"query": "hello"}},
            {"assistant_template": {"response": "Hi there!"}},
            {"user_template": {"query": "who are you?"}},
        ]
        conv = ct.render(conversation_spec)

        assert len(conv.messages) == len(conversation_spec)
        assert conv.messages[0].content == "You are a helpful assistant."
        assert conv.messages[1].content == "Query: hello"
        assert conv.messages[2].content == "Hi there!"
        assert conv.messages[3].content == "Query: who are you?"
