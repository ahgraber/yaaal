import json
import re
import textwrap

from jinja2 import StrictUndefined, Template as JinjaTemplate
from jinja2.exceptions import UndefinedError
from pydantic import BaseModel, ValidationError, create_model
import pytest

from yaaal.core.prompt import (
    JinjaMessageTemplate,
    MessageTemplate,
    PassthroughMessageTemplate,
    Prompt,
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
                self.template_vars_model = None

        return TestTemplate

    def test_init(self, test_template):
        t = test_template(
            role="system",
            template="This is a test",
        )
        assert t.role == "system"
        assert t.template == "This is a test"
        assert t.template_vars_model is None

        with pytest.raises(NotImplementedError):
            t.render_message({})


class TestStaticMessageTemplate:
    def test_valid(self):
        t = StaticMessageTemplate(
            role="system",
            template="This is a test",
        )
        assert t.role == "system"
        assert t.template == "This is a test"
        assert t.template_vars_model is None

        assert t.render_message() == Message(role="system", content="This is a test")


class TestStringMessageTemplate:
    class TemplateVarsModel(BaseModel):
        name: str
        age: int

    @pytest.fixture
    def template(self):
        return "Hi, my name is $name and I'm $age years old."

    def test_valid_template(self, template):
        template = StringMessageTemplate(
            role="user",
            template=template,
            template_vars_model=self.TemplateVarsModel,
        )
        template_vars = {"name": "Bob", "age": 42}
        message = template.render_message(template_vars=template_vars)
        assert message.role == "user"
        assert message.content == "Hi, my name is Bob and I'm 42 years old."

    def test_missing_template_vars(self, template):
        template = StringMessageTemplate(
            role="user",
            template=template,
            template_vars_model=self.TemplateVarsModel,
        )
        missing_vars = {"name": "Bob"}
        with pytest.raises(ValidationError):
            template.render_message(template_vars=missing_vars)

    def test_invalid_template_vars(self, template):
        template = StringMessageTemplate(
            role="user",
            template=template,
            template_vars_model=self.TemplateVarsModel,
        )
        invalid_vars = {"name": "Bob", "age": "forty-two"}
        with pytest.raises(ValidationError):
            template.render_message(template_vars=invalid_vars)

    def test_extra_template_vars(self, template):
        # whether extra vars are permitted is a template_vars_model setting in the pydantic BaseModel
        # TODO: In the future, consider whether a MessageTemplate should mandate "extra='forbid'"
        pass

    def test_no_validation(self, template):
        template = StringMessageTemplate(
            role="user",
            template=template,
        )
        template_vars = {"name": "Bob", "age": 42}
        message = template.render_message(template_vars=template_vars)
        assert message.role == "user"
        assert message.content == "Hi, my name is Bob and I'm 42 years old."


class TestPassthroughMessageTemplate:
    def test_valid_template(self):
        t = PassthroughMessageTemplate()
        template_vars = {"content": "Hi, my name is Bob and I'm 42 years old."}
        message = t.render_message(template_vars=template_vars)
        assert message.role == "user"
        assert message.content == "Hi, my name is Bob and I'm 42 years old."

    def test_static_role(self):
        """PassthroughMessageTemplate can only have 'user' role."""
        t = PassthroughMessageTemplate(role="system")
        assert t.role == "user"

        template_vars = {"content": "Hi, my name is Bob and I'm 42 years old."}
        message = t.render_message(template_vars=template_vars)
        assert message.role == "user"
        assert message.content == "Hi, my name is Bob and I'm 42 years old."

    def test_missing_template_vars(self):
        t = PassthroughMessageTemplate()
        with pytest.raises(ValidationError):
            t.render_message(template_vars={})

        with pytest.raises(TypeError):
            t.render_message()

    def test_invalid_template_vars(self):
        t = PassthroughMessageTemplate()
        with pytest.raises(ValidationError):
            t.render_message(template_vars={"content": 42})  # content must be str

    def test_extra_template_vars(self):
        t = PassthroughMessageTemplate()
        template_vars = {"content": "Hi, my name is Bob and I'm 42 years old.", "extra": "not used"}
        message = t.render_message(template_vars=template_vars)
        assert message.role == "user"
        assert message.content == "Hi, my name is Bob and I'm 42 years old."

    def test_no_validation(self):
        t = PassthroughMessageTemplate()
        t.template_vars_model = None  # Disable validation
        template_vars = {"content": "Hi, my name is Bob and I'm 42 years old."}
        message = t.render_message(template_vars=template_vars)
        assert message.role == "user"
        assert message.content == "Hi, my name is Bob and I'm 42 years old."


class TestJinjaMessageTemplate:
    class TemplateVarsModel(BaseModel):
        name: str
        age: int

    @pytest.fixture
    def template(self):
        return "Hi, my name is {{name}} and I'm {{age}} years old."

    def test_valid_template(self, template):
        t = JinjaMessageTemplate(
            role="user",
            template=template,
            template_vars_model=self.TemplateVarsModel,
        )
        template_vars = {"name": "Bob", "age": 42}
        message = t.render_message(template_vars=template_vars)
        assert message.role == "user"
        assert message.content == "Hi, my name is Bob and I'm 42 years old."

    def test_missing_template_vars(self, template):
        t = JinjaMessageTemplate(
            role="user",
            template=template,
            template_vars_model=self.TemplateVarsModel,
        )
        missing_vars = {"name": "Bob"}
        with pytest.raises(ValidationError):
            t.render_message(template_vars=missing_vars)

    def test_invalid_template_vars(self, template):
        t = JinjaMessageTemplate(
            role="user",
            template=template,
            template_vars_model=self.TemplateVarsModel,
        )
        invalid_vars = {"name": "Bob", "age": "forty-two"}
        with pytest.raises(ValidationError):
            t.render_message(template_vars=invalid_vars)

    def test_extra_template_vars(self, template):
        # whether extra vars are permitted is a template_vars_model setting in the pydantic BaseModel
        # TODO: In the future, consider whether a MessageTemplate should mandate "extra='forbid'"
        pass

    def test_no_validation(self, template):
        t = JinjaMessageTemplate(
            role="user",
            template=template,
        )
        template_vars = {"name": "Bob", "age": 42}
        message = t.render_message(template_vars=template_vars)
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
            t.render_message(template_vars=template_vars)

    def test_jinja_extra_template_vars(self, template):
        # If there's no pydantic validation, jinja should ignore extra variable keys
        t = JinjaMessageTemplate(
            role="user",
            template=template,
        )
        template_vars = {"name": "Bob", "age": 42, "bonus": "this is a test"}
        message = t.render_message(template_vars=template_vars)
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

        t = JinjaMessageTemplate(role="user", template=template, template_vars_model=TemplateVars)

        template_vars = {"items": [{"name": "A", "value": 1}, {"name": "B", "value": 2}], "total": 3}
        msg = t.render_message(template_vars)
        assert "- A: 1" in msg.content
        assert "- B: 2" in msg.content
        assert "Total: 3" in msg.content


class TestPrompt:
    class TemplateVarsModel(BaseModel):
        name: str
        age: int

    @pytest.fixture
    def system_template(self, system_message):
        return StaticMessageTemplate(
            role="system",
            template="You are a helpful assistant",
        )

    @pytest.fixture
    def system_message(self):
        return Message(role="system", content="You are a helpful assistant")

    @pytest.fixture
    def user_template(self):
        return StringMessageTemplate(
            role="user",
            template="Hi, my name is $name and I'm $age years old.",
            template_vars_model=self.TemplateVarsModel,
        )

    @pytest.fixture
    def user_vars(self):
        return {"name": "Bob", "age": 42}

    @pytest.fixture
    def user_message(self):
        return Message(role="user", content="Hi, my name is Bob and I'm 42 years old.")

    def test_init(self, system_template, user_template):
        prompt = Prompt(
            name="Test Prompt",
            description="A test prompt",
            system_template=system_template,
            user_template=user_template,
        )
        assert prompt.name == "test_prompt"
        assert prompt.description == "A test prompt"
        assert prompt.system_template == system_template
        assert prompt.user_template == user_template

        assert str(prompt) == "Prompt(name=test_prompt)"

    def test_render_with_user_template(
        self,
        system_template,
        user_template,
        user_vars,
        system_message,
        user_message,
    ):
        prompt = Prompt(
            name="Test Prompt",
            description="A test prompt",
            system_template=system_template,
            user_template=user_template,
        )
        conversation = prompt.render(user_vars=user_vars)
        assert isinstance(conversation, Conversation)
        assert len(conversation.messages) == 2
        assert conversation.messages[0] == system_message
        assert conversation.messages[1] == user_message

    def test_render_without_user_template(
        self,
        system_template,
        system_message,
    ):
        prompt = Prompt(
            name="Test Prompt",
            description="A test prompt",
            system_template=system_template,
        )
        conversation = prompt.render()
        assert isinstance(conversation, Conversation)
        assert len(conversation.messages) == 1
        assert conversation.messages[0] == system_message

    def test_signature_static_system_no_user(self):
        prompt = Prompt(
            name="Test Prompt",
            description="A test prompt",
            system_template=StaticMessageTemplate(role="system", template="You are a helpful assistant."),
        )
        signature = prompt.signature
        assert issubclass(signature, BaseModel)

        schema = signature.model_json_schema()
        assert {"type", "title", "description"}.issubset(set(schema))
        assert schema["title"] == prompt.name
        assert schema["description"] in prompt.description

    def test_signature_template_system_no_user(self):
        with pytest.raises(ValueError):
            # no system_template.template_vars_model
            Prompt(
                name="Test Prompt",
                description="A test prompt",
                system_template=StringMessageTemplate(
                    role="system",
                    template="You are a helpful assistant who specializes in $topic.",
                ),
            )

        prompt = Prompt(
            name="Test Prompt",
            description="A test prompt",
            system_template=StringMessageTemplate(
                role="system",
                template="You are a helpful assistant who specializes in $topic.",
                template_vars_model=create_model(
                    "system_vars",
                    topic=(str, ...),  # template var 'topic' must be str
                ),
            ),
        )

        signature = prompt.signature
        assert issubclass(signature, BaseModel)

        schema = signature.model_json_schema()
        assert {"type", "title", "description", "required"}.issubset(set(schema))
        assert schema["title"] == prompt.name
        assert schema["description"] in prompt.description
        assert "system_vars" in schema["properties"]
        assert "system_vars" in schema["required"]

    def test_signature_static_system_static_user(self):
        prompt = Prompt(
            name="Test Prompt",
            description="A test prompt",
            system_template=StaticMessageTemplate(role="system", template="You are a helpful assistant."),
            user_template=StaticMessageTemplate(role="user", template="Hi, who are you?"),
        )
        signature = prompt.signature
        assert issubclass(signature, BaseModel)

        schema = signature.model_json_schema()
        assert {"type", "title", "description"}.issubset(set(schema))
        assert schema["title"] == prompt.name
        assert schema["description"] in prompt.description

    def test_signature_static_system_passthrough_user(self):
        prompt = Prompt(
            name="Test Prompt",
            description="A test prompt",
            system_template=StaticMessageTemplate(role="system", template="You are a helpful assistant."),
            user_template=PassthroughMessageTemplate(),
        )
        signature = prompt.signature
        assert issubclass(signature, BaseModel)

        schema = signature.model_json_schema()
        assert {"type", "title", "description", "required"}.issubset(set(schema))
        assert schema["title"] == prompt.name
        assert schema["description"] in prompt.description

        assert (len(schema["properties"]) == 1) and ("user_vars" in schema["properties"])
        assert (len(schema["required"]) == 1) and ("user_vars" in schema["required"])
        assert (len(schema["$defs"]["PassthroughModel"]["properties"]) == 1) and (
            "content" in schema["$defs"]["PassthroughModel"]["properties"]
        )

    def test_signature_jinja_system_passthrough_user(self):
        class Source(BaseModel):
            """Source content and citation information."""

            title: str
            author: str
            content: str

        class SynthesizerVarsModel(BaseModel):
            """Draft an essay from provided sources."""  # the docstrings should describe what the template does!

            sources: list[Source]

        prompt = Prompt(
            name="Synthesizer",
            description="Synthesize a coherent story from multiple sources",
            system_template=JinjaMessageTemplate(
                role="system",
                template=JinjaTemplate(
                    textwrap.dedent(
                        """
                    Act as a ghostwriter. Your task is to review the attached sources and draft an informative and engaging memo that summarizes the key insights and takeaways. Follow these instructions carefully:

                    The user will provide the content and additional guidance for topics of interest for this newsletter.

                    Follow these steps to complete the task:

                    1. Synthesize key takeaways
                    2. Provide citations to the sources where appropriate.
                    3. Use a standard 5-paragraph essay format

                    Here is the content to be summarized and included in the newsletter:

                    <sources>
                    {% for source in sources %}
                      <source>
                      {{source}}
                      </source>
                    {% endfor %}
                    </sources>
                    """.strip()
                    ),
                ),
                template_vars_model=SynthesizerVarsModel,
            ),
            user_template=PassthroughMessageTemplate(),
        )

        signature = prompt.signature
        assert issubclass(signature, BaseModel)

        schema = signature.model_json_schema()
        assert {"type", "title", "description", "required"}.issubset(set(schema))
        assert schema["title"] == prompt.name
        assert schema["description"] in prompt.description

        assert set(schema["properties"]) == {"system_vars", "user_vars"}
        assert set(schema["required"]) == {"system_vars", "user_vars"}

        assert (len(schema["$defs"]["SynthesizerVarsModel"]["properties"]) == 1) and (
            "sources" in schema["$defs"]["SynthesizerVarsModel"]["properties"]
        )
        assert (len(schema["$defs"]["PassthroughModel"]["properties"]) == 1) and (
            "content" in schema["$defs"]["PassthroughModel"]["properties"]
        )

    def test_invalid_system_template(self):
        with pytest.raises(ValueError):
            Prompt(
                name="Test",
                description="Test",
                system_template=StringMessageTemplate(
                    role="system",
                    template="Hello $name",
                    # Missing template_vars_model
                ),
            )

    def test_invalid_user_template(self):
        with pytest.raises(ValueError):
            Prompt(
                name="Test",
                description="Test",
                system_template=StaticMessageTemplate(role="system", template="Static system"),
                user_template=StringMessageTemplate(
                    role="user",
                    template="Hello $name",
                    # Missing template_vars_model
                ),
            )

    def test_schema_with_complex_types(self):
        class SourceData(BaseModel):
            url: str
            content: str
            metadata: dict[str, str]

        class SystemVars(BaseModel):
            sources: list[SourceData]
            max_length: int

        class UserVars(BaseModel):
            query: str
            filters: list[str]

        prompt = Prompt(
            name="Complex Prompt",
            description="Test complex schema generation",
            system_template=JinjaMessageTemplate(
                role="system", template="Process {{sources|length}} sources", template_vars_model=SystemVars
            ),
            user_template=JinjaMessageTemplate(
                role="user", template="Query: {{query}}\nFilters: {{filters|join(', ')}}", template_vars_model=UserVars
            ),
        )

        schema = prompt.schema
        assert "SystemVars" in schema["$defs"]
        assert "UserVars" in schema["$defs"]
        assert "SourceData" in schema["$defs"]
        assert set(schema["required"]) == {"system_vars", "user_vars"}

    def test_render_with_model_input(self):
        class SystemVars(BaseModel):
            mode: str

        class UserVars(BaseModel):
            query: str

        prompt = Prompt(
            name="Test",
            description="Test",
            system_template=JinjaMessageTemplate(
                role="system", template="Mode: {{mode}}", template_vars_model=SystemVars
            ),
            user_template=JinjaMessageTemplate(role="user", template="Query: {{query}}", template_vars_model=UserVars),
        )

        conv = prompt.render(system_vars=SystemVars(mode="test"), user_vars=UserVars(query="hello"))
        assert len(conv.messages) == 2
        assert conv.messages[0].content == "Mode: test"
        assert conv.messages[1].content == "Query: hello"
