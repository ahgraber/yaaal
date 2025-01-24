"""Components for composable prompting.

The base unit is a Message(role, context), which has generally been accepted by all LLM chat APIs.

A list of Messages is a Conversation, which provides easy conversion to a messages array for API calls.

Sometimes we may want to predefine the messages in the conversation via MessageTemplates.
A MessageTemplate defines the role, the template, and the rendering method to generate a Message.
It may also add variable validation with Pydantic through the template_vars_model attribute.

StaticMessageTemplate provides a prompt template that is not templated, that is,
there are no template variables and it renders exactly the same string every time.

StringMessageTemplate uses str.format() to render a templated string based on a dict provided at render-time.

JinjaMessageTemplate uses a jinja2 Template to render a templated string based on a dict provided at render-time.

A Prompt is a way to use various MessageTemplates to render a Conversation.
We may want to treat Prompts as Functions or Tools for the tool-calling API;
Prompts provide a 'signature' method to mock a function signature that details all of the template variables necessary.
"""

from abc import abstractmethod
import logging
from string import Template as StringTemplate
from typing import Any, Literal, Type

from jinja2 import StrictUndefined, Template as JinjaTemplate
from pydantic import BaseModel, create_model
from typing_extensions import override  # TODO: import from typing when drop support for 3.11

from ..types.base import JSON
from ..types.core import Conversation, Message, Role
from ..utilities import to_snake_case

logger = logging.getLogger(__name__)


class MessageTemplate:
    """Base class for rendering a Message."""

    _role: Role
    _template: str
    _template_vars_model: Type[BaseModel] | None

    @property
    def role(self) -> Role:
        """Role used in Message."""
        return self._role

    @role.setter
    def role(self, role: Role):
        self._role = role

    @property
    def template(self) -> Any:
        """Template that defines the message content for each render."""
        return self._template

    @template.setter
    def template(self, template: Any):
        self._template = template

    @property
    def template_vars_model(self) -> Type[BaseModel] | None:
        """Template that defines the message content for each render."""
        return self._template_vars_model

    @template_vars_model.setter
    def template_vars_model(self, template_vars_model: Type[BaseModel] | None):
        self._template_vars_model = template_vars_model

    @abstractmethod
    def render_message(self, template_vars: dict[str, Any] | BaseModel) -> Message:
        """Render the message."""
        raise NotImplementedError


class StaticMessageTemplate(MessageTemplate):
    """Render static messages."""

    def __init__(self, role: Role, template: str):
        self.role = role
        self.template = template
        self.template_vars_model = None

    @override
    def render_message(self, template_vars: dict[str, Any] | None = None) -> Message:
        """Render the message."""
        return Message(role=self.role, content=self.template)


class StringMessageTemplate(MessageTemplate):
    """Render with template strings.

    Uses template strings, not format strings (variables denoted by '$varname', not '{varname}').

    Roughly equivalent to:
    >>> template = "Hi, my name is $name and I'm $age years old."
    >>> template_vars = {"name": "Bob", "age": 42}
    >>> return template.substitute(template_vars)
    """

    _template: StringTemplate

    def __init__(
        self,
        role: Role,
        template: str,
        template_vars_model: Type[BaseModel] | None = None,
    ):
        self.role = role
        self.template = template
        self.template_vars_model = template_vars_model

    @property
    def template(self) -> StringTemplate:
        """Jinja Template that defines the message content for each render."""
        return self._template

    @template.setter
    def template(self, template: str | StringTemplate):
        self._template = template if isinstance(template, StringTemplate) else StringTemplate(template)

    # NOTE: it would be cool to autogenerate a Pydanic model for template variables, but I don't think the complexity is worth it

    @override
    def render_message(self, template_vars: dict[str, Any]) -> Message:
        """Render the message."""
        vars_ = template_vars if isinstance(template_vars, dict) else template_vars.model_dump()

        # If no template_vars_model, render w/o validation
        if self.template_vars_model is None:
            logger.warning("Rendering template without variable validation!")
            return Message(role=self.role, content=self.template.substitute(vars_))

        else:
            validated_vars = self.template_vars_model(**vars_).model_dump()
            return Message(role=self.role, content=self.template.substitute(validated_vars))


class PassthroughMessageTemplate(StringMessageTemplate):
    """Render message by passing content through.

    Useful for defining Prompts where the user input is needed.
    Always has a 'user' role. Automatically creates the appropriate 'template_vars_model'.

    Pass template_vars as follows:
    >>> template = "$content"
    >>> template_vars = {"content": "Hi, my name is Bob and I'm 42 years old.}
    >>> return template.substitute(template_vars)
    """

    def __init__(
        self,
        role: Literal["user"] = "user",
        template: str = "$content",
        template_vars_model: Type[BaseModel] | None = None,
    ):
        super().__init__(
            role="user",
            template="$content",
            template_vars_model=create_model("PassthroughModel", content=(str, ...)),
        )


class JinjaMessageTemplate(MessageTemplate):
    """Render with jinja2 templates.

    Roughly equivalent to:
    >>> template = JinjaTemplate("Hi, my name is {{name}} and I'm {{age}} years old.")
    >>> template_vars = {"name": "Bob", "age": 42}
    >>> return template.render(**template_vars)
    """

    _template: JinjaTemplate

    def __init__(
        self,
        role: Role,
        template: str | JinjaTemplate,
        template_vars_model: Type[BaseModel] | None = None,
    ):
        self.role = role
        self.template = template
        self.template_vars_model = template_vars_model

    @property
    def template(self) -> JinjaTemplate:
        """Jinja Template that defines the message content for each render."""
        return self._template

    @template.setter
    def template(self, template: str | JinjaTemplate):
        if isinstance(template, JinjaTemplate):
            # ensure we raise an exception when a variable is present in the template but it is not passed
            template.environment.undefined = StrictUndefined
            self._template = template
        else:
            self._template = JinjaTemplate(
                template,
                undefined=StrictUndefined,
            )

    @override
    def render_message(self, template_vars: dict[str, Any] | BaseModel) -> Message:
        """Render the system message."""
        vars_ = template_vars if isinstance(template_vars, dict) else template_vars.model_dump()

        # If no template_vars_model, render w/o validation
        if self.template_vars_model is None:
            logger.warning("Rendering template without variable validation!")
            return Message(role=self.role, content=self.template.render(**vars_))

        else:
            validated_vars = self.template_vars_model(**vars_).model_dump()
            return Message(role=self.role, content=self.template.render(**validated_vars))


class Prompt:
    """Prompt objects define how to render messages.

    A system message/template is required; user message/templates are optional.
    """

    def __init__(
        self,
        name: str,
        description: str,
        system_template: MessageTemplate,
        user_template: MessageTemplate | None = None,
    ):
        self.name = name
        self.description = description
        self.system_template = system_template
        self.user_template = user_template

    @property
    def name(self) -> str:
        """Prompt name.

        Used as function name when defining signature to treat prompt as a tool.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = to_snake_case(name)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"

    def render(
        self,
        *,
        system_vars: dict[str, Any] | BaseModel | None = None,
        user_vars: dict[str, Any] | BaseModel | None = None,
    ) -> Conversation:
        """Render the conversation / messages array."""
        messages = [
            # always require a system message
            self.system_template.render_message(system_vars or {}),
        ]
        if self.user_template:
            messages.append(self.user_template.render_message(user_vars or {}))

        return Conversation(messages=messages)

    def signature(self) -> Type[BaseModel]:
        """Provide function signature as json schema."""
        field_definitions = {}

        if not isinstance(self.system_template, StaticMessageTemplate):
            if not self.system_template.template_vars_model:
                raise ValueError("No system template model provided; cannot define signature.")
            field_definitions["system_vars"] = (
                self.system_template.template_vars_model,
                ...,
            )  # .model_json_schema() | {"title": None}

        if not isinstance(self.user_template, StaticMessageTemplate):
            if self.user_template and self.user_template.template_vars_model:
                field_definitions["user_vars"] = (
                    self.user_template.template_vars_model,
                    ...,
                )  # .model_json_schema() | {"title": None}
            if self.user_template and not self.user_template.template_vars_model:
                raise ValueError("User template exists without user template model; cannot define signature.")

        model = create_model(self.name, __doc__=self.description, **field_definitions)

        return model
