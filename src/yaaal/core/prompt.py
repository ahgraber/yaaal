"""Components for composable prompting.

The base unit is a Message(role, context), which has generally been accepted by all LLM chat APIs.

A list of Messages is a Conversation, which provides easy conversion to a messages array for API calls.

Sometimes we may want to predefine the messages in the conversation via MessageTemplates.
A MessageTemplate defines the role, the template, and the rendering method to generate a Message.
It may also add variable validation with pydantic through the template_vars_model attribute.

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
    """Base class for rendering a Message.

    Defines interface for converting templates and variables into Messages.
    Subclasses implement specific templating engines.

    Attributes
    ----------
        role: Role used in rendered Message (system/user/assistant)
        template: Template defining message content structure
        template_vars_model: Optional Pydantic model for variable validation
    """

    role: Role
    """Role used in Message."""

    template: Any
    """Template that defines the message content for each render."""

    template_vars_model: Type[BaseModel] | None
    """Template that defines the message content for each render."""

    @abstractmethod
    def render_message(self, template_vars: dict[str, Any] | BaseModel) -> Message:
        """Render the message."""
        raise NotImplementedError


class StaticMessageTemplate(MessageTemplate):
    """Render static messages.

    Used for constant system messages or fixed responses.
    Ignores any template variables passed during rendering.

    Args:
        role: Message role (system/user/assistant)
        template: Static message content

    Examples
    --------
        >>> template = StaticMessageTemplate(role="system", template="You are a helpful assistant.")
        >>> msg = template.render_message()  # Variables optional
        >>> assert msg.content == "You are a helpful assistant."
    """

    def __init__(self, role: Role, template: str):
        self.role = role
        self.template = template
        self.template_vars_model = None

    @override
    def render_message(self, template_vars: dict[str, Any] | None = None) -> Message:
        """Render the message."""
        return Message(role=self.role, content=self.template)


class StringMessageTemplate(MessageTemplate):
    """Render with Python's string.Template.

    Uses template strings with $-based substitution:
    - Variables denoted by $varname or ${varname}
    - Optional Pydantic validation of variables
    - Strict variable substitution (raises KeyError for missing vars)

    Args:
        role: Message role (system/user/assistant)
        template: Template string using $-based substitution
        template_vars_model: Optional Pydantic model for validation

    Examples
    --------
        >>> class UserVars(BaseModel):
        ...     name: str
        ...     age: int
        >>> template = StringMessageTemplate(
        ...     role="user", template="Name: $name, Age: $age", template_vars_model=UserVars
        ... )
        >>> msg = template.render_message({"name": "Bob", "age": 42})
    """

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

    Simplifies creating prompts that need raw user input by:
    - Enforcing 'user' role
    - Creating model with 'content' field
    - Using simple string substitution

    Args:
        role: Always 'user' (enforced)
        template: Always '$content' (enforced)
        template_vars_model: Ignored, auto-generated

    Examples
    --------
        >>> template = PassthroughMessageTemplate()
        >>> msg = template.render_message({"content": "Hello!"})
        >>> assert msg.content == "Hello!"
        >>> assert msg.role == "user"
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
    """Jinja2 template renderer.

    Provides Jinja2 templating with optional Pydantic validation.
    Uses StrictUndefined to catch missing variables.

    Args:
        role (Role): Message role (system/user/assistant)
        template (str | JinjaTemplate): Template string or Jinja template
        template_vars_model (Type[BaseModel] | None): Pydantic model for variables

    Examples
    --------
        >>> class UserVars(BaseModel):
        ...     name: str
        ...     age: int
        >>> template = JinjaMessageTemplate(
        ...     role="user",
        ...     template="Hi, I'm {{name}}, {{age}} years old",
        ...     template_vars_model=UserVars,
        ... )
        >>> msg = template.render_message({"name": "Bob", "age": 42})
    """

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
        """Render the message."""
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

    Manages system and user message templates for LLM interactions.
    Provides parameter validation and JSON schema generation.

    Attributes
    ----------
        name (str): Prompt name; Used as function name when defining signature
        description (str): Human readable description; used as function docstring
        system_template (MessageTemplate): Required system message template
        user_template (MessageTemplate | None): Optional user message template
        signature (Type[BaseModel]): Pydantic model for parameters
        schema (JSON): OpenAPI-compatible schema
    """

    def __init__(
        self,
        name: str,
        description: str,
        system_template: MessageTemplate,
        user_template: MessageTemplate | None = None,
    ):
        self.name = to_snake_case(name)
        self.description = description
        self.system_template = system_template
        self.user_template = user_template

        self.signature = self._get_signature()
        self.schema = self.signature.model_json_schema()

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"

    def render(
        self,
        *,
        system_vars: dict[str, Any] | BaseModel | None = None,
        user_vars: dict[str, Any] | BaseModel | None = None,
    ) -> Conversation:
        """Render the conversation / messages array.

        Args:
            system_vars: Variables for system template
            user_vars: Variables for user template

        Returns
        -------
            Conversation with rendered messages

        Raises
        ------
            ValidationError: If variables fail validation
        """
        messages = [
            # always require a system message
            self.system_template.render_message(system_vars or {}),
        ]
        if self.user_template:
            messages.append(self.user_template.render_message(user_vars or {}))

        return Conversation(messages=messages)

    def _get_signature(self) -> Type[BaseModel]:
        """Define function signature as pydantic model."""
        field_definitions = {}

        if isinstance(self.system_template, StaticMessageTemplate):
            pass
        else:
            if self.system_template.template_vars_model:
                field_definitions["system_vars"] = (
                    self.system_template.template_vars_model,
                    ...,
                )  # .model_json_schema() | {"title": None}
            else:
                raise ValueError("No system template model provided; cannot define signature.")

        if self.user_template:
            if isinstance(self.user_template, StaticMessageTemplate):
                pass
            else:
                if self.user_template.template_vars_model:
                    field_definitions["user_vars"] = (
                        self.user_template.template_vars_model,
                        ...,
                    )
                else:
                    raise ValueError("User template exists without user template model; cannot define signature.")

        model = create_model(self.name, __doc__=self.description, **field_definitions)

        return model
