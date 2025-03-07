"""Components for composable prompting.

The base unit is a Message(role, context), which has generally been accepted by all LLM chat APIs.

A list of Messages is a Conversation, which provides easy conversion to a messages array for API calls.

Sometimes we may want to predefine the messages in the conversation via MessageTemplates.
A MessageTemplate defines the role, the template, and the rendering method to generate a Message.
It may also add variable validation with pydantic through the validation_model attribute.

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
from typing import Any, Literal, Mapping, Protocol, Sequence, Type, TypeVar, Union, cast

from jinja2 import StrictUndefined, Template as JinjaTemplate
from pydantic import BaseModel, ConfigDict, Field, create_model
from typing_extensions import override, runtime_checkable  # TODO: import from typing when drop support for 3.11

from ..types.base import JSON
from ..types.core import Conversation, Message, Role
from ..types.utils import merge_models
from ..utilities import to_snake_case

logger = logging.getLogger(__name__)


@runtime_checkable
class MessageTemplate(Protocol):
    """Base class for rendering a Message.

    Defines interface for converting templates and variables into Messages.
    Subclasses implement specific templating engines.

    Attributes
    ----------
        role: The role (system/user/assistant) for rendered messages
        template: The template definition that structures message content
        validation_model: Optional Pydantic model for validating variables
    """

    role: Role
    template: Any
    validation_model: Type[BaseModel] | None

    def render(self, template_vars: dict[str, Any] | BaseModel) -> Message:
        """Render the message."""
        raise NotImplementedError


class StaticMessageTemplate(MessageTemplate):
    """Render static messages.

    Used for constant system messages or fixed responses.
    Ignores any template variables passed during rendering.

    Args:
        role: The role for rendered messages
        template: The fixed message content to use
        validation_model: Not used, always None

    Examples
    --------
        >>> template = StaticMessageTemplate(role="system", template="You are a helpful assistant.")
        >>> msg = template.render()  # Variables optional
        >>> assert msg.content == "You are a helpful assistant."
    """

    def __init__(
        self,
        role: Role,
        template: str,
        validation_model: None = None,
    ):
        self.role = role
        self.template = template
        self.validation_model = None

    @override
    def render(self, template_vars: None = None) -> Message:
        """Render the message."""
        return Message(role=self.role, content=self.template)


class StringMessageTemplate(MessageTemplate):
    """Render with Python's string.Template.

    Uses template strings with $-based substitution:
    - Variables denoted by $varname or ${varname}
    - Optional Pydantic validation of variables
    - Strict variable substitution (raises KeyError for missing vars)

    Args:
        role: The role for rendered messages
        template: Template string using $-based substitution
        validation_model: Optional Pydantic model for variable validation

    Examples
    --------
        >>> class UserVars(BaseModel):
        ...     name: str
        ...     age: int
        >>> template = StringMessageTemplate(role="user", template="Name: $name, Age: $age", validation_model=UserVars)
        >>> msg = template.render({"name": "Bob", "age": 42})
    """

    def __init__(
        self,
        role: Role,
        template: str,
        validation_model: Type[BaseModel] | None = None,
    ):
        self.role = role
        self.template = template
        self.validation_model = validation_model

    @property
    def template(self) -> StringTemplate:
        """Jinja Template that defines the message content for each render."""
        return self._template

    @template.setter
    def template(self, template: str | StringTemplate):
        self._template = template if isinstance(template, StringTemplate) else StringTemplate(template)

    # NOTE: it would be cool to autogenerate a Pydanic model for template variables, but I don't think the complexity is worth it
    @property
    def validation_model(self) -> Type[BaseModel] | None:
        """Pydantic model for validating template variables."""
        return self._validation_model

    @validation_model.setter
    def validation_model(self, validation_model: Type[BaseModel] | None):
        if validation_model is not None and issubclass(validation_model, BaseModel):
            try:
                if validation_model.model_config.get("extra", "ignore") != "ignore":
                    logger.warning(
                        "validation_model should set extra='ignore' config or unhandled edge cases may occur."
                    )
            except KeyError:
                pass

            self._validation_model = validation_model

        else:
            self._validation_model = None

    @override
    def render(self, template_vars: dict[str, Any] | BaseModel) -> Message:
        """Render the message."""
        vars_ = template_vars if isinstance(template_vars, dict) else template_vars.model_dump()

        # If no validation_model, render w/o validation
        if self.validation_model is None:
            logger.warning("Rendering template without variable validation!")
            return Message(role=self.role, content=self.template.substitute(vars_))

        else:
            validated_vars = self.validation_model(**vars_).model_dump()
            return Message(role=self.role, content=self.template.substitute(validated_vars))


class PassthroughMessageTemplate(StringMessageTemplate):
    """Render message by passing content through.

    Simplifies creating prompts that need raw user input by:
    - Enforcing 'user' role
    - Creating model with 'content' field
    - Using simple string substitution

    Args:
        role (Role): Always 'user' (enforced)
        template (str): Always '$user' (enforced)
        validation_model (Type[BaseModel]): Ignored, auto-generated

    Examples
    --------
        >>> template = PassthroughMessageTemplate()
        >>> msg = template.render({"content": "Hello!"})
        >>> assert msg.content == "Hello!"
        >>> assert msg.role == "user"
    """

    def __init__(
        self,
        role: Literal["user"] = "user",
        template: str = "$user",
        validation_model: Type[BaseModel] | None = None,
    ):
        self.role = "user"
        self.template = "$user"
        self._validation_model = create_model("UserMessage", user=(str, ...))


class JinjaMessageTemplate(MessageTemplate):
    """Jinja2 template renderer.

    Provides Jinja2 templating with optional Pydantic validation.
    Uses StrictUndefined to catch missing variables.

    Args:
        role: The role for rendered messages
        template: Jinja template string or Template instance
        validation_model: Optional Pydantic model for variable validation

    Examples
    --------
        >>> class UserVars(BaseModel):
        ...     name: str
        ...     age: int
        >>> template = JinjaMessageTemplate(
        ...     role="user",
        ...     template="Hi, I'm {{name}}, {{age}} years old",
        ...     validation_model=UserVars,
        ... )
        >>> msg = template.render({"name": "Bob", "age": 42})
    """

    def __init__(
        self,
        role: Role,
        template: str | JinjaTemplate,
        validation_model: Type[BaseModel] | None = None,
    ):
        self.role = role
        self.template = template
        self.validation_model = validation_model

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

    @property
    def validation_model(self) -> Type[BaseModel] | None:
        """Pydantic model for validating template variables."""
        return self._validation_model

    @validation_model.setter
    def validation_model(self, validation_model: Type[BaseModel] | None):
        if validation_model is not None and issubclass(validation_model, BaseModel):
            try:
                if validation_model.model_config.get("extra", "ignore") != "ignore":  # default is ignore
                    logger.warning(
                        "validation_model should set extra='ignore' config or unhandled edge cases may occur."
                    )
            except KeyError:
                pass

            self._validation_model = validation_model

        else:
            self._validation_model = None

    @override
    def render(self, template_vars: dict[str, Any] | BaseModel) -> Message:
        """Render the message."""
        vars_ = template_vars if isinstance(template_vars, dict) else template_vars.model_dump()

        # If no validation_model, render w/o validation
        if self.validation_model is None:
            logger.warning("Rendering template without variable validation!")
            return Message(role=self.role, content=self.template.render(**vars_))

        else:
            validated_vars = self.validation_model(**vars_).model_dump()
            return Message(role=self.role, content=self.template.render(**validated_vars))


class ConversationTemplate:
    """Define conversation structure with multiple message templates.

    Manages an ordered sequence of templates/messages with:
    - Combined parameter validation across all templates
    - OpenAPI schema generation for tool calling
    - Consistent rendering of complete conversations

    Args:
        name: Template name, used as function name in schemas
        description: Human readable description for documentation
        conversation_spec: Sequence of templates/messages defining the conversation

    Attributes
    ----------
        signature: Generated Pydantic model for all parameters
        schema: OpenAPI-compatible JSON schema
    """

    def __init__(
        self,
        name: str,
        description: str,
        conversation_spec: Sequence[MessageTemplate | Message],
    ):
        self.name = name
        self.description = description
        self.conversation_spec = conversation_spec

        self._signature = self._define_signature()
        self._schema = cast(JSON, self.signature.model_json_schema())

    @property
    def name(self) -> str:
        """Get the template / function name."""
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """Set the template / function name."""
        _name = to_snake_case(name)
        if _name != name:
            logger.warning(f"Converted template name '{name}' to '{_name}'")
        self._name = _name

    @property
    def description(self) -> str:
        """Get the template description / function docstring."""
        return self._description

    @description.setter
    def description(self, description: str) -> None:
        """Set the template description / function docstring."""
        if not description or not isinstance(description, str):
            raise ValueError("Description must be provided as string for use as docstring")
        self._description = description

    @property
    def conversation_spec(self) -> Sequence[MessageTemplate | Message]:
        """Get the conversation template."""
        return self._conversation_spec

    @conversation_spec.setter
    def conversation_spec(self, conversation_spec: Sequence[MessageTemplate | Message]) -> None:
        """Validate and store templates.

        Parameters
        ----------
        conversation_spec : Sequence[MessageTemplate | Message]
            List of MessageTemplates to be validated.

        Raises
        ------
        ValueError
            If the conversation list is empty.
        """
        if not conversation_spec:
            raise ValueError("Conversation list cannot be empty")

        if not all(isinstance(message, (MessageTemplate, Message)) for message in conversation_spec):
            raise TypeError("Conversation list must contain only MessageTemplates or Messages")

        if not any(message.role == "system" for message in conversation_spec):
            raise ValueError("Template list must contain at least one system message")

        self._conversation_spec = conversation_spec

    @property
    def signature(self) -> Type[BaseModel]:
        """Get the Pydantic model for the template signature."""
        return self._signature

    @property
    def schema(self) -> JSON:
        """Get the OpenAPI-compatible schema."""
        return self._schema

    def _define_signature(self) -> Type[BaseModel]:
        """Define the render() function signature.

        Returns a Pydantic model merged from the validation models of each template.

        NOTE: This does not include merged validators and is used solely to generate the function signature;
        validation is handled by the MessageTemplate.validation_model

        Returns
        -------
        Type[BaseModel]
            A Pydantic model representing the render() function signature.

        Raises
        ------
        ValueError
            If any non-static template lacks a validation model.
        """
        message_validators = []
        for template in self.conversation_spec:
            if not isinstance(template, MessageTemplate):
                continue
            if isinstance(template, StaticMessageTemplate):
                continue
            if not template.validation_model:
                raise ValueError("All non-static templates require a validation model")

            message_validators.append(template.validation_model)

        if len(message_validators) == 0:
            return create_model(self.name, __doc__=self.description)

        return merge_models(*message_validators, name=self.name, description=self.description)

    def render(self, template_vars: dict[str, Any]) -> Conversation:
        """
        Render a complete Conversation using the provided variables.

        Each message template in the conversation specification is validated and rendered
        using the variables from 'template_vars'. The resulting messages are assembled in order
        into a Conversation object.

        Parameters
        ----------
        template_vars : dict[str, Any]
            A dictionary of variables used for rendering the message templates.

        Returns
        -------
        Conversation
            An instance containing all rendered messages in the defined order.

        Raises
        ------
        ValueError
            If a template cannot render a message due to missing or invalid variables.
        """
        messages = []
        for msg in self.conversation_spec:
            if isinstance(msg, MessageTemplate):
                messages.append(msg.render(template_vars))
            elif isinstance(msg, Message):
                messages.append(msg)
            else:
                raise TypeError(f"Invalid message type: {msg}")

        return Conversation(messages=messages)
