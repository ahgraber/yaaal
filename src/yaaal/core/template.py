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
from typing import Any, List, Literal, Mapping, Protocol, Sequence, Type, TypeVar, Union, cast

from jinja2 import StrictUndefined, Template as JinjaTemplate
from pydantic import BaseModel, ConfigDict, Field, create_model
from typing_extensions import override, runtime_checkable  # TODO: import from typing when drop support for 3.11

from ..types.base import JSON
from ..types.core import Conversation, ConversationSpec, Message, MessageSpec, Role
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

    Manages an arbitrary number of message templates and static messages.
    Provides parameter validation and JSON schema generation.

    Attributes
    ----------
    name : str
        Template name; Used as function name when defining signature
    description : str
        Human readable description; used as function docstring
    templates : dict[str, MessageTemplate]
        Map of template names to templates
    signature : Type[BaseModel]
        Pydantic model for parameters
    schema : JSON
        OpenAPI-compatible schema
    """

    def __init__(
        self,
        name: str,
        description: str,
        templates: list[MessageTemplate],
    ):
        self.name = to_snake_case(name)
        self.description = description
        self.templates = templates

        self.signature = self._get_signature()
        self.schema = self.signature.model_json_schema()

    @property
    def templates(self) -> Mapping[str, MessageTemplate]:
        """Get the dictionary of validated templates."""
        return self._templates

    @templates.setter
    def templates(self, templates: Sequence[MessageTemplate]) -> None:
        """Validate and store templates.

        Parameters
        ----------
        templates : Sequence[MessageTemplate]
            List of MessageTemplates to be validated and converted into a dictionary.

        Raises
        ------
        ValueError
            If the template list is empty, a template lacks a name or validation model,
            or if duplicate names exist.
        """
        if not templates:
            raise ValueError("Template list cannot be empty")
        template_dict = {}
        for template in templates:
            if not template.name:
                raise ValueError(f"Template {template} requires a name")
            if template.name in template_dict:
                raise ValueError(f"Duplicate template name '{template.name}' found")

            if isinstance(template, StaticMessageTemplate):
                template_dict[template.name] = template
                continue

            if not template.validation_model:
                raise ValueError(f"Template {template.name} requires a validation model")

            template_dict[template.name] = template
        self._templates = template_dict

    def _get_signature(self) -> Type[BaseModel]:
        """Define the render() function signature.

        Returns a Pydantic model with a single field 'conversation_spec' which is a list.
        Each list item is a MessageSpec model dynamically generated per non-static template;
        the MessageSpec model is parameterized by the template's validation model.

        Returns
        -------
        Type[BaseModel]
            A Pydantic model representing the render() function signature.

        Raises
        ------
        ValueError
            If any non-static template lacks a validation model.
        """
        spec_models = []
        for name, template in self.templates.items():
            if isinstance(template, StaticMessageTemplate):
                continue
            if not template.validation_model:
                raise ValueError(f"Template {name} requires a validation model")
            # Each MessageSpec model has a literal 'name' field and a 'vars'
            # field typed with the template's validation model.
            msg_spec_model = create_model(
                f"{name}_MessageSpec",
                name=(Literal[name], ...),
                vars=(template.validation_model, ...),
            )
            spec_models.append(msg_spec_model)

        if not spec_models:
            # Fallback: no non-static templates yields an empty signature.
            return create_model(self.name, __doc__=self.description)

        MessageSpecUnion = Union[tuple(spec_models)]  # NOQA: N806 # type: ignore

        model = create_model(
            self.name,
            __doc__=self.description,
            __config__=ConfigDict(extra="forbid"),
            messages=(List[MessageSpecUnion], ...),
        )
        return model

    def _validate_conversation_spec(
        self,
        conversation_spec: ConversationSpec
        | dict[str, list[dict[str, dict[str, Any] | BaseModel | None]]]
        | list[dict[str, dict[str, Any] | BaseModel | None]],
    ) -> ConversationSpec:
        """Validate (and coerce) a conversation_spec input into a ConversationSpec instance.

        Parameters
        ----------
        conversation_spec : ConversationSpec or list[dict[str, dict[str, Any] | BaseModel]]
            The conversation specification provided by the user.

        Returns
        -------
        ConversationSpec
            A validated ConversationSpec instance containing message instructions.

        Raises
        ------
        ValueError
            If any instruction does not contain exactly one template name and vars pair.
        """
        if isinstance(conversation_spec, ConversationSpec):
            return conversation_spec

        if isinstance(conversation_spec, dict):
            return ConversationSpec(**conversation_spec)

        messages = []
        for item in conversation_spec:
            if len(item) != 1:
                raise ValueError("Each instruction must have exactly one template name and vars pair")
            name, vars_ = next(iter(item.items()))
            messages.append(MessageSpec(name=name, vars=vars_))
        return ConversationSpec(messages=messages)

    def render(
        self, conversation_spec: ConversationSpec | list[dict[str, dict[str, Any] | BaseModel]]
    ) -> Conversation:
        """Render a Conversation by applying each template with its variables.

        The method first validates the conversation specification then renders each message
        using the corresponding template. The messages are returned in the order provided.

        Parameters
        ----------
        conversation_spec : ConversationSpec or list[dict[str, dict[str, Any] | BaseModel]]
            A ConversationSpec instance or a list of dictionaries mapping a template name
            to its variables.

        Returns
        -------
        Conversation
            A Conversation instance containing the rendered messages in instruction order.

        Raises
        ------
        ValueError
            If a referenced template is not found in the template dictionary.
        """
        spec = self._validate_conversation_spec(conversation_spec)
        messages = []
        for msg_spec in spec.messages:
            if msg_spec.name not in self.templates:
                raise ValueError(f"Template '{msg_spec.name}' not found")
            messages.append(self.templates[msg_spec.name].render(msg_spec.vars))
        return Conversation(messages=messages)
