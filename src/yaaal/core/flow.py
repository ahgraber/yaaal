"""Components for composable LLM calls.

Flows process a predefined series of steps (Caller | CallableWithSchema) before returning a final response.

A Flow must:
- manage the in-Flow conversation
- manage the interaction between the conversation history and the Flow response
  Open question: Should the response include the entire in-Flow conversation or just the response?
- manage any context outside the conversation
- ensure all function calls are auto-invoked
- be callable

"""

import json
import logging
from typing import Match, Pattern, Type

import json_repair
from pydantic import BaseModel, Field

from .base import CallableWithSchema
from .exceptions import ValidationError
from .handler import ResponseHandler, ToolHandler  # , CompositeHandler
from .template import (
    ConversationTemplate,
    JinjaMessageTemplate,
    StaticMessageTemplate,
    StringMessageTemplate,
    UserMessageTemplate,
)
from .tool import Tool, anthropic_pydantic_function_tool
from .validator import PassthroughValidator, PydanticValidator, RegexValidator, ToolValidator
from ..types_.base import JSON
from ..types_.core import Conversation
from ..types_.openai_compat import ChatCompletion, convert_response

logger = logging.getLogger(__file__)
