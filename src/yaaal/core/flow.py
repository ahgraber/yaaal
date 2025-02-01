"""Components for composable LLM calls.

Flows process a predefined series of steps (Caller | CallableWithSignature) before returning a final response.

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
from typing import Any, Match, Pattern, Type

import json_repair
from pydantic import BaseModel, Field

from .caller import BaseCaller
from .prompt import Prompt
from .tools import CallableWithSignature, respond_as_tool
from ..types.base import JSON
from ..types.core import Conversation, Message, ToolMessage

logger = logging.getLogger(__file__)
