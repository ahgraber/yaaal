"""Components for composable LLM calls.

Agents use LLMs control the workflow -- the AI system handles conditional logic instead of relying on code or the end user.

An Agent may:
- identify applicable tools for the request;
  dynamically create a Caller with a subset of applicable tools
  --> (this might actually be a special type of Caller, do we need a factory method?)
- create a plan to follow;
- determine when to continue or when to revert control to the user

Therefore, an Agent must:
- manage the conversation history
- manage any context outside the conversation
- be callable

"""

import json
import logging
from typing import Match, Pattern, Type

import json_repair
from pydantic import BaseModel, Field

from .caller import Caller
from .template import (
    ConversationTemplate,
    JinjaMessageTemplate,
    StaticMessageTemplate,
    StringMessageTemplate,
    UserMessageTemplate,
)
from .tool import CallableWithSignature
from ..types_.base import JSON
from ..types_.core import Conversation, Message, ToolResultMessage

logger = logging.getLogger(__file__)

# TODO
# tool calling with auto continuation


class Agent:
    input: ...
    conversation: ...
    steps: ...  # steps taken by agent
    outcome: ...  # response
