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
from typing import Any, Match, Pattern, Type

import json_repair
from pydantic import BaseModel, Field

from .caller import BaseCaller
from .prompt import Prompt
from .tools import CallableWithSignature, respond_as_tool
from ..types.base import JSON
from ..types.core import Conversation, Message, ToolMessage

logger = logging.getLogger(__file__)

# TODO
# tool calling with auto continuation


class Agent:
    input: ...
    conversation: ...
    steps: ...  # steps taken by agent
    outcome: ...  # response
