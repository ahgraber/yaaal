"""Core components for composable LLM interactions.

This module provides the foundational components for building templated conversations,
validating responses, and managing the execution of LLM calls with tools.
"""

from .base import CallableWithSignature, Handler, Validator
from .caller import (
    Caller,
    create_chat_caller,
    create_structured_caller,
    create_tool_caller,
)
from .exceptions import ResponseError, ValidationError
from .handler import ResponseHandler, ToolHandler
from .tool import Tool, tool
from .validator import (
    PassthroughValidator,
    PydanticValidator,
    RegexValidator,
    ToolValidator,
)

__all__ = [
    # Base protocols
    "CallableWithSignature",
    "Handler",
    "Validator",
    # Validators
    "PassthroughValidator",
    "PydanticValidator",
    "RegexValidator",
    "ToolValidator",
    # Handlers
    "ResponseHandler",
    "ToolHandler",
    # Callers and Tools
    "Caller",
    "create_chat_caller",
    "create_structured_caller",
    "create_tool_caller",
    "Tool",
    "tool",
    # Exceptions
    "ResponseError",
    "ValidationError",
    # Validators
    "PassthroughValidator",
    "PydanticValidator",
    "RegexValidator",
    "ToolValidator",
    # Handlers
    "ResponseHandler",
    "ToolHandler",
    # Callers and Tools
    "Caller",
    "create_chat_caller",
    "create_structured_caller",
    "create_tool_caller",
    "Tool",
    "tool",
    # Exceptions
    "ResponseError",
    "ValidationError",
]
