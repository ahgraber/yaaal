"""Core components for composable LLM interactions.

This module provides the foundational components for building templated conversations,
validating responses, and managing the execution of LLM calls with tools.
"""

from .base import CallableWithSchema, Handler, Validator
from .caller import Caller
from .exceptions import ResponseError, ValidationError
from .handler import ResponseHandler, ToolHandler
from .tool import Tool, function_schema, tool
from .validator import (
    PassthroughValidator,
    PydanticValidator,
    RegexValidator,
    ToolValidator,
)

__all__ = [
    # Base protocols
    "CallableWithSchema",
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
    "Tool",
    "tool",
    # Exceptions
    "ResponseError",
    "ValidationError",
]
