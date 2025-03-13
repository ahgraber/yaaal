from pydantic import BaseModel, Field

from .extractor import Extract
from .summarizer import Summary
from ..core.template import (
    ConversationTemplate,
    JinjaMessageTemplate,
    StringMessageTemplate,
    UserMessageTemplate,
)


# --------------------------------------------------------------
# CoT
class CoTStep(BaseModel):
    explanation: str
    output: str


class CoT(BaseModel):
    steps: list[CoTStep]
    final_answer: str


# --------------------------------------------------------------
# ReACT
# action is a tool invocation; implies ReACT is a graph not a prompt
class ReACTStep(BaseModel):
    thought: str
    action: str
    observation: str


class ReACT(BaseModel):
    steps: list[ReACTStep]
    final_answer: str
