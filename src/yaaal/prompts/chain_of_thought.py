from pydantic import BaseModel, Field

from .extractor import Extract
from .summarizer import Summary
from ..core.prompt import (
    JinjaMessageTemplate,
    PassthroughMessageTemplate,
    Prompt,
    StringMessageTemplate,
)


# --------------------------------------------------------------
# CoT -> Caller
class Step(BaseModel):
    explanation: str
    output: str


class Reasoning(BaseModel):
    steps: list[Step]
    final_answer: str


# ReACT
class Step(BaseModel):
    thought: str
    action: str
    observation: str


# Reflexion
class Reflect(BaseModel): ...
