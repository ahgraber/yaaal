from pydantic import BaseModel, Field

from ..core.prompt import (
    JinjaMessageTemplate,
    PassthroughMessageTemplate,
    Prompt,
    StringMessageTemplate,
)
from ..tools.web import URLContent


class BaseContext(BaseModel):
    context: str = Field(description="Supporting quotation or paraphrase from source document")


class Fact(BaseContext, extra="ignore"):
    fact: str = Field(description="A concrete, verifiable piece of information")


class Concept(BaseContext, extra="ignore"):
    concept: str = Field(description="A complete abstract idea or theory")


class Opinion(BaseContext, extra="ignore"):
    opinion: str = Field(
        description="An impression, judgement, belief, or appraisal that is held by an individual and informed by their experience"
    )


class Topic(BaseContext, extra="ignore"):
    topic: str = Field(description="A main subject matter or theme")


class Trend(BaseContext, extra="ignore"):
    trend: str = Field(description="A pattern, movement, or direction of change")


class Extract(BaseModel, extra="ignore"):
    url: str
    title: str
    information: list[Fact | Concept | Opinion | Topic | Trend] = Field(
        description="specific facts, concepts, opinions, topics, and trends, including supporting context",
        min_length=3,
    )


extractor_prompt_template = f"""
You are an AI research assistant. Your task is to extract distinct pieces of information from content. The user may provide additional guidance for topics of interest or directions for investigation.

Please follow these steps to complete your task:

1. Carefully read and analyze the entire content.

2. Extract information in the following categories

   a. Facts: Identify and list concrete, verifiable pieces of information presented in the content.
   b. Concepts: Identify abstract ideas or theories discussed in the content.
   c. Opinions: Note any impressions, judgements, or beliefs held by individuals mentioned in the content.
   d. Topics: Determine the main subjects or themes covered in the content.
   e. Trends: Identify any patterns, movements, or directions of change discussed in the content.

3. Guidelines for extraction:

   - Extract every piece of information you can find
   - Extracted information should be concise but also cite sufficient context from the source document to back up the claim
   - If it exists, consider the user guidance provided and ensure that the information characterized with respect to the user's interests.
   - If information can be attributed to an individual, ensure to cite the source
   - Ensure each extracted item is distinct and non-redundant

4. Present your analysis adhering to the following json schema:

{Extract.model_json_schema()}

Here is the source content you need to analyze:

<source>
{{{{source}}}}
</source>
""".strip()


class ExtractorSystemVars(BaseModel):
    source: URLContent = Field(description="The text to be analyzed")


class ExtractorUserVars(BaseModel):
    guidance: str = Field(description="The user guidance to focus the analysis")


extractor_prompt = Prompt(
    name="Extractor",
    description="Extract information from provided content",
    system_template=JinjaMessageTemplate(
        role="system",
        template=extractor_prompt_template,
        template_vars_model=ExtractorSystemVars,
    ),
    user_template=PassthroughMessageTemplate(),
)
