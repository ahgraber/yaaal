from pydantic import BaseModel, Field

from ..core import (
    JinjaMessageTemplate,
    PassthroughMessageTemplate,
    Prompt,
    StringMessageTemplate,
)
from ..tools.web import URLContent


class Summary(BaseModel, extra="ignore"):
    url: str  # Annotated[str, AnyHttpUrl]
    title: str
    summary: str = Field(
        description="A comprehensive but concise summary of the source content that captures the essence of the original information."
    )


summarizer_prompt_template = f"""
You are an AI research assistant. Your task is to summarize a piece of content and synthesize key takeaways. The user may provide additional guidance for topics of interest or directions for investigation.

Please follow these steps to complete your task:

1. Carefully read and analyze the provided content.

2. Summarize the main points of the content. Your summary should be detailed and comprehensive, capturing the essence of the content and the source's relevance with respect to the user's guidance.

3. If it exists, consider the user-provided guidance and ensure that your summary and analysis address the specified topics of interest or directions for investigation.

4. The summary may use up to three paragraphs to highlight the main idea, argument or goal, clarify critical information, and identify actionable insights or key takeaways. The summary should be presented as a text document, not as a JSON object.

5. Once written, format the summary as a JSON object with the following structure:

{Summary.model_json_schema()}

Here is the source you need to analyze:

<source>
{{{{source}}}}
</source>
""".strip()


class SummarizerSystemVars(BaseModel):
    source: URLContent = Field(description="The text to be analyzed")


class SummarizerUserVars(BaseModel):
    guidance: str = Field(description="The user guidance to focus the analysis")


SummarizerPrompt = Prompt(
    name="Summarizer",
    description="Generate a summary of provided content",
    system_template=JinjaMessageTemplate(
        role="system",
        template=summarizer_prompt_template,
        template_vars_model=SummarizerSystemVars,
    ),
    user_template=PassthroughMessageTemplate(),
)
