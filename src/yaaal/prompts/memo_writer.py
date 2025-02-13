from pydantic import BaseModel, Field

from .extractor import Extract
from .summarizer import Summary
from ..core.template import (
    ConversationTemplate,
    JinjaMessageTemplate,
    PassthroughMessageTemplate,
    StringMessageTemplate,
)
from ..tools.web import URLContent

MemoWriter_prompt_template = """
You are a professional ghostwriter. Your task is to review this information and draft an informative and engaging memo based on the provided sources that summarizes the key insights and takeaways. The user may provide additional guidance to help you complete the task.

Follow these instructions carefully:

1. Synthesize key takeaways:
   - Identify common themes, trends, or insights across the source content.
   - Look for unique or contrasting perspectives on similar topics.
   - Consider the implications of the information with respect to the user's guidance.
   - Aim to produce 3-7 key takeaways that provide valuable insights for professionals.

2. Provide citations:
   - For each summary and key takeaway, include a citation(s) to the original source content.
   - If multiple sources support a takeaway, include all relevant citations.
   - Use markdown links as citations `[title](url)`.

3. Format your output as follows:

```md
# [Insert an engaging subject line]

## Summary

[Insert your summary of the key themes and insights from the sources]

## Key Takeaways

1. [Insert key takeaway 1]
   (Citations)

2. [Insert key takeaway 2]
   (Citations)

...[Continue for all key takeaways]

```

Here is the content to be summarized and included in the newsletter:

<sources>
{% for source in sources %}
    <source>
    {{source}}
    </source>
{% endfor %}
</sources>

""".strip()


class MemoWriterSystemVars(BaseModel):
    source: URLContent | Summary | Extract = Field(description="The text to be analyzed")


class MemoWriterUserVars(BaseModel):
    guidance: str = Field(description="The user guidance to direct the draft")


memowriter_prompt = ConversationTemplate(
    name="Memo Writer",
    description="Generate a summary of provided content",
    templates=[
        JinjaMessageTemplate(
            name="System Instructions",
            role="system",
            template=MemoWriter_prompt_template,
            validation_model=MemoWriterSystemVars,
        ),
        PassthroughMessageTemplate(name="User request"),
    ],
)
