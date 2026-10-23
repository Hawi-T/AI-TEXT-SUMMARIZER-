"""
summarizer_langchain.py
=======================
LangChain module for AI Text Summarizer.
Demonstrates building an LCEL (LangChain Expression Language) chain:
PromptTemplate -> ChatGoogleGenerativeAI -> with_structured_output(SummaryResponse).

Educational Concept: What is a LangChain Chain?
----------------------------------------------
A "chain" connects multiple AI steps into a single reusable pipeline:
  [Input Text] -> [Prompt Template] -> [Gemini Model] -> [Structured Output] -> [SummaryResponse Object]
Each step passes its result directly to the next step using the pipe operator (|).
"""

import json
from typing import Tuple
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


class SummaryResponse(BaseModel):
    """
    Pydantic schema defining the structured output format.
    LangChain uses this model to enforce the schema on the model response.
    """
    summary: str = Field(
        description="The executive summary of the source text according to the requested style."
    )
    key_points: list[str] = Field(
        description="A list of 3 to 5 core takeaways or key points."
    )
    keywords: list[str] = Field(
        description="A list of 3 to 5 topic tags or keywords."
    )


def summarize_with_langchain(
    text: str,
    style: str,
    api_key: str,
    model: str = "gemini-3.8-flash",
) -> Tuple[str, SummaryResponse]:
    """
    Executes a LangChain chain to summarize text using Google Gemini with structured output.
    
    Returns:
        tuple of (raw_json_str, SummaryResponse)
    """
    # 1. Initialize the Gemini chat model via LangChain
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
    )

    # 2. Bind the Pydantic schema to the model for structured output
    # What is .with_structured_output()?
    # It instructs the model to return a structured Pydantic object instead of a raw message.
    structured_llm = llm.with_structured_output(SummaryResponse)

    # 3. Create the Prompt Template using LangChain
    # System role + style instruction + anti-hallucination constraint
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert executive research assistant and professional synthesizer.\n"
            "Style: {style}.\n"
            "- Extract 3 to 5 clear, actionable key points.\n"
            "- Extract 3 to 5 keywords or topic tags.\n"
            "- Do not invent facts not present in the text.\n"
            "- Follow the requested output structure strictly."
        ),
        (
            "human",
            "Summarize the following text:\n\n{text}"
        )
    ])

    # 4. Construct the LangChain LCEL chain
    # prompt | structured_llm creates an automated pipeline
    chain = prompt | structured_llm

    # 5. Invoke the chain with variables
    parsed_result: SummaryResponse = chain.invoke({
        "style": style,
        "text": text,
    })

    # Convert the parsed model back to JSON string for inspection
    raw_json_str = parsed_result.model_dump_json(indent=2)
    return raw_json_str, parsed_result
