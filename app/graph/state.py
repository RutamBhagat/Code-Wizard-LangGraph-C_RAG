from typing import Any, List, Optional

from langchain_core.pydantic_v1 import BaseModel, Field


class GraphState(BaseModel):
    """Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        is_web_search_needed: whether to add search
        documents: list of documents"""

    question: str = Field(..., description="The question to be answered")
    generation: Optional[str] = ""
    is_web_search_needed: Optional[bool] = False
    documents: Optional[List[Any]] = []
