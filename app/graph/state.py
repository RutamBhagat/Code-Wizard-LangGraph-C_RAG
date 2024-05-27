from typing import Any, List, Optional

from pydantic import BaseModel


class GraphState(BaseModel):
    """Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        is_web_search_needed: whether to add search
        documents: list of documents"""

    question: str
    generation: Optional[str] = ""
    is_web_search_needed: bool = False
    documents: Optional[List[Any]] = []
