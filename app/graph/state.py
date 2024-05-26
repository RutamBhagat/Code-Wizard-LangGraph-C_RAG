from typing import List, Optional, TypedDict


class GraphState(TypedDict):
    """Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        is_web_search_needed: whether to add search
        documents: list of strings"""

    question: str
    generation: Optional[str] = None
    is_web_search_needed: Optional[bool] = None
    documents: Optional[List[str]] = None
