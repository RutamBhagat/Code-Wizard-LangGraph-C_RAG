from typing import List, TypedDict

from langchain.schema import Document


class GraphState(TypedDict):
    """Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        is_web_search_needed: whether to add search
        documents: list of documents"""

    question: str
    generation: str
    is_web_search_needed: bool
    documents: List[Document]
