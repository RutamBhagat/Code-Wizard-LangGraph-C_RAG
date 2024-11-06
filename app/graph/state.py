import operator
from typing import Any, List, Annotated, Optional, Sequence, Union

from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field


class GraphState(BaseModel):
    """Represents the state of our graph.

    Attributes:
        generation: LLM generation
        is_web_search_needed: whether to add search
        documents: list of documents
        chat_history: list of chat messages
    """

    question: str = Field(description="Question to be answered")
    enhanced_query: Optional[str] = ""
    generation: Optional[str] = ""
    is_web_search_needed: Optional[bool] = False
    documents: Optional[List[Any]] = []
    chat_history: Optional[
        Annotated[Sequence[Union[HumanMessage, AIMessage]], operator.add]
    ] = []
