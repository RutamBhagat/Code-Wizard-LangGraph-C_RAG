import operator
from typing import Any, List, Annotated, Sequence, Union

from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field


class GraphState(BaseModel):
    """Represents the state of our graph.

    Attributes:
        documents: list of documents
        chat_history: list of chat messages
    """

    question: str = Field(description="Question to be answered")
    enhanced_query: str = Field(default="")
    documents: Annotated[List[Any], operator.add] = Field(default_factory=list)
    chat_history: Annotated[Sequence[Union[HumanMessage, AIMessage]], operator.add] = (
        Field(default_factory=list)
    )
