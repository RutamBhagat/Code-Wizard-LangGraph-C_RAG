import operator
from typing import Any, List, Annotated, Optional, Sequence, Union

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage


class GraphState(BaseModel):
    """Represents the state of our graph.

    Attributes:
        generation: LLM generation
        is_web_search_needed: whether to add search
        documents: list of documents
        chat_history: list of chat messages
    """

    question: Optional[str] = Field(description="Question to be answered")
    generation: Optional[str] = ""
    is_web_search_needed: Optional[bool] = False
    documents: Optional[List[Any]] = []
    chat_history: Annotated[Sequence[Union[HumanMessage, AIMessage]], operator.add] = []
