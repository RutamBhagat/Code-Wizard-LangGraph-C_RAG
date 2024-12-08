from typing import Any, List, Dict

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field


class GraphState(BaseModel):
    """Represents the state of our graph.

    Attributes:
        question: question to be answered
        enhanced_query: query after being enhanced
        documents: list of documents
        messages: list of chat messages
        generation: response to the question
        execution_times: Dictionary to store execution times of each node.
    """

    question: str = Field(description="Question to be answered")
    enhanced_query: str = Field(default="")
    documents: List[Any] = Field(default_factory=list)
    messages: List[AnyMessage] = Field(default_factory=list)
    generation: str = Field(default="")
    execution_times: Dict[str, str] = Field(default_factory=dict)


class GraphInputState(BaseModel):
    """Represents the state of our graph.

    Attributes:
        question: question to be answered
    """

    question: str = Field(description="Question to be answered")


class GraphOutputState(BaseModel):
    """Represents the state of our graph.

    Attributes:
        generation: response to the question
    """

    generation: str = Field(default="")
