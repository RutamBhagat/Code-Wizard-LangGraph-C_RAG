from typing import Dict
from app.graph.state import GraphState


def retrieve_and_web_search(state: GraphState) -> Dict:
    """
    Intermediate node to fork the flow to both retrieve and web search.
    This node does not modify the state.
    """
    return state


def combine_documents(state: GraphState) -> Dict:
    """
    Intermediate node to fork the flow to both retrieve and web search.
    This node does not modify the state.
    """
    return state
