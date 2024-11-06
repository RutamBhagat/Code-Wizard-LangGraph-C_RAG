from typing import Any, Dict
from app.graph.chains.generation import generation_chain
from app.graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    documents = state.documents
    chat_history = state.chat_history
    state.generation = generation_chain.invoke(
        {
            "context": documents,
            "question": state.enhanced_query,
            "chat_history": chat_history or [],
        }
    )
    return state
