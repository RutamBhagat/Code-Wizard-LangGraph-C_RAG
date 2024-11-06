from typing import Any, Dict

from langchain.schema import AIMessage, HumanMessage
from app.graph.chains.generation import generation_chain
from app.graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    state.generation = generation_chain.invoke(
        {
            "context": state.documents,
            "question": state.enhanced_query,
            "chat_history": state.chat_history or [],
        }
    )
    state.chat_history.append(HumanMessage(content=state.question))
    state.chat_history.append(AIMessage(content=state.generation))
    return state
