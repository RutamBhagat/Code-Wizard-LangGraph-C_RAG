from typing import Any, Dict

from langchain.schema import AIMessage, HumanMessage
from app.graph.chains.generation_chain import generation_chain
from app.graph.state import GraphState
from app.graph.utils.time import track_execution_time


@track_execution_time
def generate_node(state: GraphState) -> Dict[str, Any]:
    generation = generation_chain.invoke(
        {
            "context": state.documents,
            "question": state.enhanced_query,
            "chat_history": state.chat_history or [],
        }
    )
    return {
        "chat_history": [
            HumanMessage(content=state.question),
            AIMessage(content=generation),
        ],
        "generation": generation,
    }
