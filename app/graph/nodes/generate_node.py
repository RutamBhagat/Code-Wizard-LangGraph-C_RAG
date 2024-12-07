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

    state.chat_history.extend(
        [HumanMessage(content=state.question), AIMessage(content=generation)]
    )

    return {
        "chat_history": state.chat_history,
        "generation": generation,
    }
