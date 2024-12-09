from typing import Any, Dict

from langchain.schema import AIMessage, HumanMessage
from app.graph.chains.generation_chain import generation_chain
from app.graph.state import GraphState
from app.graph.utils.time import track_execution_time


@track_execution_time
async def generate_node(state: GraphState) -> Dict[str, Any]:
    generation = await generation_chain.ainvoke(
        {
            "context": state.documents,
            "question": state.enhanced_query,
            "messages": state.messages or [],
        }
    )

    state.messages.extend(
        [HumanMessage(content=state.question), AIMessage(content=generation)]
    )

    return {
        "messages": state.messages,
        "generation": generation,
    }
