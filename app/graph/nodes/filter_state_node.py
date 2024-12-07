from app.graph.state import GraphState
from app.graph.utils.time import track_execution_time


@track_execution_time
def filter_state_node(state: GraphState):
    state.execution_times.clear()
    return {
        "documents": [],
        "generation": "",
        "execution_times": state.execution_times,
        "chat_history": state.chat_history[-6:],
    }
