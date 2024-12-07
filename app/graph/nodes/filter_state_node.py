from langchain_core.messages import RemoveMessage
from app.graph.state import GraphState
from app.graph.utils.time import track_execution_time


@track_execution_time
def filter_state_node(state: GraphState):
    # Clear all keys from execution_times
    for key in list(state.execution_times.keys()):
        del state.execution_times[key]

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state.chat_history[:-2]]

    return {
        "documents": [],
        "generation": "",
        "execution_times": state.execution_times,
        "chat_history": delete_messages,
    }
