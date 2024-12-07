from langchain_core.messages import trim_messages
from langchain_openai import ChatOpenAI
from app.graph.state import GraphState
from app.graph.utils.time import track_execution_time


@track_execution_time
def filter_state_node(state: GraphState):
    state.execution_times.clear()
    messages = trim_messages(
        messages=state.chat_history,
        max_tokens=1000,
        token_counter=ChatOpenAI(model="gpt-4o-mini"),
        strategy="last",
        allow_partial=False,
    )
    return {
        "documents": [],
        "generation": "",
        "execution_times": state.execution_times,
        "chat_history": messages,
    }
