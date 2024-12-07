from langchain_core.messages import trim_messages
from langchain_openai import ChatOpenAI
from app.graph.state import GraphState
from app.graph.utils.time import track_execution_time


@track_execution_time
def filter_state_node(state: GraphState):
    MAX_TOKENS = 1000
    MODEL = "gpt-4o-mini"

    state.execution_times.clear()

    messages = (
        trim_messages(
            messages=state.chat_history,
            max_tokens=MAX_TOKENS,
            token_counter=ChatOpenAI(model=MODEL),
            strategy="last",
            allow_partial=False,
        )
        if len(state.chat_history) > 2
        else state.chat_history
    )

    return {
        "documents": [],
        "generation": "",
        "execution_times": state.execution_times,
        "chat_history": messages,
    }
