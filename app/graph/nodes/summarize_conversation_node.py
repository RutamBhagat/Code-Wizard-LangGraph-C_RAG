from langchain.schema import AIMessage
from langchain_core.messages import trim_messages
from langchain_openai import ChatOpenAI
from app.graph.state import GraphState
from app.graph.utils import model
from app.graph.utils.time import track_execution_time


@track_execution_time
def summarize_conversation_node(state: GraphState):
    state.execution_times.clear()

    chat_history = (
        summarize_conversation(state)["chat_history"]
        if len(state.chat_history) > 4
        else state.chat_history
    )

    return {
        "documents": [],
        "generation": "",
        "execution_times": state.execution_times,
        "chat_history": chat_history,
    }


def summarize_conversation(state: GraphState):
    MAX_TOKENS = 100
    MODEL = "gpt-4o-mini"

    # Extract content from chat history
    chat_content = "\n".join([msg.content for msg in state.chat_history])

    # Create our summarization prompt
    summary_message = f"""
        # IDENTITY and PURPOSE
        You are an expert content summarizer. You take content in and output a Markdown formatted summary using the format below.
        # OUTPUT INSTRUCTIONS
        - You only output human readable Markdown.
        - Do not repeat items in the output sections.
        - Do not start items with the same opening words.
        INPUT:
        {chat_content}
    """

    # Add prompt to our history
    summarized_messages = [AIMessage(content=summary_message)]
    response = model.invoke(summarized_messages)

    messages = trim_messages(
        messages=state.chat_history,
        max_tokens=MAX_TOKENS,
        token_counter=ChatOpenAI(model=MODEL),
        strategy="last",
        allow_partial=False,
    )

    return {"chat_history": [response, *messages]}
