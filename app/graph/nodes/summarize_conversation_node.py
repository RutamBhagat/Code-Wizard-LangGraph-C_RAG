from langchain.schema import AIMessage
from langchain_core.messages import trim_messages
from langchain_openai import ChatOpenAI
from app.graph.consts import MODEL_NAME
from app.graph.state import GraphState
from app.graph.utils.time import track_execution_time


@track_execution_time
def summarize_conversation_node(state: GraphState):
    state.execution_times.clear()

    if len(state.chat_history) > 4:
        state = summarize_conversation(state)
        chat_history = state.chat_history
    else:
        chat_history = state.chat_history

    return {
        "enhanced_query": "",
        "documents": [],
        "chat_history": chat_history,
        "generation": "",
        "execution_times": state.execution_times,
    }


def summarize_conversation(state: GraphState):
    MAX_TOKENS = 250
    MODEL = "gpt-4o-mini"

    # Extract content from chat history
    chat_content = "\n".join([msg.content for msg in state.chat_history])

    # Create our summarization prompt
    summary_message = f"""
Summarize the following conversation in a clear and concise way.

Requirements:
- Provide a brief overview of the main topics and key points
- Use simple, direct language
- Stay under {MAX_TOKENS} tokens
- Format output in Markdown
- Focus on essential information only

Input conversation:
{chat_content}

Format your response as:
# Summary
[Your concise summary here]

# Key Points
- [Point 1]
- [Point 2]
- [Point 3]
"""

    # Add prompt to our history
    summarized_messages = [AIMessage(content=summary_message)]
    model = ChatOpenAI(model=MODEL_NAME, temperature=0)
    response = model.invoke(summarized_messages)

    ## DO NOT DELETE THIS COULD BE USED LATER IN PROD
    # messages = trim_messages(
    #     messages=state.chat_history,
    #     max_tokens=MAX_TOKENS,
    #     token_counter=ChatOpenAI(model=MODEL),
    #     strategy="last",
    #     allow_partial=False,
    # )

    messages = state.chat_history[-2:]
    state.chat_history = [response, *messages]
    return state
