from langchain.schema import HumanMessage
from app.graph.state import GraphState
from app.graph.utils.time import track_execution_time
from app.graph.consts import MODEL_NAME
from langchain_google_genai import ChatGoogleGenerativeAI


@track_execution_time
async def summarize_conversation_node(state: GraphState):
    execution_times = {}
    messages = state.messages

    if len(messages) > 4:
        summarized_state = await summarize_conversation(state)
        messages = summarized_state.messages
        execution_times = summarized_state.execution_times

    return {
        "enhanced_query": "",
        "documents": [],
        "messages": messages,
        "generation": "",
        "execution_times": execution_times,
    }


async def summarize_conversation(state: GraphState):
    MAX_TOKENS = 250

    chat_content = "\n".join([msg.content for msg in state.messages])

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

    summarized_messages = [HumanMessage(content=summary_message)]
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    response = await llm.ainvoke(summarized_messages)

    messages = state.messages[-2:]
    state.messages = [
        HumanMessage(content="Here's a summary of our conversation up to this point:"),
        response,
        *messages,
    ]
    return state
