from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.graph.state import GraphState
from app.graph.utils.time import track_execution_time
from app.graph.consts import MODEL_NAME
from langchain_google_genai import ChatGoogleGenerativeAI

template = """You are a query enhancement system. Your role is to analyze the user's question and chat history to create a more comprehensive search query, but only if necessary. You are not supposed to answer the question directly, but rather formulate a better question based on the chat history.
Instructions:
1. **Check Chat History:**
- If the chat history is empty, DO NOT modify the question. Return the original question as is.
2. **Assess Question Sufficiency:**
- If the question is clear, specific, and self-contained (i.e., it can be understood and answered without additional context), DO NOT modify the question. Return the original question as is even if the answer is readily available in the chat history.
3. **Enhance When Necessary:**
- If the question is ambiguous, incomplete, or relies on information from the chat history for full understanding, then enhance the query.
- In this case, formulate an enhanced search query that:
- Incorporates relevant context from the chat history.
- Maintains the original intent of the question.
- Includes important contextual details that were mentioned earlier in the chat history.
- Replaces ambiguous pronouns with their referents from the context.
Chat History for context: {chat_history}
Current question: {question}
Return only the (potentially enhanced) query text without any prefixes or explanations.
Example:
Chat History:-
User: What is LangChain?
Assistant: LangChain is a framework for developing applications powered by language models.
Question:  Explain it in detail
Expected Enhanced query: Explain LangChain in detail
Chat History:- ""
Question: "Explain what is LangChain in detail"
Enhanced query: "Explain what is LangChain in detail"
Format your response as a single query string without any prefixes or explanations."""

query_generator_prompt = ChatPromptTemplate.from_messages([("system", template)])
generate_enhanced_query = (
    query_generator_prompt
    | ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0,
    )
    | StrOutputParser()
)


@track_execution_time
def generate_enhanced_query_node(state: GraphState) -> GraphState:
    """Node for generating an enhanced query from question and chat history"""
    enhanced_query = generate_enhanced_query.invoke(
        {"question": state.question, "chat_history": state.chat_history or []}
    )

    return {
        "enhanced_query": enhanced_query,
    }
