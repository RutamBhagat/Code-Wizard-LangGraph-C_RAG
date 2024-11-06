# enhanced_query_node.py
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from app.graph.state import GraphState

template = """You are an AI language model assistant. 
            Your task is to generate an enhanced search query based on the user's question and chat history.
            Consider the context from the chat history to create a more informative query.
            Chat History for context: {chat_history}
            Current question: {question}
            Generate a single, comprehensive search query that captures the user's intent and context.
            Keep the query concise but informative. Return only the enhanced query text."""

query_generator_prompt = ChatPromptTemplate.from_messages([("system", template)])
generate_enhanced_query = (
    query_generator_prompt | ChatOpenAI(temperature=0) | StrOutputParser()
)


def generate_enhanced_query_node(state: GraphState) -> GraphState:
    """Node for generating an enhanced query from question and chat history"""
    state.enhanced_query = generate_enhanced_query.invoke(
        {"question": state.question, "chat_history": state.chat_history or []}
    )
    return state
