from typing import Any, Dict
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from app.graph.state import GraphState
from app.ingestion import retriever

# Template for generating a single enhanced query
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


def retrieve(state: GraphState) -> Dict[str, Any]:
    question = state.question
    chat_history = state.chat_history

    # Generate enhanced query considering chat history
    state.enhanced_query = generate_enhanced_query.invoke(
        {"question": question, "chat_history": chat_history or []}
    )

    # Retrieve documents using the enhanced query
    state.documents = retriever.get_relevant_documents(state.enhanced_query)

    return state


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    chat_history = [HumanMessage(content="What is Agent Memory?")]
    res = retrieve(
        GraphState(
            question="Tell me more about Agent Memory", chat_history=chat_history
        )
    )
