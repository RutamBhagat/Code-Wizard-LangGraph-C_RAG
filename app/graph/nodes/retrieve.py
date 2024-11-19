# retrieval_node.py
from app.graph.nodes.enhanced_query import generate_enhanced_query_node
from app.graph.state import GraphState
from app.ingestion import retriever


def retrieve_documents_node(state: GraphState) -> GraphState:
    """Node for retrieving documents using the enhanced query"""
    # Updated to use invoke() instead of deprecated get_relevant_documents()
    state.documents = retriever.invoke(state.enhanced_query)
    return state


# Usage example
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    # Create initial state
    state = GraphState(
        question="Tell me more about Agent Memory",
        chat_history=[HumanMessage(content="What is Agent Memory?")],
    )

    # Execute nodes in sequence
    state = generate_enhanced_query_node(state)
    state = retrieve_documents_node(state)
