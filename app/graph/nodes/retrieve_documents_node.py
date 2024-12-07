from app.graph.nodes.generate_enhanced_query_node import generate_enhanced_query_node
from app.graph.state import GraphState
from app.graph.utils.time import time_it
from app.ingestion import retriever


@time_it
def retrieve_documents_node(state: GraphState) -> GraphState:
    """Node for retrieving documents using the enhanced query"""
    new_docs = retriever.invoke(state.enhanced_query)
    return {"documents": [*state.documents, *new_docs]}
