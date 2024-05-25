from typing import Any, Dict
from app.graph.state import GraphState
from app.ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("Retrieving data...")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
