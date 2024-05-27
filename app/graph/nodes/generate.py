from typing import Any, Dict
from app.graph.chains.generation import generation_chain
from app.graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("Generating answer...")
    question = state.question
    documents = state.documents
    print("Length of Documents: ", len(documents))
    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "generation": generation, "question": question}
