from typing import Any, Dict
from app.graph.chains.generation import generation_chain
from app.graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("Generating answer...")
    question = state.question
    documents = state.documents
    chat_history = state.chat_history
    print("Length of Documents: ", len(documents))
    generation = generation_chain.invoke(
        {"context": documents, "question": question, "chat_history": chat_history}
    )
    return {"generation": generation}
