from typing import Any, Dict
from app.graph.chains.generation import generation_chain
from app.graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    question = state.question
    documents = state.documents
    chat_history = state.chat_history
    generation = generation_chain.invoke(
        {"context": documents, "question": question, "chat_history": chat_history or []}
    )
    return {"generation": generation}
