from typing import Any, Dict
from app.graph.state import GraphState
from app.ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("Retrieving data...")
    question = state["question"]
    print("Question: ", question)
    documents = retriever.invoke(question)
    print("Length of Retrieved Documents: ", len(documents))
    return {"documents": documents, "question": question}


if __name__ == "__main__":
    res = retrieve(
        {
            "question": "What is Agent Memory?",
            "generation": "",
            "is_web_search_needed": False,
            "documents": [],
        }
    )
    print(res)
