import time
import httpx
from typing import Any, Dict

from app.graph.chains.retrieval_grader import retrieval_grader
from app.graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """Determines whether the retrieved documents are relevant to the question
    If any document is not relevant we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated is_web_search_needed flag
    """

    print("Checking the relevance of the retrieved documents to the question")
    question = state.question
    documents = state.documents

    filtered_docs = []
    is_web_search_needed = False

    for doc in documents:
        try:
            is_document_relevant = retrieval_grader.invoke(
                {"question": question, "document": doc}
            ).is_document_relevant

            if is_document_relevant:
                print("GRADE: Document is relevant")
                filtered_docs.append(doc)
            else:
                print("GRADE: Document is not relevant")
                is_web_search_needed = True

            # Introduce a delay to avoid rate limiting
            time.sleep(1)  # wait for 1 second
        except httpx.HTTPStatusError as e:
            print(f"Error: {e}")
            # Handle the error, you can retry the request or log the error

    print("Length of Filtered Documents: ", len(filtered_docs))
    print("Is Web Search Needed: ", is_web_search_needed)
    return {
        "documents": filtered_docs,
        "is_web_search_needed": is_web_search_needed,
        "question": question,
    }
