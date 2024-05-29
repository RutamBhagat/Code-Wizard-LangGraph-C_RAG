from typing import Any, Dict
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads
from app.graph.state import GraphState
from app.ingestion import retriever


# Utility Function
def reciprocal_rank_fusion(results: list[list], k=60):
    """Reciprocal_rank_fusion that takes multiple lists of ranked documents
    and an optional parameter k used in the RRF formula"""

    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] = previous_score + 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


# Multi Query: Different Perspectives
template = """You are an AI language model assistant. Your task is to generate three 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)


generate_queries = (
    prompt_perspectives
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("Retrieving data...")
    question = state.question
    print("Question: ", question)
    retrieval_chain_rag_fusion = (
        generate_queries | retriever.map() | reciprocal_rank_fusion
    )
    documents = retrieval_chain_rag_fusion.invoke({"question": question})
    print("Length of Retrieved Documents: ", len(documents))
    # only take top 4 documents because of the limited context window
    # if the length of documents is less than 4 then take all
    documents = documents[:4] if len(documents) > 4 else documents
    print("Length of Top Retrieved Documents: ", len(documents))
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
