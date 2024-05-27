from typing import Any, Dict
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads
from app.graph.state import GraphState
from app.ingestion import retriever


# Multi Query: Different Perspectives
template = """You are an AI language model assistant. Your task is to generate five 
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


def get_unique_union(documents: list[list]):
    """Unique union of retrieved docs"""
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("Retrieving data...")
    question = state["question"]
    print("Question: ", question)
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    documents = retrieval_chain.invoke({"question": question})
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
