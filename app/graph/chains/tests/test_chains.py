from pprint import pprint
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv())  # Do not move this below other app. imports

from app.ingestion import retriever
from app.graph.chains.generation import generation_chain
from app.graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from app.graph.chains.router import question_router
from app.graph.chains.hallucination_grader import (
    GradeHallucinations,
    hallucination_grader,
)


def test_retrival_grader_answer_yes() -> None:
    question = "LangChain Expression Language?"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_text}
    )
    assert res.is_document_relevant == True


def test_retrival_grader_answer_no() -> None:
    question = "how to make pizza"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_text}
    )
    # pprint(res)
    assert res.is_document_relevant == False


def test_generation_chain() -> None:
    question = "LangChain Expression Language?"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    # pprint(generation)


def test_hallucination_grader_answer_yes() -> None:
    question = "LangChain Expression Language?"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.is_grounded == True


def test_hallucination_grader_answer_no() -> None:
    question = "LangChain Expression Language?"
    docs = retriever.invoke(question)

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "As an OpenAI model I can not answer that question",
        }
    )
    assert res.is_grounded == False


def test_router_to_vectorstore() -> None:
    question = "LangChain Expression Language?"
    res = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"


def test_router_to_websearch() -> None:
    question = "Who is Harrison Chase?"
    res = question_router.invoke({"question": question})
    assert res.datasource == "web_search"
