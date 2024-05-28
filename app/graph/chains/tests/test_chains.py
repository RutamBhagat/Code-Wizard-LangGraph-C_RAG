from pprint import pprint
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv())  # Do not move this below other app. imports
from app.graph.chains import hallucination_grader
from app.graph.chains.hallucination_grader import (
    GradeHallucinations,
    hallucination_grader,
)
from app.graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from app.graph.chains.generation import generation_chain
from app.ingestion import retriever


# def test_retrival_grader_answer_yes() -> None:
#     question = "agent memory?"
#     docs = retriever.invoke(question)
#     doc_text = docs[0].page_content
#     res: GradeDocuments = retrieval_grader.invoke(
#         {"question": question, "document": doc_text}
#     )
#     assert res.is_document_relevant == "yes"


# def test_retrival_grader_answer_no() -> None:
#     question = "how to make pizza"
#     docs = retriever.invoke(question)
#     doc_text = docs[0].page_content
#     res: GradeDocuments = retrieval_grader.invoke(
#         {"question": question, "document": doc_text}
#     )
#     assert res.is_document_relevant == "no"


# def test_generation_chain() -> None:
#     question = "agent memory"
#     docs = retriever.invoke(question)
#     generation = generation_chain.invoke({"context": docs, "question": question})
#     pprint(generation)


# def test_hallucination_grader_answer_yes() -> None:
#     question = "agent memory"
#     docs = retriever.invoke(question)

#     generation = generation_chain.invoke({"context": docs, "question": question})
#     res: GradeHallucinations = hallucination_grader.invoke(
#         {"documents": docs, "generation": generation}
#     )
#     assert res.is_grounded == True


def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "As an OpenAI model I can not answer that question",
        }
    )
    assert res.is_grounded == False
