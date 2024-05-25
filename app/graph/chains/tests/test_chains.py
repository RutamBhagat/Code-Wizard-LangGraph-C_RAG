from dotenv import load_dotenv, find_dotenv
from app.ingestion import retriever
from app.graph.chains.retrieval_grader import GradeDocuments, retrieval_grader

_ = load_dotenv(find_dotenv())


def test_retrival_grader_answer_yes() -> None:
    question = "What is agent memory?"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_text}
    )
    assert res.is_document_relevant == "yes"
