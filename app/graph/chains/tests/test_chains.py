from pprint import pprint
from dotenv import load_dotenv, find_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from app.graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from app.graph.chains.generation import generation_chain


_ = load_dotenv(find_dotenv())
# Do not import this from retrieval_grader.py its giving an error because of the way you test in vscode as opposed to pycharm
retriever = Chroma(
    collection_name="rag_chroma",
    persist_directory="/home/voldemort/Downloads/Code/Eden/C_RAG/app/.chroma",
    embedding_function=OpenAIEmbeddings(),
).as_retriever()


def test_retrival_grader_answer_yes() -> None:
    question = "agent memory?"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_text}
    )
    assert res.is_document_relevant == "yes"


def test_retrival_grader_answer_no() -> None:
    question = "how to make pizza"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_text}
    )
    assert res.is_document_relevant == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    # pprint(generation)
