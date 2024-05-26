from pprint import pprint
from dotenv import load_dotenv, find_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

from app.graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from app.graph.chains.generation import generation_chain
from app.graph.consts import INDEX_NAME


_ = load_dotenv(find_dotenv())
# Do not import this from retrieval_grader.py its giving an error because of the way you test in vscode as opposed to pycharm
pc = Pinecone(environment="northamerica-northeast1-gcp")
embeddings = OpenAIEmbeddings(disallowed_special=set())
retriever = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME, embedding=embeddings
)


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
    pprint(generation)
