import os
from dotenv import load_dotenv, find_dotenv
from pinecone.control import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

from app.graph.consts import INDEX_NAME

_ = load_dotenv(find_dotenv())

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get(
    "PINECONE_ENVIRONMENT", "northamerica-northeast1-gcp"
)

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set.")

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = INDEX_NAME

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
    )

index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(disallowed_special=set())

docsearch = PineconeVectorStore(
    index=index,
    embedding=embeddings,
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)
