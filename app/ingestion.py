import os
from dotenv import load_dotenv, find_dotenv
from pinecone.control import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

from app.graph.consts import INDEX_NAME

# Load environment variables
_ = load_dotenv(find_dotenv())

# Pinecone initialization
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get(
    "PINECONE_ENVIRONMENT", "northamerica-northeast1-gcp"
)

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set.")

# Initialize Pinecone client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = INDEX_NAME

# Get or create the index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embeddings dimension
        metric="cosine",
    )

index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(disallowed_special=set())

# Pinecone VectorStore interaction
docsearch = PineconeVectorStore(
    index=index,
    embedding=embeddings,
)

# Create and export the retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},  # Adjust k value as needed
)
