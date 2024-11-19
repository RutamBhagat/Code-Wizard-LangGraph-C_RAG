import os
from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone  # Updated import
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.graph.consts import INDEX_NAME

# Load environment variables
_ = load_dotenv(find_dotenv())

# Pinecone initialization
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "northamerica-northeast1-gcp")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set.")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = INDEX_NAME

# Get or create the index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embeddings dimension
        metric='cosine'
    )

index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(disallowed_special=set())

# Document loading and processing
current_dir = os.getcwd()
docs_path = os.path.join(
    current_dir,
    "..",
    "langchain_docs",  # replace with your path if necessary
)

loader = DirectoryLoader(docs_path, glob="**/*.md")  # Load .md files directly
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Add more structured metadata if needed (adapt as necessary)
for i, doc in enumerate(docs):
    if not doc.metadata: # Initialize if metadata is missing.
        doc.metadata = {}
    doc.metadata['source'] = doc.metadata.get('source') or f"doc_{i}"

# Pinecone VectorStore interaction
if index.describe_index_stats()["namespaces"][""]["vector_count"] == 0:
    print(f"Going to insert {len(docs)} Documents to Pinecone index {index_name}")
    docsearch = PineconeVectorStore.from_documents(
        documents=docs, embedding=embeddings, index_name=index_name, index=index
    )
    print("****** All Embeddings Added to Pinecone Vectorstore ******")
else:
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name, embedding=embeddings, index=index
    )

retriever = docsearch.as_retriever(search_kwargs={"k": 4})