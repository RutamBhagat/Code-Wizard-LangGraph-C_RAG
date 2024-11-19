import os
from dotenv import load_dotenv, find_dotenv
import pinecone  # Import pinecone directly
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
# from unstructured.chunking.title import chunk_by_title # uncomment if using
# from unstructured.partition.md import partition_md  # uncomment if using
from langchain.document_loaders import DirectoryLoader  # For loading from a directory
from langchain.text_splitter import RecursiveCharacterTextSplitter # For splitting documents

from app.graph.consts import INDEX_NAME

# Load environment variables
_ = load_dotenv(find_dotenv())

# Pinecone initialization
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "northamerica-northeast1-gcp")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set.")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

index_name = INDEX_NAME  # Use the constant for the index name

# Get or create the index (include logic here if needed)
index = pinecone.Index(index_name)

embeddings = OpenAIEmbeddings(disallowed_special=set())



# Document loading and processing
current_dir = os.getcwd()
docs_path = os.path.join(
    current_dir,
    "..",
    "langchain_docs", # replace with your path if necessary
)

# md_elements = []
# for filename in os.listdir(docs_path):
#     if filename.endswith(".md") or filename.endswith(".mdx"):
#         file_path = os.path.join(docs_path, filename)
#         md_elements.extend(partition_md(filename=file_path))

# elements = chunk_by_title(md_elements)

loader = DirectoryLoader(docs_path, glob="**/*.md")  # Load .md files directly
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)




# documents = []
# for element in elements:
#     metadata = element.metadata.to_dict()
#     del metadata["languages"]
#     metadata["source"] = metadata["filename"]
#     documents.append(Document(page_content=element.text, metadata=metadata))


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