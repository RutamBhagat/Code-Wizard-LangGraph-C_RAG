import os
from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.md import partition_md

from app.graph.consts import INDEX_NAME


_ = load_dotenv(find_dotenv())
pc = Pinecone(environment="northamerica-northeast1-gcp")
embeddings = OpenAIEmbeddings(disallowed_special=set())

# current_dir = os.getcwd()
# docs_path = os.path.join(
#     current_dir,
#     "..",
#     "langchain_docs",
# )

# md_elements = []
# for filename in os.listdir(docs_path):
#     if filename.endswith(".md") or filename.endswith(".mdx"):
#         file_path = os.path.join(docs_path, filename)
#         md_elements.extend(partition_md(filename=file_path))

# elements = chunk_by_title(md_elements)

# documents = []
# for element in elements:
#     metadata = element.metadata.to_dict()
#     del metadata["languages"]
#     metadata["source"] = metadata["filename"]
#     documents.append(Document(page_content=element.text, metadata=metadata))

# Indexing the documents to Pinecone
# print(f"Going to insert {len(documents)} Documents to Pinecone index {INDEX_NAME}")
# docsearch = PineconeVectorStore.from_documents(
#     documents=documents,
#     embedding=embeddings,
#     index_name=INDEX_NAME,
# )
# print("****** All Embeddings Added to Pinecone Vectorstore ******")

docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME, embedding=embeddings
)

retriever = docsearch.as_retriever(search_kwargs={"k": 4})


if __name__ == "__main__":
    question = "Agent Memory?"
    docs = retriever.invoke(question)
    print(docs)
    print(docs[0].page_content)
