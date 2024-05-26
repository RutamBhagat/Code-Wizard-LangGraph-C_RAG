from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


_ = load_dotenv(find_dotenv())

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]

docs_list = [item for sublist in docs for item in sublist]

# print("Length of docs: ", len(docs_list))
# print("Docs List: ", docs_list)

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# print("Length of docs: ", len(doc_splits))
# print("Docs List: ", doc_splits)

# vectorstore = Chroma.from_documents(
#     doc_splits,
#     embedding=OpenAIEmbeddings(),
#     collection_name="rag_chroma",
#     persist_directory="./.chroma",
# )

retriever = Chroma(
    collection_name="rag_chroma",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings(),
).as_retriever()

if __name__ == "__main__":
    question = "Agent Memory"
    docs = retriever.invoke(question)
    print(docs)
    print(docs[0].page_content)
