from dotenv import load_dotenv, find_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.document_loaders import UnstructuredURLLoader
from unstructured.cleaners.core import remove_punctuation, clean, clean_extra_whitespace
from langchain_community.vectorstores.utils import filter_complex_metadata


_ = load_dotenv(find_dotenv())


def generate_document(urls):
    "Given an URL, return a langchain Document to futher processing"
    loader = UnstructuredURLLoader(
        urls=urls,
        mode="elements",
        post_processors=[
            clean,
            remove_punctuation,
            clean_extra_whitespace,
        ],
    )
    elements = loader.load()
    return filter_complex_metadata(elements)


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs_list = generate_document(urls)

# print("Length of docs: ", len(docs_list))
# print("Docs List: ", docs_list)

# vectorstore = Chroma.from_documents(docs_list,
#                                     embedding=OpenAIEmbeddings(),
#                                     collection_name="rag_chroma",
#                                     persist_directory="./.chroma")

retriever = Chroma(
    collection_name="rag_chroma",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings(),
).as_retriever()
