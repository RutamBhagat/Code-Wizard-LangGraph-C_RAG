from typing import Any, Dict
from dotenv import load_dotenv, find_dotenv
from langchain.schema import Document
from langchain_community.tools.tavily_search.tool import TavilySearchResults

from app.graph.state import GraphState


_ = load_dotenv(find_dotenv())
web_search_tool = TavilySearchResults(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("Searching the web for additional context...")
    question = state["question"]
    documents = state["documents"]
    tavily_results = web_search_tool.invoke({"query": question})
    joined_tavily_results = "\n".join([inst["content"] for inst in tavily_results])
    web_results = Document(page_content=joined_tavily_results)
    if documents is None:
        documents = []
    documents.append(web_results)
    return {"documents": documents, "question": question}


if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})
