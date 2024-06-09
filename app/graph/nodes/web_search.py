from typing import Any, Dict
from dotenv import load_dotenv, find_dotenv
from langchain.schema import Document
from langchain_community.tools.tavily_search.tool import TavilySearchResults

from app.graph.state import GraphState


_ = load_dotenv(find_dotenv())
web_search_tool = TavilySearchResults(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("Searching the web for additional context...")
    question = state.chat_history[-1].content
    documents = state.documents or []  # Handle None case

    tavily_results = web_search_tool.invoke({"query": question})
    joined_tavily_results = "\n".join(
        [
            result["content"]
            for result in tavily_results
            if isinstance(result, dict) and "content" in result
        ]
    )

    if joined_tavily_results:
        web_results = Document(page_content=joined_tavily_results)
        documents.append(web_results)

    return {"documents": documents}


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    chat_history = [HumanMessage(content="What is Agent Memory?")]
    web_search(state={"chat_history": chat_history})
