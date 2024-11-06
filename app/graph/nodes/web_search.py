from typing import Any, Dict
from dotenv import load_dotenv, find_dotenv
from langchain.schema import Document
from langchain_community.tools.tavily_search.tool import TavilySearchResults

from app.graph.state import GraphState


_ = load_dotenv(find_dotenv())
web_search_tool = TavilySearchResults(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    tavily_results = web_search_tool.invoke({"query": state.enhanced_query})
    joined_tavily_results = "\n".join(
        [
            result["content"]
            for result in tavily_results
            if isinstance(result, dict) and "content" in result
        ]
    )

    if joined_tavily_results:
        web_results = Document(page_content=joined_tavily_results)
        state.documents.append(web_results)

    return state


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    chat_history = [HumanMessage(content="What is Agent Memory?")]
    web_search(state={"chat_history": chat_history})
