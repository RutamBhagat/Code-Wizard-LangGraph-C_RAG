from typing import Any, Dict
from dotenv import load_dotenv, find_dotenv
from langchain.schema import Document
from langchain_community.tools.tavily_search.tool import TavilySearchResults

from app.graph.state import GraphState
from app.graph.utils.time import track_execution_time


_ = load_dotenv(find_dotenv())
web_search_tool = TavilySearchResults(max_results=3)


@track_execution_time
async def web_search_node(state: GraphState) -> Dict[str, Any]:
    tavily_results = await web_search_tool.ainvoke({"query": state.enhanced_query})
    web_results = [
        Document(page_content=result["content"])
        for result in tavily_results
        if "content" in result
    ]
    return {"documents": [*state.documents, *web_results]}
