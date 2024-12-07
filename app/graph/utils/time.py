import time
from functools import wraps
from typing import Any, Callable, Dict
from app.graph.state import GraphState
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_ = load_dotenv(find_dotenv())

# Initialize Tavily search tool
web_search_tool = TavilySearchResults(max_results=3)


# --- Decorator ---
def track_execution_time(node_func: Callable) -> Callable:
    """Decorator to track the execution time of a node."""

    @wraps(node_func)
    def wrapper(state: GraphState) -> Dict[str, Any]:
        start_time = time.time()
        updated_data = node_func(state)
        end_time = time.time()
        elapsed_time = f"{(end_time - start_time):.2f} seconds"

        # Update execution_times
        updated_data["execution_times"] = {
            **state.execution_times,
            **updated_data.get("execution_times", {}),
            node_func.__name__: elapsed_time,
        }
        return updated_data

    return wrapper
