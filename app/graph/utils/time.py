import time
from functools import wraps
from typing import Any, Callable, Dict
from app.graph.state import GraphState
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from dotenv import load_dotenv, find_dotenv
import inspect

# Load environment variables
_ = load_dotenv(find_dotenv())

# Initialize Tavily search tool
web_search_tool = TavilySearchResults(max_results=3)


# --- Decorator ---
def track_execution_time(node_func: Callable) -> Callable:
    """Decorator to track the execution time of a node."""

    @wraps(node_func)
    async def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            if inspect.iscoroutinefunction(node_func):
                result = await node_func(*args, **kwargs)
            else:
                result = node_func(*args, **kwargs)

            execution_time = time.time() - start_time

            if isinstance(result, dict):
                if "execution_times" not in result:
                    result["execution_times"] = {}
                result["execution_times"][
                    node_func.__name__
                ] = f"{execution_time:.4f} seconds"

            return result

        except Exception as e:
            raise e

    return wrapper
