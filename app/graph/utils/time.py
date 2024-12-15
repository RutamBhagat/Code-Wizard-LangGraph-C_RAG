import time
from functools import wraps
from typing import Any, Callable
import inspect


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
                result["execution_times"][node_func.__name__] = (
                    f"{execution_time:.4f} seconds"
                )

            return result

        except Exception as e:
            raise e

    return wrapper
