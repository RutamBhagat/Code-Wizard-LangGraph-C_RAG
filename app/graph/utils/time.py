import time
from typing import Callable, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def time_it(func: Callable) -> Callable:
    """Decorator to measure and log the execution time of a function."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Function '{func.__name__}' took {execution_time:.4f} seconds")
        if isinstance(result, dict):
            result["execution_time"] = {func.__name__: execution_time}
        return result

    return wrapper
