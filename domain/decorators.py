import logging
from functools import wraps
from typing import TypeVar, Callable, Any

F = TypeVar('F', bound=Callable[..., Any])

def log_function_call(func: F) -> F:
    """
    Decorator for logging the function call and its arguments.

    Parameters
    ----------
    func : Callable
        Function to decorate.

    Returns
    -------
    Callable
        Decorated function.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = logging.getLogger(func.__code__.co_filename)
        logger.info(f"Calling {func.__name__} with args: {args} and kwargs: {kwargs}")
        result = func(*args, **kwargs)
        return result
    return wrapper
