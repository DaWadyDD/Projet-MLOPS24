import logging
from functools import wraps
from typing import TypeVar, Callable, Any

F = TypeVar('F', bound=Callable[..., Any])

def log_function_call(func: F) -> F:
    """
    Décorateur pour enregistrer l'appel de fonction et ses arguments.

    Paramètres
    ----------
    func : Callable
        Fonction à décorer.

    Retourne
    -------
    Callable
        Fonction décorée.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = logging.getLogger(func.__code__.co_filename)
        logger.info(f"Calling {func.__name__} with args: {args} and kwargs: {kwargs}")
        result = func(*args, **kwargs)
        return result
    return wrapper