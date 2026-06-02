from functools import wraps
from typing import Callable, Optional, TypeVar, Union
from langchain.tools import tool

F = TypeVar('F', bound=Callable)


def conditional_tool(
    condition: Union[bool, Callable[[], bool]],
    *args,
    **kwargs
) -> Callable[[F], F]:
    """
    Conditional decorator that wraps the langchain @tool decorator.
    
    Applies the @tool decorator only if the condition is True.
    
    Args:
        condition: Boolean or callable that returns bool. Determines if @tool should be applied.
        *args: Additional arguments to pass to the @tool decorator.
        **kwargs: Additional keyword arguments to pass to the @tool decorator.
    
    Returns:
        Decorator function that conditionally applies @tool.
    
    Example:
        @conditional_tool(condition=True)
        def my_tool(x: int) -> int:
            return x + 1
        
        @conditional_tool(condition=lambda: os.getenv('ENABLE_TOOLS') == 'true')
        def optional_tool(x: str) -> str:
            return x.upper()
    """
    
    def decorator(func: F) -> F:
        # Evaluate condition if callable
        should_apply = condition() if callable(condition) else condition
        
        if should_apply:
            # Apply the tool decorator
            return tool(*args, **kwargs)(func)
        else:
            # Return function unchanged
            return func
    
    return decorator
