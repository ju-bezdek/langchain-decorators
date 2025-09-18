import asyncio
from langchain_core.tools import BaseTool, tool, StructuredTool
import functools
import inspect
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from typing import (
    TYPE_CHECKING,
    Any,
    Type,
    get_type_hints,
    get_origin,
    get_args,
    Literal,
    overload,
    Callable,
    TypeVar,
    Generic,
    List,
    Optional,
    Union,
)
from langchain_decorators.function_decorator import build_func_schema
from langgraph.types import interrupt
from langgraph.prebuilt.interrupt import (
    HumanInterruptConfig,
    HumanInterrupt,
    ActionRequest,
)


def conditional_transition(after: str | list[str] | tuple[str]):
    """
    Decorator that marks a function as a node transition.

    Features:
        - Ensures the transition is added to the graph as conditional node transition.
        - Extracts possible transitions from the return type, to annotate the graph.
        - By default, exposes the transition to the LLM as a tool (disable by setting visible_to_llm).
    """

    def decorator(func, from_node=after):

        type_hints = get_type_hints(func)
        return_type = type_hints.get("return", None)

        # Extract transitions from Literal return type if present
        transitions = None
        if return_type and get_origin(return_type) is Command:
            return_type = get_args(return_type)[0]
        if return_type and get_origin(return_type) is Literal:
            transitions = list(get_args(return_type))
        else:
            raise TypeError(
                f"Return type of node_transition_tool (`{func.__name__}`) must be a Literal[<transitions>] or Command[Literal[<transitions>]] defining the possible transitions."
            )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            # Ensure the result is always a string
            if result and not isinstance(result, str):
                raise TypeError(
                    f"Node transition function `{func.__name__}` must return a string or None, got {type(result).__name__}."
                )
            return result

        # Add metadata attributes
        wrapper.__is_transition__ = True
        from_nodes = (from_node,) if isinstance(from_node, str) else from_node

        if transitions:
            wrapper.__transitions__ = {
                from_node: transitions for from_node in from_nodes
            }

        return wrapper

    # if func:
    #     return decorator(func)
    # else:
    return decorator


def is_node_transition(func):
    """
    Check if a function is a node transition.
    """
    return getattr(func, "__is_transition__", False)


def get_conditional_transition_edges(func) -> dict[str, tuple[str]]:
    """
    Get the transitions from a node transition function.
    """
    res = getattr(func, "__transitions__", None)
    if res is None:
        raise ValueError(
            f"Function `{func.__name__}` is not a node transition or does not have transitions defined."
        )
    return res
