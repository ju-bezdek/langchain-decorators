from functools import wraps
from typing import TypedDict


class NodeInfo(TypedDict):
    name: str
    after: list[str] | None


class NodeDecorator:  # renamed from SubnodeDecorator

    def after(self, node_name_or_names: str | list[str]):
        def decorator(func):
            self._decorate(func, after=node_name_or_names)
            return func

        return decorator

    def __call__(self, func, *, name: str = None):
        # Case 1: @node without parentheses
        if callable(func):
            return self._decorate(func)
        # Case 2: @node(name)
        else:
            return self._create_with_name(name)

    def _create_with_name(self, name):
        def decorator(func):
            return self._decorate(func, name)

        return decorator

    def _create_after_decorator(self, name):  # renamed from _create_on_decorator
        def decorator(func):
            return self._decorate(func, name)

        return decorator

    def _decorate(self, func, name=None, after=None):
        func.__is_node__ = True

        func.__node_name__ = name or func.__name__
        func.__node_after__ = [after] if isinstance(after, str) else after
        return func


# Create the decorator instance (renamed)
node = NodeDecorator()


def is_graph_node(func):
    """Check if the function is decorated as a node."""
    return getattr(func, "__is_node__", False)


def get_node_info(func) -> NodeInfo:
    """Get the node information of a decorated function."""
    if not is_graph_node(func):
        return None
    return {
        "name": func.__node_name__,
        "after": func.__node_after__,
    }
