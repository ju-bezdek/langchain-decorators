from functools import wraps
from typing import TypedDict


class NodeInfo(TypedDict):
    name: str
    after: list[str] | None
    store_output_as: str | None


class NodeDecorator:  # renamed from SubnodeDecorator

    def after(self, node_name_or_names: str | list[str]):
        def decorator(func):
            self._decorate(func, after=node_name_or_names)
            return func

        return decorator

    def __call__(
        self,
        func=None,
        *args,
        name: str = None,
        after: str = None,
        store_output_as: str = None,
    ):
        if args:
            raise ValueError("Positional arguments are not supported")
        # Case 1: @node without parentheses
        if callable(func):
            return self._decorate(func, after=after, store_output_as=store_output_as)
        # Case 2: @node(name)
        elif func:
            # func is defined but not as a callable
            raise ValueError("Positional arguments are not supported")
        else:
            return self._create_with_name(
                name, after=after, store_output_as=store_output_as
            )

    def _create_with_name(self, name, after: str = None, store_output_as: str = None):
        def decorator(func):
            return self._decorate(
                func, name, after=after, store_output_as=store_output_as
            )

        return decorator

    def _create_after_decorator(self, name):  # renamed from _create_on_decorator
        def decorator(func):
            return self._decorate(func, name)

        return decorator

    def _decorate(self, func, name=None, after=None, store_output_as: str = None):
        func.__is_node__ = True

        func.__node_name__ = name or func.__name__
        func.__node_after__ = [after] if isinstance(after, str) else after
        func.__store_output_as__ = store_output_as
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
        "store_output_as": func.__store_output_as__,
    }
