from asyncio import iscoroutine

from typing import (
    Annotated,
    Any,
    Callable,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypedDict,
)
from langchain_core.runnables import (
    Runnable,
)
from langchain_core.tools import BaseTool, tool
from langchain_decorators import llm_prompt, LlmChatSession, is_llm_prompt
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.graph._node import StateNodeSpec, StateNode
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_core.messages import ToolCall, AIMessage
from langgraph.types import (
    All,
    CachePolicy,
    Checkpointer,
    Command,
    RetryPolicy,
    Send,
)

from pydantic import BaseModel, create_model
from typing import TypeVar
from .nodes_base import LlmNodeBase, Node, Router

TState = TypeVar("TState")


class SequentialGraphMeta(type):
    def __getitem__(cls, state_schema) -> Type["SequentialGraph"]:
        # Create a new class that remembers the state type
        class TypedGraph(cls):
            graph_state_schema = state_schema

        return TypedGraph


class _GraphBuilder:
    # state_graph: StateGraph
    graph_state_schema: Type[Any]
    llm_nodes: list[LlmNodeBase]

    def __init__(
        self, state_graph: StateGraph = None, graph_state_schema: Type[Any] = None
    ):
        self.graph_state_schema = graph_state_schema
        self.llm_nodes: list[Type] = list()
        if not graph_state_schema:
            raise Exception(
                f"Graph state schema is required ... pls define schema using `{self.__class__.__name__}[YourGraphSchema]`.start()... syntax"
            )
        if state_graph is None:
            self.state_graph = StateGraph(state_schema=graph_state_schema)
        else:
            self.state_graph = state_graph

    def _path_to_key_and_action(
        self, action: Node
    ) -> tuple[str, StateNode | Runnable | callable]:

        if isinstance(action, tuple):
            key, action = action
            if isinstance(action, type) and issubclass(action, LlmNodeBase):
                self.llm_nodes.append(action)
                action = action.compile()
            if not isinstance(key, str):
                raise ValueError(
                    f"When providing a tuple as node spec, the first element must be a string key, got {type(key)}"
                )
            elif not (isinstance(action, Runnable) or callable(action)):
                raise ValueError(
                    f"When providing a tuple as node spec, the second element must be subclass of LlmNodeBase or callable, got {type(action)}"
                )
            return key, action
        else:

            if isinstance(action, str):
                # we can use than to create edge, but we cannot add it to the graph...
                # we assume this has been done by providing the graph with this node
                return action, None
            if isinstance(action, type) and issubclass(action, LlmNodeBase):
                key = action.__name__
                self.llm_nodes.append(action)
                action = action.compile()
            elif hasattr(action, "__name__"):
                key = action.__name__
            elif isinstance(action, Runnable):
                key = action.get_name()
            return key, action

    def _then(
        self,
        from_node_key: str | list[str],
        next_node_or_nodes: Node | Tuple[Node, ...],
    ):
        """
        Add a node spec to the graph.
        """

        if isinstance(next_node_or_nodes, tuple):
            if isinstance(next_node_or_nodes[0], str):
                paths = [next_node_or_nodes]
            else:
                paths = next_node_or_nodes
        else:
            paths = [next_node_or_nodes]
        next_keys = []
        for path in paths:
            key, action = self._path_to_key_and_action(path)
            if action is None:
                # action is None if the path was just a string-key... we assume node been added to the graph already
                next_keys.append(key)
                continue

            if key in self.state_graph.nodes:
                raise ValueError(
                    f"Node with key '{key}' already exists in the graph ... pls assign unique key to the node, if this was intentional"
                )

            self.state_graph.add_node(
                key, action  # , input_schema=self._get_node_input_schema(action)
            )
            next_keys.append(key)

        if isinstance(from_node_key, list):
            if len(next_keys) > 1:
                raise ValueError(
                    "Cannot add parallel paths after a parallel paths in previous step. If you need multiple steps in parallel paths, add the whole chain as a single node using `then((SubGraph, ...))` syntax."
                )
            for last_key in from_node_key:
                self.state_graph.add_edge(last_key, next_keys[0])
        else:
            for key in next_keys:
                self.state_graph.add_edge(from_node_key, key)

        return SequentialGraphBuilderCursor(self, next_keys)

    def _then_route(
        self,
        from_node_key: str | list[str],
        routes: dict[callable, Type[LlmNodeBase] | StateNode | str],
        default: Type[LlmNodeBase] | StateNode | str | None = END,
    ):
        """
        Add router as next step in the graph to direct state to different nodes based on on conditional logic.

        Args:
            routes: A mapping of route keys to LLM node classes or node specs.
            default: The default route if no conditions match (defaults to END).
        """

        if from_node_key is None:
            raise ValueError("No previous node to route from ... graph is empty")
        if isinstance(from_node_key, list):
            if len(from_node_key) > 1:
                raise ValueError(
                    "Cannot add router after parallel paths. Use `then` to add nodes after parallel paths."
                )
            from_node_key = from_node_key[0]
        routes_to_keys = {}
        for condition, node_key in routes.items():
            key, action = self._path_to_key_and_action(node_key)
            routes_to_keys[condition] = key

        Router.add_edges_to_graph(
            self.state_graph,
            from_node_key,
            routes_to_keys,
            default,
        )

        next_keys = []
        for path in routes.values():
            key, action = self._path_to_key_and_action(path)
            if not action:
                action = key

            next_keys.append(key)
            if isinstance(action, str):
                # we can use than to create edge, but we cannot add it to the graph...
                # we assume this has been done by providing the graph with this node
                continue
            self.state_graph.add_node(
                key, action  # , input_schema=self._get_node_input_schema(action)
            )

        return SequentialGraphBuilderCursor(self, next_keys)

    def compile(self, **kwargs):

        return self.state_graph.compile(**kwargs)


class SequentialGraph(_GraphBuilder, metaclass=SequentialGraphMeta):

    @classmethod
    def start(cls) -> "SequentialGraphBuilderCursor":
        """
        Start a new sequential graph builder with the provided graph.
        If no graph is provided, a new StateGraph will be created.
        """

        state_type = None
        if hasattr(cls, "graph_state_schema"):
            # Try to determine the schema from generic arguments of the class
            state_type = getattr(cls, "graph_state_schema")
        else:
            state_type = None

        # If no graph provided, create one with the state type

        _graph = cls(graph_state_schema=state_type)

        return SequentialGraphBuilderCursor(_graph, "__start__")


class StagedGraphStateMin(TypedDict):
    _current_stage: str
    messages: Annotated[list[AnyMessage], add_messages]


class StagedGraphMeta(type):

    @classmethod
    def _get_graph_schema(cls, state_type) -> Type["StagedGraphStateMin"]:
        if (
            issubclass(state_type, dict)
            and getattr(cls, "__annotations__", None) is not None
        ):
            graph_state_schema = TypedDict(
                f"{state_type.__name__}StagedGraphState",
                {
                    **StagedGraphStateMin.__annotations__,
                    **getattr(state_type, "__annotations__", {}),
                },
            )
        elif issubclass(state_type, BaseModel):
            graph_state_schema = create_model(
                f"{state_type.__name__}StagedGraphState",
                __base__=state_type,
                **StagedGraphStateMin.__annotations__,
            )
        else:
            raise TypeError(f"Unsupported state type: {state_type}")

        return graph_state_schema

    def __getitem__(cls, state_type) -> Type["StagedGraph"]:

        # Create a new class that remembers the state type
        class TypedGraph(cls):
            graph_state_schema = StagedGraphMeta._get_graph_schema(state_type)

        return TypedGraph


class StagedGraph(_GraphBuilder, metaclass=StagedGraphMeta):
    __stages__: list[str] = None
    CMD_NEXT = Command(
        update={"stage": "improvements"}, goto="__next__", graph=Command.PARENT
    )

    @classmethod
    def cmd_next_with_update(cls, update: dict):
        return Command(
            update={**cls.CMD_NEXT.update, **update},
            goto=cls.CMD_NEXT.goto,
            graph=cls.CMD_NEXT.graph,
        )

    @classmethod
    def _resume(cls, state: dict, config):
        current_stage = state.get("_current_stage") or cls.__stages__[0]
        return Command(update={"_current_stage": current_stage}, goto=current_stage)

    @classmethod
    def _next(cls, state: dict, config):
        current_stage = state.get("_current_stage") or cls.__stages__[0]

        current_stage_index = cls.__stages__.index(current_stage)
        if current_stage_index + 1 < len(cls.__stages__):
            next_stage = cls.__stages__[current_stage_index + 1]
            return Command(update={"_current_stage": next_stage})
        else:

            return Command(goto="__end__")

    @classmethod
    def _compile(cls, cursor: "StagedGraphBuilderCursor", **kwargs):
        stages_list = list(cursor.stages)

        command_with_first_stage_annotated = Command[Literal[*tuple(stages_list[:1])]]

        class StagedGraphCompiled(StagedGraph):
            __stages__ = stages_list

            @classmethod
            def next(cls, state, config) -> Command:

                return cls._next(state, config)

            @classmethod
            def resume(cls, state, config):

                return cls._resume(state, config)

        StagedGraphCompiled.resume.__annotations__["return"] = (
            command_with_first_stage_annotated
        )

        cursor.graph.state_graph.add_node(
            "__next__", StagedGraphCompiled.next  # , destinations=("__continue__",)
        )

        stages_point = cls._then(
            cursor.graph,
            "__start__",
            ("__continue__", StagedGraphCompiled.resume),
        )
        # cursor.graph.state_graph.add_edge("__start__", "__next__")
        cursor.graph.state_graph.add_edge("__next__", "__continue__")

        for stage_name, stage_action in cursor.stages.items():

            stages_point = stages_point.then_route(
                {lambda never: None: (stage_name, stage_action)}
            )

        return stages_point.compile(**kwargs)

    @classmethod
    def start(cls):
        """
        Start a new sequential graph builder with the provided graph.
        If no graph is provided, a new StateGraph will be created.
        """
        state_schema = None
        if hasattr(cls, "graph_state_schema"):
            # Try to determine the schema from generic arguments of the class
            state_schema = getattr(cls, "graph_state_schema")
        else:
            state_schema = None

        # If no graph provided, create one with the state type

        _graph = cls(graph_state_schema=state_schema)

        return StagedGraphBuilderCursor(_graph)


class SequentialGraphBuilderCursor:

    def __init__(
        self, graph: _GraphBuilder = None, last_node_key: str | list[str] = None
    ):
        self.graph = graph
        self.pending_route = None
        self.last_node_key = last_node_key or [*graph.state_graph.nodes][-1]

    def then(
        self,
        next: Node | Tuple[str, Node] | List[Node],
        after: str | list[str] = None,
    ):
        """
        Add a node spec to the graph.
        """

        self._flush_pending(next)
        return self.graph._then(after or self.last_node_key, next)

    def then_route(
        self,
        routes: dict[callable, Type[LlmNodeBase] | StateNode | str],
        default: Type[LlmNodeBase] | StateNode | str = END,
        after: str | list[str] = None,
    ):
        """
        Add router as next step in the graph to direct state to different nodes based on on conditional logic.

        Args:
            routes: A mapping of route keys to LLM node classes or node specs.
            default: The default route if no conditions match (defaults to END).
            exclusive: If True, only the first matching route will be executed, otherwise all matching routes will be executed.
        """
        if self.pending_route:
            if self.pending_route.get("__default__") == "__next__":
                raise Exception(
                    "Can't follow conditional edges with another conditional routes... with '__next__' pseudo key "
                )
            else:
                self._flush_pending(
                    END
                )  # next_key does not matter here... since there is not __next__ key

        next_cursor = SequentialGraphBuilderCursor(
            self.graph, after or self.last_node_key
        )
        next_cursor.pending_route = {**routes, "__default__": default}

        return next_cursor

    def then_if(
        self,
        condition: Callable[[TState], bool],
        stage_action: Node,
        else_key="__next__",
    ) -> "SequentialGraphBuilderCursor":
        next_cursor = SequentialGraphBuilderCursor(self.graph, self.last_node_key)
        next_cursor.pending_route = {condition: stage_action, "__default__": else_key}
        return next_cursor

    def _update_pending_route(self, paths: dict = None):
        if not self.pending_route:
            self.pending_route = paths
        else:
            self.pending_route.update(paths)

    def _flush_pending(self, next_key: str):
        if self.pending_route:
            next_key, _ = self.graph._path_to_key_and_action(next_key)
            default = self.pending_route.pop("__default__", None)
            if default == "__next__":
                default = next_key
            res = self.graph._then_route(
                self.last_node_key, self.pending_route, default=default
            )
            self.pending_route = None
            self.last_node_key = res.last_node_key

    def compile(self, **kwargs):
        self._flush_pending(END)
        self.graph._then(self.last_node_key, END)
        return self.graph.compile(**kwargs)


class StagedGraphBuilderCursor:
    stages: dict[str, Node | tuple[callable, Node] | dict[callable, Node]]
    graph: StagedGraph

    def __init__(self, graph: StagedGraph = None):
        self.graph = graph
        self.stages = {}

    def then(self, stage_action: Node | tuple[str, Node]) -> "StagedGraphBuilderCursor":
        key, action = self.graph._path_to_key_and_action(stage_action)
        self.stages[key] = action
        return self

    def then_if(
        self, condition: Callable[[TState], bool], stage_action: Node
    ) -> "StagedGraphBuilderCursor":
        key, action = self.graph._path_to_key_and_action(stage_action)
        self.stages[key] = {condition: action}
        return self

    def compile(self, **kwargs):
        return self.graph._compile(self, **kwargs)
