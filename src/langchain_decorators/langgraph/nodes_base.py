from asyncio import iscoroutine
import asyncio

import functools
import inspect


from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    Optional,
    Tuple,
    Type,
    TypedDict,
    get_args,
    get_origin,
    overload,
)


from .node_decorator import is_graph_node, get_node_info

from .node_tool import (
    NodeTool,
    node_tool,
)

from .transitions import (
    get_conditional_transition_edges,
    is_node_transition,
    conditional_transition,
)

from langchain_core.runnables import (
    RunnableConfig,
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
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
    Checkpointer,
)

from pydantic import BaseModel, create_model
from typing import TypeVar


Node = StateNode | Type["LlmNodeBase"] | Tuple[str, StateNode | Type["LlmNodeBase"]]


class LlmNodeDescriptor(TypedDict):
    prompt_nodes: tuple[str, ...]
    node_to_func: dict[str, str]
    nodes: dict[str, str]
    fields: dict[str, Type[Any]]
    tools: dict[str, list[str] | None]
    transitions: dict[str, tuple[str]]


def _as_node_call(func, default: str, llm_node_class: Type["LlmNodeBase"]):
    """
    Decorator to provide a default value for a function if it returns None.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        node = llm_node_class(args[0])
        result = func(node, *args[1:], **kwargs)
        return result if result is not None else default

    return wrapper


class LlmNodeBase:

    __state__: Any
    __config__: RunnableConfig
    __description__: LlmNodeDescriptor
    messages: Annotated[list[AnyMessage], add_messages]

    @property
    def state(self) -> Any:
        """Get the current state of the node."""
        return self.__state__

    @property
    def config(self) -> RunnableConfig:
        """Get the configuration of the node."""
        return self.__config__

    def get_node_args(self, func_name: str):
        prompt_func = getattr(self.__class__, func_name)
        kwargs = {}
        if prompt_func and inspect.isfunction(prompt_func):
            params_signature = {
                k: v
                for k, v in inspect.signature(prompt_func).parameters.items()
                if k != "self"
            }
            for k, v in params_signature.items():
                if k in self:
                    kwargs[k] = self[k]
                if self.__config__.get("metadata") and k in self.__config__["metadata"]:
                    kwargs[k] = self.__config__["metadata"][k]
            if not kwargs and len(params_signature) <= 2:
                kwargs = {
                    k: v
                    for k, v in zip(params_signature, (self.__state__, self.__config__))
                }
        return kwargs

    @classmethod
    async def ainvoke(cls, state: Any, config: RunnableConfig):
        myself = cls(state, config)
        if node_name := config["metadata"].get("langgraph_node"):
            _description = cls.get_description()
            func_name = _description["node_to_func"].get(node_name, node_name)

            kwargs = myself.get_node_args(func_name)
            if node_name in _description["prompt_nodes"]:
                prompt = getattr(myself, node_name)
                res = await myself.acall_prompt(node_name, prompt, kwargs)
            else:
                res = getattr(myself, func_name)(**kwargs)
                if asyncio.iscoroutine(res):
                    res = await res
            return res

    @classmethod
    async def invoke(cls, state: Any, config: RunnableConfig):
        myself = cls(state, config)
        if node_name := config["metadata"].get("langgraph_node"):
            _description = cls.get_description()
            func_name = _description["node_to_func"].get(node_name, node_name)

            kwargs = myself.get_node_args(func_name)
            if node_name in _description["prompt_nodes"]:
                prompt = getattr(myself, node_name)
                res = myself.call_prompt(node_name, prompt, kwargs)
            else:
                res = getattr(myself, func_name)(**kwargs)
            return res

    def call_prompt(
        self, prompt_node: str, prompt_call: callable, kwargs: dict[str, Any]
    ) -> AnyMessage:
        with LlmChatSession(
            tools=self.get_tools(prompt_node), message_history=self.messages
        ) as session:
            prompt_response = prompt_call(**kwargs)
            session.last_llm_response.response_metadata["graph_node"] = prompt_node
            return self._handle_prompt_response(
                prompt_node, prompt_response, session.last_llm_response
            )

    async def acall_prompt(
        self, prompt_node: str, prompt_call: callable, kwargs: dict[str, Any]
    ) -> AnyMessage:
        with LlmChatSession(
            tools=self.get_tools(prompt_node), message_history=self.messages
        ).with_stream(lambda token: None) as session:
            prompt_response = prompt_call(**kwargs)
            if asyncio.iscoroutine(prompt_response):
                prompt_response = await prompt_response
            session.last_llm_response.response_metadata["graph_node"] = prompt_node
            return self._handle_prompt_response(
                prompt_node, prompt_response, session.last_llm_response
            )

    @classmethod
    def _get_previous_llm_node(cls, state: Any) -> str | None:
        if isinstance(state, dict):
            messages = state.get("messages") or []
        last_ai_message = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage)), None
        )
        if (
            last_ai_message
            and last_ai_message.response_metadata
            and "graph_node" in last_ai_message.response_metadata
        ):
            return last_ai_message.response_metadata["graph_node"]
        else:
            raise Exception(
                "Unable to determine previous LLM node form last message... if you've are modifying messages or overriding some of the native LLM call handling functions, you have to override _get_previous_llm_node too and implement logic to determine LLM node to return from tool calls"
            )

    def _handle_prompt_response(
        self, prompt_name: str, response: Any, ai_message: AIMessage
    ) -> dict:
        _description = self.get_description()
        node_info = _description["nodes"].get(prompt_name)
        if node_info and node_info.get("store_output_as"):
            updates = {node_info["store_output_as"]: response}
        else:
            updates = {}
        return {"messages": [ai_message], **updates}

    def get_tools(self, prompt_node: str):
        return self._get_tools_for_node(prompt_node)

    def _get_tools_for_node(self, prompt_node: str):
        tools = self.get_description()["tools"]
        if tools:
            return [
                getattr(self, tool_name)
                for tool_name, bind_to_prompts in tools.items()
                if not (bind_to_prompts and prompt_node not in bind_to_prompts)
            ]

    def __getitem__(self, key: str) -> Any:
        """
        Get the value of the key from the state.
        This is a helper method to access state attributes dynamically.
        """
        if isinstance(self.__state__, dict):
            return self.__state__.get(key, None)
        else:
            return getattr(self.__state__, key, None)

    def __contains__(self, key: str) -> bool:
        """
        Check if the key exists in the state.
        This is a helper method to check state attributes dynamically.
        """
        if key in self.get_description()["fields"]:
            return True
        if isinstance(self.__state__, dict):
            return key in self.__state__
        else:
            return hasattr(self.__state__, key)

    def __init__(self, state: Any, config: RunnableConfig = None):
        cls = type(self)
        fields = cls.get_description()["fields"]
        for field_name, field_type in fields.items():
            property_field = property(
                lambda _self, fname=field_name: (
                    getattr(_self.state, fname, None)
                    if not isinstance(_self.state, dict)
                    else _self.state.get(fname, None)
                )
            )
            setattr(self.__class__, field_name, property_field)
        self.__state__ = state  # must be here, after the property init
        self.__config__ = config

    def __setattr__(self, name, value):
        fields = type(self).get_description()["fields"]
        if getattr(self, "__state__", None) and name in fields:
            raise AttributeError(
                f"Cannot update '{self}.{name}' directly. To modify the state use Command or return dictionary with updates ... (read more in LangGraph documentation)."
            )
        else:
            super().__setattr__(name, value)

    @classmethod
    def get_input_schema(cls) -> type[dict[str, Any]]:
        """
        Get the schema class for the state of this LLM node.
        This is used to define the state type in the graph.
        """
        return TypedDict(f"{cls.__name__}State", cls.get_description()["fields"])

    @classmethod
    def _describe(cls) -> LlmNodeDescriptor:
        # Get all type-hinted fields from the inheritance chain up to LlmNodeBase
        fields = {}
        for base_cls in reversed(cls.__mro__):
            if not issubclass(base_cls, LlmNodeBase):
                continue
            if hasattr(base_cls, "__annotations__"):
                base_fields = {
                    k: v
                    for k, v in base_cls.__annotations__.items()
                    if not k.startswith("_")
                }
                fields.update(base_fields)

        # Get all members from the inheritance chain up to LlmNodeBase
        all_members = []
        for base_cls in reversed(cls.__mro__):
            if not issubclass(base_cls, LlmNodeBase):
                continue
            members = [
                (k, v)
                for k, v in inspect.getmembers(
                    base_cls,
                    predicate=lambda f: (
                        inspect.isfunction(f)
                        or inspect.iscoroutinefunction(f)
                        or isinstance(f, BaseTool)
                    ),
                )
                if not k.startswith("_")
            ]
            all_members.extend(members)

        # Remove duplicates while preserving order (most derived class first)
        seen = set()
        unique_members = []
        for name, func in all_members:
            if name not in seen:
                seen.add(name)
                unique_members.append((name, func))

        tools = dict(
            (name, (func.bind_to_prompt_nodes) if isinstance(func, NodeTool) else None)
            for name, func in unique_members
            if isinstance(func, BaseTool)
        )
        transitions = {
            name: get_conditional_transition_edges(func)
            for name, func in unique_members
            if is_node_transition(func)
        }
        nodes = {
            name: get_node_info(func)
            for name, func in unique_members
            if is_graph_node(func)
        }
        node_to_func = {v["name"]: k for k, v in nodes.items()}
        prompts = tuple(
            name
            for name, func in unique_members
            if is_llm_prompt(func) and func.__name__ in nodes
        )

        description = LlmNodeDescriptor(
            prompt_nodes=prompts,
            node_to_func=node_to_func,
            nodes=nodes,
            fields=fields,
            tools=tools,
            transitions=transitions,
        )

        cls.__description__ = description

        return description

    @classmethod
    def get_description(cls) -> LlmNodeDescriptor:
        """
        Get the description of the LLM node, including prompts, fields, tools, and transitions.
        """
        if not hasattr(cls, "__description__"):
            cls._describe()
        return cls.__description__

    @classmethod
    def compile(
        cls, state_type: Type[Any] = None, checkpointer: Checkpointer = None
    ) -> StateGraph:
        if state_type is None:
            state_type = cls.get_input_schema()

        subgraph = StateGraph(state_type)
        cls.add_to_graph(subgraph)
        return subgraph.compile(checkpointer=checkpointer, name=cls.__name__)

    @classmethod
    def add_to_graph(cls, graph: StateGraph, *args, **kwargs):
        """
        Add this LLM node to the provided graph.
        """
        if not isinstance(graph, StateGraph):
            raise TypeError("Provided graph must be an instance of StateGraph")

        description = cls.get_description()

        nodes = description["nodes"]
        _after_start_added = False

        for node_func_name, node_info in nodes.items():
            graph.add_node(node_info["name"], cls.ainvoke)

            if node_info["after"]:
                for after_node in node_info["after"]:
                    if after_node == "__start__":
                        _after_start_added = True
                    graph.add_edge(after_node, node_info["name"])

        tools_key = None
        if description["tools"]:
            tools_key = f"tools"

            graph.add_node(
                tools_key,
                BoundedToolNode(
                    name=tools_key,
                    node_class=cls,
                    tools=[
                        getattr(cls, tool_name) for tool_name in description["tools"]
                    ],
                ),
            )
            prompt_nodes_with_tools = set()
            for tool_bindings in description["tools"].values():
                if not tool_bindings:
                    prompt_nodes_with_tools = set(description["prompt_nodes"])
                else:
                    prompt_nodes_with_tools.update(tool_bindings)

            for prompt_node_key in prompt_nodes_with_tools:
                graph.add_conditional_edges(
                    prompt_node_key,
                    RunnableLambda(
                        tools_condition, name=f"{prompt_node_key}_tool_edge"
                    ),
                    {
                        # Translate the condition outputs to nodes in our graph
                        "tools": tools_key,
                        END: END,
                    },
                )
            graph.add_conditional_edges(
                tools_key,
                cls._get_previous_llm_node,
                {
                    # Translate the condition outputs to nodes in our graph
                    **{k: k for k in prompt_nodes_with_tools},
                    END: END,
                },
            )

        if description["transitions"]:
            for transition_func_name, transition_edges in description[
                "transitions"
            ].items():
                transition_func = getattr(cls, transition_func_name, None)

                for source, targets in transition_edges.items():
                    graph.add_conditional_edges(
                        source,
                        _as_node_call(transition_func, END, cls),
                        {
                            **{k: k for k in targets},
                            END: END,
                        },
                    )

        if not _after_start_added and len(nodes) == 1:
            for node_key in nodes.keys():
                graph.add_edge("__start__", node_key)
        return graph


class Router:
    __routes__: dict[callable, str] = None
    __default__: str
    __state__: Any

    def __getattr__(self, key: str) -> Any:
        if not key.startswith("__"):
            return self.__getkey__(key)
        else:
            return object.__getattribute__(self, key)

    def get(self, key, default=None):
        return self.__state__.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.__state__

    def __getkey__(self, key: str) -> Any:
        """
        Get the value of the key from the state.
        This is a helper method to access state attributes dynamically.
        """
        if isinstance(self.__state__, dict):
            return self.__state__.get(key, None)
        else:
            return getattr(self.__state__, key)

    def __init__(self, state: Any):
        """
        Initialize the RouterNode with the given state.
        """
        if self.__class__ == Router:
            raise TypeError(
                "RouterNode is an abstract class and cannot be instantiated directly."
            )
        self.__state__ = state

    @classmethod
    def invoke(cls, state: Any, config: RunnableConfig):
        """
        Invoke the router node with the given state and configuration.
        """
        if not cls.__routes__:
            raise ValueError("No routes defined for this router node")
        node_state = cls(state)
        for route_key, node_key in cls.__routes__.items():
            result = route_key(node_state)
            if result:
                return node_key
        return "__default__"

    @staticmethod
    def _declare(
        name: str,
        routes: dict[callable, Type[LlmNodeBase] | StateNode | str],
        default: Type[LlmNodeBase] | StateNode | str = END,
    ) -> Type["Router"]:
        """
        Declare the routes for this router node.
        """
        return type(name, (Router,), {"__routes__": routes, "__default__": default})

    @classmethod
    def as_runnable(cls):
        """
        Convert the router node into a runnable node.
        """
        return RunnableLambda(cls.invoke, name=cls.__name__)

    @classmethod
    def add_edges_to_graph(
        cls,
        graph: StateGraph,
        source: str,
        routes: dict[callable, str],
        default: Type[LlmNodeBase] | StateNode | str | None = END,
    ):
        """
        Add this router node to the provided graph.
        """
        if not routes:
            raise ValueError("No routes provided for conditional edges")

        if not isinstance(graph, StateGraph):
            raise TypeError("Provided graph must be an instance of StateGraph")

        RouterCls = cls._declare(
            f"RouterFrom_{source}",
            routes,
            default=default,
        )
        path_map = {
            **{k: k for k in routes.values()},
        }
        if default is not None:
            path_map["__default__"] = default

        graph.add_conditional_edges(
            source,
            RouterCls.as_runnable(),
            path_map=path_map,
        )

        return graph


class BoundedToolNode(ToolNode):
    node_class: Type[LlmNodeBase]

    def __init__(
        self,
        tools,
        *,
        name,
        node_class: Type[LlmNodeBase],
        tags=None,
        handle_tool_errors=True,
        messages_key="messages",
    ):
        super().__init__(
            tools,
            name=name,
            tags=tags,
            handle_tool_errors=handle_tool_errors,
            messages_key=messages_key,
        )
        self.node_class = node_class

    def _inject_self_node(self, tool_call, state):
        tool = self.tools_by_name[tool_call["name"]]
        if getattr(tool, "bound_func", None) == True:

            tool_call["args"] = {
                "__self__": self.node_class(state),
                **tool_call["args"],
            }

    def inject_tool_args(self, tool_call: ToolCall, input: dict | BaseModel, store):
        """
        Inject the tool arguments into the tool call.
        This method is called by the base class to prepare the tool call.
        """

        enriched_tool_call = super().inject_tool_args(tool_call, input, store)
        self._inject_self_node(enriched_tool_call, input)
        return enriched_tool_call
