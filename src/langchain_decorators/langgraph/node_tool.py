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
    Union,
    Optional,
    get_type_hints,
    get_origin,
    get_args,
    Literal,
)
from langchain_decorators.function_decorator import build_func_schema
from langgraph.types import interrupt
from langgraph.prebuilt.interrupt import (
    HumanInterruptConfig,
    HumanInterrupt,
    ActionRequest,
)


class NodeTool(StructuredTool):
    bound_func: bool
    require_confirmation: bool
    bind_to_prompt_nodes: list[str] | None

    def __init__(
        self,
        func,
        name=None,
        description=None,
        args_schema=None,
        node_class: Type = None,
        require_confirmation: bool = False,
        bind_to_prompt_nodes: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(
            func=func if not asyncio.iscoroutinefunction(func) else None,
            coroutine=func if asyncio.iscoroutinefunction(func) else None,
            name=name,
            description=description,
            args_schema=args_schema,
            node_class=node_class,
            require_confirmation=require_confirmation,
            bound_func="." in repr(func),
            bind_to_prompt_nodes=bind_to_prompt_nodes,
            **kwargs,
        )

    def maybe_ask_for_approval(self, input: dict) -> bool:
        if self.require_confirmation:
            is_approved = interrupt(
                HumanInterrupt(
                    action_request=ActionRequest(
                        action=self.name,
                        args={
                            k: v
                            for k, v in (input.get("args") or {}).items()
                            if k != "__self__"
                        },
                    ),
                    config=HumanInterruptConfig(
                        allow_ignore=False,
                        allow_respond=True,
                        allow_edit=False,
                        allow_accept=True,
                    ),
                    description=f"Please approve the action to execute tool '{self.name}' with arguments {input.get('args')}",
                )
            )
            if not is_approved == True:
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                tool_name=self.name,
                                tool_call_id=input.get("id"),
                                content=f"Action not approved",
                            ),
                        ]
                    },
                    goto="__end__",
                )
        return True

    def invoke(self, input: dict, *args, **kwargs):
        if approval_result := self.maybe_ask_for_approval(input) != True:
            return approval_result

        res = super().invoke(input, *args, **kwargs)
        return self.postprocess(res, input)

    async def ainvoke(self, input: dict, *args, **kwargs):
        if approval_result := self.maybe_ask_for_approval(input) != True:
            return approval_result
        res = await super().ainvoke(input, *args, **kwargs)
        return self.postprocess(res, input)

    def postprocess(self, result: Any, input: dict) -> Any:
        if isinstance(result, Command):
            if isinstance(result.update, dict):
                if not result.update.get("messages"):
                    result.update["messages"] = [
                        ToolMessage(
                            tool_name=self.name,
                            tool_call_id=input.get("id"),
                            content=f"Done",
                        )
                    ]

        return result

    def _to_args_and_kwargs(
        self, tool_input: Union[str, dict], tool_call_id: Optional[str]
    ) -> tuple[tuple, dict]:
        """
        Override native _parse_input to handle bound methods.
        """
        _self = tool_input.pop(
            "__self__", None
        )  # Ensure __self__ is not passed to the function
        args, kwargs = super()._to_args_and_kwargs(tool_input, tool_call_id)
        if _self is not None:
            args = (_self,) + args
        return args, kwargs

    # async def _arun(self, *args, **kwargs):
    #     """
    #     Run the tool with the given arguments.
    #     This method is called by the base class to execute the tool.
    #     """
    #     _self = kwargs.pop(
    #         "__self__", None
    #     )  # Ensure __self__ is not passed to the function
    #     if _self is not None:
    #         args = (_self,) + args
    #     if self.coroutine:
    #         return await self.coroutine(*args, **kwargs)
    #     else:
    #         return self.func(*args, **kwargs)


def node_tool(
    func=None,
    *,
    name=None,
    description=None,
    bind_to_prompt_nodes: list[str] | None = None,
    args_schema=None,
    require_confirmation=False,
    **kwargs,
):
    """
    Decorator that marks a function as a node transition tool.

    Features:
        - Extracts possible transitions from the return type, to annotate the graph.
        - Handles bound arguments for the tool.
    """

    def decorator(
        func,
        name=name,
        func_info=None,
        description=description,
        bind_to_prompt_nodes=bind_to_prompt_nodes,
        **kwargs,
    ):

        type_hints = get_type_hints(func)
        return_type = type_hints.get("return", None)

        # Extract transitions from Literal return type if present
        transitions = None
        if return_type and get_origin(return_type) is Command:
            transitions_literal = get_args(return_type)[0]
            if transitions_literal and get_origin(transitions_literal) is Literal:
                transitions = list(get_args(transitions_literal))
        if not func_info:
            func_info = build_func_schema(
                func=func,
                function_name=name,
                func_description=description,
                as_pydantic_model=True,
            )

        wrapped_tool = NodeTool(
            func=func,
            name=name or func_info.__name__,
            description=description or func_info.__doc__ or "",
            args_schema=func_info,
            require_confirmation=require_confirmation,
            bind_to_prompt_nodes=bind_to_prompt_nodes,
        )

        # Add metadata attributes
        if transitions:
            wrapped_tool.__is_transition__ = True
            # TODO: remove hardcoded tools key!
            wrapped_tool.__transitions__ = {"tools": transitions}

        return wrapped_tool

    if func is not None:
        if not callable(func):
            raise TypeError("node_tool decorator must be applied to a function.")
        return decorator(func, func_info=args_schema, description=description, **kwargs)
    else:
        return decorator
