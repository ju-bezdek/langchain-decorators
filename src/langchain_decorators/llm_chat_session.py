import asyncio
from contextvars import ContextVar
import json
from typing import TYPE_CHECKING, Callable, Literal, Optional, Union
from langchain.tools.base import BaseTool


from langchain.schema.messages import ToolMessage, AIMessage
from openai import BaseModel

if TYPE_CHECKING:
    from .llm_tool_use import ToolsProvider, ToolCall  # Avoid circular import issues

class LlmChatSession:
    _context = ContextVar("llm_chat_session", default=None)

    def __init__(
        self,
        tools: Union[list[Union[Callable, BaseTool]], "ToolsProvider"],
        message_history: list = None,
        copy_parent_context: bool = False,
        suppress_output_parser: bool = False,
    ):
        """Initialize the LlmChatSession with a list of tools or a ToolsProvider."""
        from .llm_tool_use import ToolsProvider

        if isinstance(tools, ToolsProvider):
            self.tools_provider = tools
        elif tools:
            self.tools_provider = ToolsProvider(tools)
        else:
            self.tools_provider = None
        self.parent_context: LlmChatSession = self._context.get()
        if not message_history and self.parent_context and self._copy_messages_from_parent_context:
            self.message_history = list(self.parent_context.message_history)
        else:
            self.message_history = message_history or []

        if self.parent_context and self.parent_context.tools_provider and not tools:
            self.tools_provider = self.parent_context.tools_provider

        self._copy_messages_from_parent_context = copy_parent_context
        self.last_llm_response:AIMessage = None
        self._last_response_tool_calls: Optional[list["ToolCall"]] = None
        self.suppress_output_parser = suppress_output_parser

    @classmethod
    def get_current_session(cls) -> "LlmChatSession":
        """Get the current ToolsProvider context."""
        return cls._context.get()

    def __enter__(self):
        """Enter the LlmChatSession context."""
        self._context.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the LlmChatSession context."""
        self._context.set(self.parent_context)

    @property
    def last_response_tool_calls(self)->Optional[list["ToolCall"]]:
        """Get the tool calls from the last LLM response."""
        if not self.last_llm_response:
            return None

        return self._last_response_tool_calls or []

    def is_calling_tool(self, tool: Union[str, Callable]) -> bool:
        """Check if the last LLM response is calling a specific tool."""
        if not self.last_llm_response:
            return False
        if not self.last_response_tool_calls:
            return False
        if isinstance(tool, str):
            tool_name = tool
        elif isinstance(tool, BaseTool):
            tool_name = tool.name
        elif isinstance(tool, Callable) and hasattr(tool, "function_name"):
            tool_name = tool.function_name
        else:
            tool_name = tool.__name__ if hasattr(tool, "__name__") else str(tool)

        return any(
            tool_call.name == tool_name for tool_call in self.last_response_tool_calls
        )

    async def aexecute_tool_calls(
        self,
        error_handling: Literal["fail_safe", "fail_fast", "custom"] = "fail_safe",
        custom_error_handler: Optional[Callable[["ToolCall", Exception], str]] = None,
    ):
        """Execute the tool calls from the last LLM response."""
        if not self.last_response_tool_calls:
            return

        async def tool_exec_wrapper(
            tool_call: "ToolCall", session: "LlmChatSession" = self
        ):
            """Wrapper to execute tool calls with error handling."""
            with session:
                try:
                    return await tool_call.ainvoke()
                except Exception as e:
                    tool_call._result_original_value = e
                    error_handled = e
                    if custom_error_handler:
                        error_handled = custom_error_handler(tool_call, e)
                    tool_call.set_error_result(error_handled)

                    if error_handling == "custom" and error_handled == e:
                        raise e
                    elif error_handling == "fail_fast":
                        raise e
                    else:
                        return error_handled

        tasks = []
        for tool_call in self.last_response_tool_calls:
            tasks.append(
                asyncio.create_task(tool_exec_wrapper(tool_call, session=self))
            )
        return await asyncio.gather(
            *tasks, return_exceptions=True if error_handling == "fail_safe" else False
        )

    def execute_tool_calls(
        self,
        error_handling: Literal["fail_safe", "fail_fast", "custom"] = "fail_safe",
        custom_error_handler: Optional[Callable[["ToolCall", Exception], str]] = None,
    ):
        """Execute the tool calls from the last LLM response."""
        if not self.last_response_tool_calls:
            return

        results = []
        for tool_call in self.last_response_tool_calls:
            try:
                results.append(tool_call.invoke())
            except Exception as e:
                tool_call._result_original_value = e
                error_handled = e
                if custom_error_handler:
                    error_handled = custom_error_handler(tool_call, e)
                tool_call.set_error_result(error_handled)

                if error_handling == "custom" and error_handled == e:
                    raise e
                elif error_handling == "fail_fast":
                    raise e
                else:
                    results.append(error_handled)
        return results

    def add_message(self, message: AIMessage):
        """Add a message to the session."""
        self.message_history.append(message)
        if isinstance(message, AIMessage):
            self.last_llm_response = message
            if message.tool_calls:
                self._last_response_tool_calls = []
                from .llm_tool_use import ToolCall
                for tool_call_dict in  self.last_llm_response.tool_calls:
                    func = self.tools_provider.get_tool_by_name(tool_call_dict["name"],raise_errors=False)
                    self._last_response_tool_calls.append(
                        ToolCall(
                            name=tool_call_dict["name"],
                            args=tool_call_dict["args"],
                            function=func if isinstance(func, Callable) else None,
                            id=tool_call_dict.get("id"),
                            metadata={
                                k: v
                                for k, v in tool_call_dict.items()
                                if k not in ["name", "arguments", "id"]
                            },
                        )
                    )
            else:
                self._last_response_tool_calls = None
