import asyncio
from contextvars import ContextVar
import json
from typing import TYPE_CHECKING, Callable, Coroutine, Literal, Optional, Union
from langchain.tools.base import BaseTool


from langchain.schema.messages import ToolMessage, AIMessage, HumanMessage, BaseMessage
from langchain.prompts.chat import MessagesPlaceholder
from openai import BaseModel

if TYPE_CHECKING:
    from .llm_tool_use import ToolsProvider, ToolCall  # Avoid circular import issues


class LlmChatSession:
    _context = ContextVar("llm_chat_session", default=None)

    def __init__(
        self,
        tools: Union[list[Union[Callable, BaseTool]], "ToolsProvider"] = None,
        message_history: list = None,
        copy_parent_context: bool = False,
        suppress_output_parser: bool = False,
        on_stream_token: Optional[
            Callable[[str], Union[None, Coroutine[None, None, None]]]
        ] = None,
        context: dict = None,
        _messages_memory_arg_name: str = "messages",
    ):
        """Initialize the LlmChatSession with a list of tools or a ToolsProvider."""
        from .llm_tool_use import ToolsProvider

        if isinstance(tools, ToolsProvider):
            self.tools_provider = tools
        elif tools:
            self.tools_provider = ToolsProvider(tools)
        else:
            self.tools_provider = None

        self.messages_memory_arg_name = _messages_memory_arg_name
        self.parent_context: LlmChatSession = self._context.get()
        self.copy_parent_context = copy_parent_context
        if not message_history and self.parent_context and self.copy_parent_context:
            self.message_history = list(self.parent_context.message_history)
            self.context = {
                **(context or {}),
                **(self.parent_context.context),
            }
        else:
            self.message_history: list[BaseMessage] = (
                message_history if message_history is not None else []
            )
            self.context = context or {}

        if self.parent_context and self.parent_context.tools_provider and not tools:
            self.tools_provider = self.parent_context.tools_provider

        self.last_llm_response: AIMessage = None
        self._last_response_tool_calls: Optional[list["ToolCall"]] = None
        self.suppressed_output_parser = suppress_output_parser
        self.suppressed_structured_output = False
        self.on_stream_token = on_stream_token
        self._streaming_context = None
        self._hash_set = set(((m.type, m.content) for m in self.message_history))

    def get_prompt_context(self) -> dict:
        """Get the current context of the session."""
        messages = self.message_history
        return {**self.context, self.messages_memory_arg_name: messages}

    def with_stream(
        self,
        on_stream_token: Callable[[str], Union[None, Coroutine]],
    ):
        """Set the streaming callback for the session."""
        self.on_stream_token = on_stream_token

        return self

    def suppress_structured_output(self, suppress: bool = True):
        """Suppress the output parser for the session."""
        self.suppressed_output_parser = suppress
        self.suppressed_structured_output = suppress
        return self

    @classmethod
    def get_current_session(cls) -> "LlmChatSession":
        """Get the current ToolsProvider context."""
        return cls._context.get()

    def __enter__(self):
        """Enter the LlmChatSession context."""
        self._context.set(self)
        if self.on_stream_token:
            from .streaming_context import StreamingContext

            self._streaming_context = StreamingContext(self.on_stream_token).__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the LlmChatSession context."""
        self._context.set(self.parent_context)
        if self._streaming_context:
            self._streaming_context.__exit__(exc_type, exc_value, traceback)
            self._streaming_context = None

    @property
    def last_response_tool_calls(self) -> Optional[list["ToolCall"]]:
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

    def add_message(
        self,
        message: Union[BaseMessage, str],
        message_type: Literal["ai", "user", "tool"] = None,
        ignore_duplicates: bool = False,
    ):
        """Add a message to the session."""
        if isinstance(message, str):
            if message_type == "ai":
                message = AIMessage(content=message)
            elif message_type == "human" or message_type == "user":
                message = HumanMessage(content=message)
            elif message_type == "tool":
                message = ToolMessage(content=message)
            else:
                raise ValueError("Invalid message type. Use 'ai', 'human', or 'tool'.")
        if (message.type, message.content) in self._hash_set and ignore_duplicates:
            return
        self._hash_set.add((message.type, message.content))
        self.message_history.append(message)
        if isinstance(message, AIMessage):
            self.last_llm_response = message
            if message.tool_calls and self.tools_provider:
                self._last_response_tool_calls = []
                from .llm_tool_use import ToolCall

                for tool_call_dict in self.last_llm_response.tool_calls:

                    func = self.tools_provider.get_tool_by_name(
                        tool_call_dict["name"], raise_errors=False
                    )
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
