from typing import Any, Callable, Coroutine, Dict, List, Optional, Type, Union, cast
import asyncio
import logging
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Union, List
from langchain.schema.messages import AIMessage, ToolMessage

import json

from .llm_chat_session import LlmChatSession
from pydantic import PrivateAttr, BaseModel
from .common import deprecated
from langchain.tools.convert_to_openai import format_tool_to_openai_function


from langchain.tools.base import BaseTool

from .function_decorator import get_function_schema


class ToolsProvider:

    def __init__(
        self,
        tools: Union[
            List[Union[Callable, BaseTool]], Dict[str, Union[Callable, BaseTool]]
        ] = None,
        **kwargs,
    ) -> None:
        """Initialize ToolsProvider with list of tools of dictionary where key is the unique function name alias"""

        if "functions" in kwargs:
            # Backwards compatibility
            tools = kwargs["functions"]
        self.tools = []
        self.aliases = []
        self.tool_schemas = []
        self.tool_name_map = {}
        if not (isinstance(tools, dict) or isinstance(tools, list)):
            raise ValueError(
                "ToolsProvider must be initialized with list of functions or dictionary where key is the unique function name alias"
            )

        for i, f in enumerate(tools):

            if isinstance(f, str):
                function_alias = f
                f = tools[f]
            else:
                function_alias = None

            self.add_function(f, function_alias)

    def add_tool(self, tool: Union[Callable, BaseTool], alias: str = None):
        """Add function to ToolsProvider. If alias is provided, it will be used as function name in LLM"""
        self.tools.append(tool)
        self.aliases.append(alias)
        if isinstance(tool, BaseTool):
            self.tool_schemas.append(format_tool_to_openai_function(tool))
            f_name = alias or tool.name
        elif callable(tool) and hasattr(tool, "get_function_schema"):
            if hasattr(tool, "function_name"):
                f_name = alias or tool.function_name
            else:
                raise Exception(
                    f"Function {tool} does not have function_name attribute. All functions must be marked with @llm_function decorator"
                )
            self.tool_schemas.append(
                lambda kwargs, f=tool: get_function_schema(f, kwargs)
            )
        else:
            raise ValueError(
                f"Invalid item value in functions. Only Tools or functions decorated with @llm_function are allowed. Got: {tool}"
            )
        if f_name in self.tool_name_map:
            if alias:
                raise ValueError(f"Invalid alias - duplicate function name: {f_name}.")
            else:
                raise ValueError(
                    f"Duplicate function name: {f_name}. Use unique function names, or use ToolsProvider and assign a unique alias to each function."
                )
        self.tool_name_map[f_name] = tool

    @deprecated("Use add_tool instead. This method will be removed in future versions.")
    def add_function(self, function: Union[Callable, BaseTool], alias: str = None):
        """Add function to ToolsProvider. If alias is provided, it will be used as function name in LLM"""
        self.add_tool(function, alias)

    def __contains__(self, function):
        return function in self.tools

    def get_function_schemas(self, inputs, _index: int = None):
        if self.tool_schemas:
            _f_schemas = []
            for i, (alias, f_schema_builder) in enumerate(
                zip(self.aliases, self.tool_schemas)
            ):
                if _index is not None and i != _index:
                    continue

                if callable(f_schema_builder):
                    _f_schema = f_schema_builder(inputs)
                else:
                    _f_schema = f_schema_builder

                if alias:
                    _f_schema["name"] = alias

                _f_schemas.append(_f_schema)

            return _f_schemas
        else:
            None

    def get_function_schema(self, function: Union[str, Callable], inputs: dict):
        index = None
        if isinstance(function, str):
            func = self.tool_name_map[function]
        else:
            func = function

        _index = self.tools.index(func)
        return self.get_function_schemas(inputs, _index=_index)[0]

    def get_function(self, function_name: str = None):
        if function_name in self.tool_name_map:
            return self.tool_name_map[function_name]
        else:
            raise KeyError(f"Invalid function {function_name}")

    def __iter__(self):
        return iter(self.tools)

    def index(self, function):
        return self.tools.index(function)

    def get_tool_by_name(self, function_name: str, raise_errors: bool = True):
        """Get tool by name. If function_name is not provided, return the first tool."""

        if function_name in self.tool_name_map:
            return self.tool_name_map[function_name]
        elif raise_errors:
            raise KeyError(f"Invalid function {function_name}")


T = TypeVar("T")


class OutputWithFunctionCall(BaseModel, Generic[T]):
    output_text:str
    output_message:AIMessage
    output:T
    tool_call_id:Union[str,None]= None
    function_name:Union[str,None] =None
    function_arguments:Union[Dict[str,Any],str,None]
    function:Union[Callable,None] = None
    function_async:Union[Callable,None] = None
    result: Union[Any,None] = None
    _result_generated = PrivateAttr(False)

    @property
    def is_function_call(self):
        return bool(self.function or self.function_async)

    @property
    def support_async(self):
        return bool(self.function_async)

    @property
    def support_sync(self):
        return bool(self.function)

    async def execute_async(self, augment_args:Callable[[Dict[str,Any]],Dict[str,Any]]=None):
        """Executes the function asynchronously."""
        if not (self.function or self.function_async):
            raise ValueError("No function to execute")
        if not (self.function or self.function_async):
            raise ValueError("No function to execute")

        if not self.function:
            raise ValueError("Function not set")
        else:
            if isinstance(self.function, BaseTool):
                if augment_args:
                    call_kwargs = augment_args(**self.function_arguments)
                else:
                    call_kwargs = self.function_arguments or {}
                result=await self.function.ainvoke(call_kwargs)
            else:
                call_args, call_kwargs = self._get_final_func_args(augment_args)
                result= self.function(*call_args, **call_kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
        self.result = result
        self._result_generated=True
        return result

    def _get_final_func_args(self, augment_args:Callable) -> Dict[str, Any]:
        call_kwargs = self.function_arguments or {}
        call_args=[]
        for k in list(call_kwargs):
            if k.startswith("__arg"):
                # skip private attributes
                call_args.append(call_kwargs.pop(k))
        if augment_args:
            call_kwargs = augment_args(**call_kwargs)
        return call_args, call_kwargs

    def execute(self, augment_args:Callable[[Dict[str,Any]],Dict[str,Any]]=None):
        """ Executes the function synchronously. 
        If the function is async, it will be executed in a event loop.
        """
        if not (self.function or self.function_async):
            raise ValueError("No function to execute")

        if not self.function:
            raise ValueError("Function not set")
        else:
            if isinstance(self.function, BaseTool):
                if augment_args:
                    call_kwargs = augment_args(**self.function_arguments)
                else:
                    call_kwargs = self.function_arguments or {}
                result=self.function.invoke(call_kwargs)
            else:
                call_args, call_kwargs = self._get_final_func_args(augment_args)
                result= self.function(*call_args, **call_kwargs)

        if asyncio.iscoroutine(result):
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None
            if current_loop:
                raise RuntimeError("Cannot execute async function synchronously. Please use execute_async() instead.",)
            else:
                logging.warning("Executing async function synchronously. This is not recommended. Consider using execute_async() instead.")
                result= asyncio.run(self.function_async(**self.function_arguments))
        self.result = result
        self._result_generated=True
        return result

    @property
    def function_call_message(self):
        """ Returns the function call message"""
        if not self.is_function_call:
            raise ValueError("Output was not a function call. You can test this with is_function_call property")
        if self.output_message:
            return self.output_message

    def to_function_message(self, result=None):
        """
        Deprecated: Use function_output_to_message instead
        """
        logging.warning("to_function_message is deprecated, use function_output_to_message instead")
        return self.function_output_to_message(function_output=result)

    def function_output_to_message(self, function_output=None):
        """
        Converts the result of the functional call to a FunctionMessage... 
        you can override the result collected via execute with your own by providing function_output

        Args:
            function_output (Any, optional): function output. If None, it the result collected via execute() or execute_async() will be used. (One of them must be called before).
        """
        if not function_output:
            if not self._result_generated:
                try:
                    self.result = self.execute()
                except RuntimeError as e:
                    if "Cannot execute async function synchronously." in str(e):
                        raise RuntimeError("Cannot execute async function synchronously. Please use await execute_async() to generate the output of the function first") from e
            function_output = self.result
        if isinstance(function_output,BaseModel):
            function_output = function_output.json()
        elif not isinstance(function_output,str):
            function_output = repr(function_output)

        return ToolMessage(tool_call_id=self.tool_call_id, name=self.function_name, content=function_output)


class ToolCall(BaseModel):
    """A tool call that can be used in a LLM chat session."""

    id: str
    name: str
    args: Union[Dict[str, Any], str, None] = None
    function: Optional[Callable] = None
    result: str = None
    is_error_result: bool = False
    metadata: Optional[Dict[str, Any]] = None
    _result_original_value: Any = PrivateAttr(None)

    def __init__(self, **data):
        if "args" in data and isinstance(data["args"], str):
            try:
                data["args"] = json.loads(data["args"])
            except json.JSONDecodeError:
                # If args is a string but not valid JSON, we leave it as is
                data["metadata"] = data.get("metadata", {})
                data["metadata"]["args_string"] = data["args"]
                data["args"] = None
        super().__init__(**data)

    def invoke(self, **kwargs):
        """Executes the tool call and sets the result + adds it to LlmChatSession if available"""
        _res = self(**kwargs)
        if asyncio.iscoroutine(_res):
            raise RuntimeError(
                "Cannot execute async function synchronously. Please use ainvoke() instead."
            )

        self.set_result(_res)
        return _res

    async def ainvoke(self, **kwargs):
        """
        Executes the tool call and sets the result ... regardless of whether the function is async or sync.
        """
        _res = await self(**kwargs)
        if asyncio.iscoroutine(_res):
            self._result_original_value = await _res
        else:
            self._result_original_value = _res

        self.set_result(_res)
        return _res

    def __call__(self, **kwargs):
        """Calls the function with the provided arguments."""
        if not self.function:
            raise ValueError("No function to execute")
        if isinstance(self.function,BaseTool):
            return self.function.invoke(kwargs)
        else:
            args=[]
            for key in kwargs:
                if key.startswith("__arg"):
                    args.append(kwargs.pop(key))

            call_kwargs = self.args or {}
            call_kwargs.update(kwargs)
            return self.function(*args,**call_kwargs)

    def _serialize_result(
        self, tool_results: Union[BaseModel, dict, str, Exception, list, dict]
    ) -> str:
        """Serialize tool results to a JSON string."""
        if isinstance(tool_results, BaseModel):
            return tool_results.model_dump_json()
        elif isinstance(tool_results, dict):
            return json.dumps(tool_results)
        elif isinstance(tool_results, list):
            return "\n".join([self._serialize_result(item) for item in tool_results])
        elif isinstance(tool_results, Exception):
            return f"Error {tool_results.__class__.__name__}: {str(tool_results)}"
        else:
            return str(tool_results)

    def set_result(self, result: Union[BaseModel, dict, str, list]):
        if asyncio.iscoroutine(result):
            raise RuntimeError(
                "Result must be evaluated before setting it! Trying to set coroutine result"
            )
        self._result_original_value = result
        self.result = self._serialize_result(result)
        if LlmChatSession.get_current_session():
            LlmChatSession.get_current_session().add_message(self.to_tool_message())
        return self.result

    def set_error_result(self, error: Union[Exception, str]):
        self.is_error_result = True
        if isinstance(error, Exception):
            self.result = f"Error {error.__class__.__name__}: {str(error)}"
        else:
            self.result = str(error)
        if LlmChatSession.get_current_session():
            LlmChatSession.get_current_session().add_message(self.to_tool_message())

    def to_tool_message(
        self, result: Union[BaseModel, dict, str, list] = None
    ) -> ToolMessage:
        """Converts the tool call to a ToolMessage."""
        result = result or self.result

        return ToolMessage(
            tool_call_id=self.id, content=result or self.result, name=self.name
        )
