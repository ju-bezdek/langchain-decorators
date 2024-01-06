import asyncio
import logging
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Union
from langchain.schema import AIMessage
from langchain.schema import FunctionMessage
import json

import pydantic
if pydantic.__version__ <"2.0.0":
    from pydantic import BaseModel, PrivateAttr
else:
    from pydantic.v1 import BaseModel, PrivateAttr



T = TypeVar("T")

class OutputWithFunctionCall(Generic[T],BaseModel):
    output_text:str
    output_message:AIMessage
    output:T
    function_name:str =None
    function_arguments:Union[Dict[str,Any],str,None]
    function:Callable = None
    function_async:Callable = None
    result: Any = None
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
        call_args = self.function_arguments or {}
        if augment_args:
            call_args = augment_args(call_args)
        if self.function_async:
            result= await self.function_async(**call_args)
        else:
            result= self.function(**call_args)
            if result and asyncio.iscoroutine(result):
                # this handles special scenario when fake @llm_function is used
                result = await result
        self.result = result
        self._result_generated=True
        return result
        
    def execute(self, augment_args:Callable[[Dict[str,Any]],Dict[str,Any]]=None):
        """ Executes the function synchronously. 
        If the function is async, it will be executed in a event loop.
        """
        if not (self.function or self.function_async):
            raise ValueError("No function to execute")
        call_args = self.function_arguments or {}
        if augment_args:
            call_args = augment_args(**call_args)

        if self.function:
            result= self.function(**call_args)
        else:
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

        return FunctionMessage(name=self.function_name, content=function_output)



