import asyncio
import logging
import re
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Union
from pydantic import BaseModel
from langchain.schema import AIMessage
from langchain.schema import FunctionMessage
import json



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
    
    @property
    def is_function_call(self):
        return bool(self.function or self.function_async)
    
    @property
    def support_async(self):
        return bool(self.function_async)
    
    @property
    def support_sync(self):
        return bool(self.function)

    async def execute_async(self):
        """Executes the function asynchronously."""
        if not (self.function or self.function_async):
            raise ValueError("No function to execute")
        if self.function_async:
            if isinstance(self.function_arguments, dict):
                result= await self.function_async(**self.function_arguments)
            else:
                result= await self.function_async(self.function_arguments)
        else:
            await asyncio.sleep(0)
            result= self.function(**self.function_arguments)
            if result and asyncio.iscoroutine(result):
                # this handles special scenario when fake @llm_function is used
                result = await result
        return result
        
    def execute(self):
        """ Executes the function synchronously. 
        If the function is async, it will be executed in a event loop.
        """
        if not (self.function or self.function_async):
            raise ValueError("No function to execute")
        if self.function:
            if isinstance(self.function_arguments, dict):
                result= self.function(**self.function_arguments)
            else:
                result=self.function(self.function_arguments)
        
        else:
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None
            if current_loop:
                result= current_loop.run_until_complete(self.function_async(**self.function_arguments))
            else:
                result= asyncio.run(self.function_async(**self.function_arguments))
        self.result = result
        return result
    
    
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
            if not self.result:
                raise Exception("The function has not been executed yet, or didn't return a result")
            function_output = self.result

        if not isinstance(function_output,str):
            function_output = json.dumps(function_output)

        return FunctionMessage(name=self.function_name, content=function_output)



