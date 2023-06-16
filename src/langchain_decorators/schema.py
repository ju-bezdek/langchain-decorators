import asyncio
import re
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Union
from pydantic import BaseModel
from langchain.schema import FunctionMessage
import json

from regex import F

T = TypeVar("T")

class OutputWithFunctionCall(Generic[T],BaseModel):
    output_text:str
    output:T
    function_name:str =None
    function_arguments:Union[Dict[str,Any],str,None]
    function:Callable = None
    function_async:Callable = None
    result: Any = None
    
    @property
    def is_function_call(self):
        return (self.function or self.function_async)
    
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
        if not result:
            if not self.result:
                raise Exception("The function has not been executed yet, or didn't return a result")
            result = self.result

        if not isinstance(result,str):
            result = json.dumps(result)

        return FunctionMessage(name=self.function_name, content=result)
    



