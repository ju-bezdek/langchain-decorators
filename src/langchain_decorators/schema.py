import asyncio
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from pydantic import BaseModel

T = TypeVar("T")

class OutputWithFunctionCall(BaseModel):
    output_text:str
    output:T
    function_name:str =None
    function_arguments:Union[Dict[str,Any],str,None]
    function:Callable = None
    function_async:Callable = None
    
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
                return await self.function_async(**self.function_arguments)
            else:
                return await self.function_async(self.function_arguments)
        else:
            await asyncio.sleep(0)
            return self.function(**self.function_arguments)
        
    def execute(self):
        """ Executes the function synchronously. 
        If the function is async, it will be executed in a event loop.
        """
        if not (self.function or self.function_async):
            raise ValueError("No function to execute")
        if self.function:
            if isinstance(self.function_arguments, dict):
                return self.function(**self.function_arguments)
            else:
                self.function(self.function_arguments)
        
        else:
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None
            if current_loop:
                return current_loop.run_until_complete(self.function_async(**self.function_arguments))
            else:
                return asyncio.run(self.function_async(**self.function_arguments))


