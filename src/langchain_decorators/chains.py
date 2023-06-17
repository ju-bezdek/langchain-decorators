from email.mime import message
import json
import logging
from multiprocessing import Value
from pyexpat.errors import messages
from typing import Any, Callable, Dict, List, Optional, Union
from langchain import LLMChain
from langchain.schema import LLMResult
from langchain.callbacks.manager import CallbackManagerForChainRun, AsyncCallbackManagerForChainRun
from langchain.tools.base import BaseTool
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.prompts.chat import  ChatPromptValue
from langchain.schema import PromptValue, ChatGeneration
from pydantic import root_validator

from .function_decorator import get_function_schema
try:
    from langchain.tools.convert_to_openai import format_tool_to_openai_function
except ImportError:
    pass

MODELS_WITH_FUNCTIONS_SUPPORT=["gpt-3.5-turbo-0613","gpt-4-0613"]


class LLMChainWithFunctionSupport(LLMChain):


    functions:Optional[List[Union[Callable, BaseTool]]] = []
    function_schemas:List[dict]=None
    function_call_output_key:str="function_call_info"
    function_output_key:str="function"
    message_output_key:str="message"

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key, self.function_output_key, self.function_call_output_key]

    
    @root_validator(pre=True)
    def validate_and_prepare_chain(cls, values):
        functions = values.get("functions",None)
        llm = values.get("llm",None)
        if functions:
            function_schemas=[None for _ in functions]
            for i,f in enumerate(functions):
                if isinstance(f, BaseTool):
                    function_schemas[i] = format_tool_to_openai_function(f)
                elif callable(f) and hasattr(f,"get_function_schema"):
                    schema = get_function_schema(f)
                    if not schema:
                        raise ValueError(f"Invalid item value in functions. Unable to retrieve schema from function {f}")
                    function_schemas[i] =schema
                else:
                    raise ValueError(f"Invalid item value in functions. Only Tools or functions decorated with @llm_function are allowed. Got: {f}")
            values["function_schemas"] = function_schemas
        if not llm:
            raise ValueError("llm must be defined")
        elif functions:
            if not isinstance(llm,ChatOpenAI):
                raise ValueError(f"llm must be a ChatOpenAI instance. Got: {llm}")
            else:
                if llm.model_name not in MODELS_WITH_FUNCTIONS_SUPPORT:
                    # keeping this as a warning to keep it future proof
                    logging.warn(f"WARNING! Model {llm.model_name} likely does not support functions. Functions will be likely ignored!)")

        return values
    

    def preprocess_inputs(self, input_list):
        additional_kwargs={}
        if "function_call" in input_list[0]:
            for input in input_list:
                function_call=input.pop("function_call")
            # function call should be only one... and the same for all inputs... there shouldn't be more anyway
            if not isinstance(function_call,str):
                #find the index of the function in self.functions
                function_index = self.functions.index(function_call)
                if function_index == -1:
                    raise ValueError(f"Invalid function call. Function {function_call} is not defined in this chain")
                function_call = {"name": self.function_schemas[function_index]["name"]}
            elif function_call not in ["none","auto"]:
                function_call = {"name": function_call}

            additional_kwargs["function_call"]=function_call 
        return additional_kwargs



    def generate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> LLMResult:
        """Generate LLM result from inputs."""
        
        additional_kwargs = self.preprocess_inputs(input_list)
           
        prompts, stop = self.prep_prompts(input_list, run_manager=run_manager)
        if self.functions:
            
            chat_model:BaseChatModel=self.llm
          
            messages = [prompt.to_messages() for prompt in prompts]
             
            result =  chat_model.generate(messages=messages, 
                                        stop=stop, callbacks=run_manager.get_child() if run_manager else None,
                                        functions=self.function_schemas,
                                        **additional_kwargs
                                        )
            return result
        else:
            return self.llm.generate_prompt(
                prompts, stop, callbacks=run_manager.get_child() if run_manager else None
            )

    async def agenerate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> LLMResult:
        """Generate LLM result from inputs."""
        additional_kwargs = self.preprocess_inputs(input_list)
    
        prompts, stop = await self.aprep_prompts(input_list, run_manager=run_manager)
        if self.functions:
            chat_model:BaseChatModel=self.llm
            if len(prompts)!=1:
                raise ValueError("Only one prompt is supported when using functions")
            messages = [prompt.to_messages() for prompt in prompts]
             
            return  await chat_model.agenerate(messages=messages, 
                                         stop=stop, callbacks=run_manager.get_child() if run_manager else None,
                                         functions=self.function_schemas,
                                         **additional_kwargs
                                         )
        else:
            return await self.llm.agenerate_prompt(
                prompts, stop, callbacks=run_manager.get_child() if run_manager else None
            )
        
    def _create_output(self,generation):
        res = {
                self.output_key: generation.text,
                self.function_call_output_key: None,
                self.function_output_key: None,
             }
        if isinstance(generation, ChatGeneration):
            res[self.message_output_key] = generation.message
            # let's make a copy of the function call so that we don't modify the original
            function_call = dict(generation.message.additional_kwargs.get("function_call")) if generation.message.additional_kwargs else {}
            if function_call:
                if isinstance(function_call["arguments"],str):
                    function_call["arguments"]=json.loads(function_call["arguments"])
            if generation.message.additional_kwargs and generation.message.additional_kwargs.get("function_call"):
                res[self.function_call_output_key] = function_call
                res[self.function_output_key] = self.find_func(function_call["name"]) if function_call else None
        return res

    def find_func(self,function_name):
        for i,f in enumerate(self.function_schemas):
            if f["name"] == function_name:
                return self.functions[i]
        #else (not found)
        ## TODO: raise error or retry?
    

    def create_outputs(self, response: LLMResult) -> List[Dict[str, str]]:
        """Create outputs from response."""
        
        return [
            self._create_output(generation[0])
            for generation in response.generations
        ]


    