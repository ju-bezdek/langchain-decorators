import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from langchain import LLMChain
from langchain.schema import LLMResult
from langchain.callbacks.manager import CallbackManagerForChainRun, AsyncCallbackManagerForChainRun
from langchain.tools.base import BaseTool
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.prompts.chat import  ChatPromptValue
from langchain.schema import ChatGeneration
from pydantic import root_validator
from promptwatch import CachedChatLLM
from .common import LlmSelector

from .function_decorator import get_function_schema
try:
    from langchain.tools.convert_to_openai import format_tool_to_openai_function
except ImportError:
    pass

MODELS_WITH_FUNCTIONS_SUPPORT=["gpt-3.5-turbo-0613","gpt-4-0613"]



class FunctionsProvider:

    def __init__(self, functions:Union[List[Union[Callable, BaseTool]], Dict[str,Union[Callable, BaseTool]]]) -> None:
        """ Initialize FunctionsProvider with list of funcitons of dictionary where key is the unique function name alias"""
        self.functions=[]
        self.aliases=[]
        self.function_schemas=[]
        self.func_name_map={}
        if not (isinstance(functions, dict) or isinstance(functions, list)):
            raise ValueError("FunctionsProvider must be initialized with list of functions or dictionary where key is the unique function name alias")
        
        for i,f in enumerate(functions):
            
            if isinstance(f, str):
                function_alias = f
                f = functions[f]
            else:
               function_alias=None

            self.add_function(f, function_alias)
        
    
    def add_function(self, function:Union[Callable, BaseTool], alias:str=None):
        """ Add function to FunctionsProvider. If alias is provided, it will be used as function name in LLM"""
        self.functions.append(function)
        self.aliases.append(alias)
        if isinstance(function, BaseTool):
            self.function_schemas.append(format_tool_to_openai_function(function))
            f_name = alias or function.name
        elif callable(function) and hasattr(function,"get_function_schema"):
            if hasattr(function,"function_name"):
                f_name = alias or function.function_name
            else:
                raise Exception(f"Function {function} does not have function_name attribute. All functions must be marked with @llm_function decorator")
            self.function_schemas.append(lambda kwargs, f=function: get_function_schema(f, kwargs))
        else:
            raise ValueError(f"Invalid item value in functions. Only Tools or functions decorated with @llm_function are allowed. Got: {function}")
        if f_name in self.func_name_map:
            if alias:
                raise ValueError(f"Invalid alias - duplicate function name: {f_name}.")
            else:
                raise ValueError(f"Duplicate function name: {f_name}. Use unique function names, or use FunctionsProvider and assign a unique alias to each function.")
        self.func_name_map[f_name]=function
        
    def __contains__(self, function):
        return function in self.functions

    def get_function_schemas(self, inputs, _index:int=None):
        if self.function_schemas:
            _f_schemas = []
            for i, (alias, f_schema_builder) in enumerate(zip(self.aliases,self.function_schemas)):
                if _index is not None and i!=_index:
                    continue

                if callable(f_schema_builder):
                    _f_schema = f_schema_builder(inputs)
                else:
                    _f_schema = f_schema_builder

                if alias:
                    _f_schema["name"]=alias
                
                _f_schemas.append(_f_schema)
                
            return _f_schemas
        else:
            None


    
    def get_function_schema(self, function:Union[str, Callable], inputs:dict):
        index=None
        if isinstance(function, str):
            func = self.func_name_map[function]
        else:
            func = function
            
        _index = self.functions.index(func)
        return self.get_function_schemas(inputs, _index=_index)[0]




    def get_function(self, function_name:str=None):
        if function_name in self.func_name_map:
            return self.func_name_map[function_name]
        else:
            raise KeyError(f"Invalid function {function_name}")
        
    def __iter__(self):
        return iter(self.functions)
    
    def index(self, function):
        return self.functions.index(function)


class LLMDecoratorChain(LLMChain):

    llm_selector:LlmSelector=None
    """ Optional LLM selector to pick the right LLM for the job. """
    capture_stream:bool=False
    expected_gen_tokens:Optional[int]=None
    llm_selector_rule_key:Optional[str]=None

    def select_llm(self, prompts, inputs=None):
        if self.llm_selector:
            # we pick the right LLM based on the first prompt
            first_prompt = prompts[0]
            if isinstance(first_prompt, ChatPromptValue):
                llm = self.llm_selector.get_llm(first_prompt.messages,**self._additional_llm_selector_args(inputs))
            else:
                llm =  self.llm_selector.get_llm(first_prompt.to_string(),**self._additional_llm_selector_args(inputs))
        else:
            llm = self.llm
        return llm
    
    def _additional_llm_selector_args(self, inputs):
        return {
            "expected_generated_tokens":self.expected_gen_tokens, 
            "streaming":self.capture_stream,
            "llm_selector_rule_key":self.llm_selector_rule_key
            }

    def generate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> LLMResult:
        """Generate LLM result from inputs."""
        prompts, stop = self.prep_prompts(input_list, run_manager=run_manager)


        return self.select_llm(prompts, input_list[0]).generate_prompt(
            prompts, stop, callbacks=run_manager.get_child() if run_manager else None
        )

    async def agenerate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> LLMResult:
        """Generate LLM result from inputs."""
        prompts, stop = await self.aprep_prompts(input_list, run_manager=run_manager)

        return await self.select_llm(prompts, input_list[0]).agenerate_prompt(
            prompts, stop, callbacks=run_manager.get_child() if run_manager else None
        )

class LLMDecoratorChainWithFunctionSupport(LLMDecoratorChain):


    functions:Union[FunctionsProvider,List[Union[Callable, BaseTool]]]
    func_name_map:dict=None
    
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
        if isinstance(functions,list):
            values["functions"] = FunctionsProvider(functions)
        elif isinstance(functions,FunctionsProvider):
            values["functions"] = functions
        elif functions:
            raise ValueError(f"functions must be a List[Callable|BaseTool] or FunctionsProvider instance. Got: {functions.__class__}")
        
        if not llm:
            raise ValueError("llm must be defined")
        
        
        if not isinstance(llm,ChatOpenAI) and not isinstance(llm, CachedChatLLM):
            raise ValueError(f"llm must be a ChatOpenAI instance. Got: {llm}")
        else:
            if not isinstance(llm, CachedChatLLM) and getattr(llm,"model_name",None) not in MODELS_WITH_FUNCTIONS_SUPPORT:
                # keeping this as a warning to keep it future proof
                logging.warn(f'WARNING! Model {getattr(llm,"model_name", "-unknown-")} likely does not support functions. Functions will be likely ignored!)')

        return values
    

    def get_final_function_schemas(self, inputs):
        return self.functions.get_function_schemas(inputs)

            
    def _additional_llm_selector_args(self, inputs):
        args = super()._additional_llm_selector_args(inputs)
        args["function_schemas"]=self.get_final_function_schemas(inputs)
        return args
    
    def preprocess_inputs(self, input_list):
        additional_kwargs={}
        final_function_schemas=None
        if self.functions:
            if self.memory is not None:
                # we are sending out more outputs... memory expects only one (AIMessage... so let's set it, becasue user has no way to know these internals)
                if hasattr(self.memory, "output_key") and not self.memory.output_key:
                    self.memory.output_key = "message"
            if len(input_list)!=1:
                    raise ValueError("Only one input is allowed when using functions")
            if "function_call" in input_list[0]:
                for input in input_list:
                    function_call=input.pop("function_call")
                # function call should be only one... and the same for all inputs... there shouldn't be more anyway
                if not isinstance(function_call,str):
                    f_name = next((f_name for f_name, func in  self.functions.func_name_map.items() if func == function_call), None)
                    if not f_name:
                        raise ValueError(f"Invalid function call. Function {function_call} is not defined in this chain")
                    function_call = {"name": f_name}
                elif function_call not in ["none","auto"]:
                    # test if it's a valid function name
                    self.get_function(function_call)

                    function_call = {"name": function_call}

                additional_kwargs["function_call"]=function_call 
            final_function_schemas = self.get_final_function_schemas(input_list[0])
        return additional_kwargs, final_function_schemas



    def generate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> LLMResult:
        """Generate LLM result from inputs."""
        
        

        additional_kwargs, final_function_schemas = self.preprocess_inputs(input_list)
           
        prompts, stop = self.prep_prompts(input_list, run_manager=run_manager)
        if self.functions:
            
            chat_model:BaseChatModel=self.select_llm(prompts, input_list[0])
          
            messages = [prompt.to_messages() for prompt in prompts]

            result =  chat_model.generate(messages=messages, 
                                        stop=stop, callbacks=run_manager.get_child() if run_manager else None,
                                        functions=final_function_schemas,
                                        **additional_kwargs
                                        )
            return result
        else:
            return self.select_llm(prompts, input_list[0]).generate_prompt(
                prompts, stop, callbacks=run_manager.get_child() if run_manager else None
            )

    async def agenerate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> LLMResult:
        """Generate LLM result from inputs."""
        additional_kwargs, final_function_schemas = self.preprocess_inputs(input_list)
    
        prompts, stop = await self.aprep_prompts(input_list, run_manager=run_manager)
        if self.functions:
            chat_model:BaseChatModel=self.select_llm(prompts, input_list[0])
            
            messages = [prompt.to_messages() for prompt in prompts]
             
            return  await chat_model.agenerate(messages=messages, 
                                         stop=stop, callbacks=run_manager.get_child() if run_manager else None,
                                         functions=final_function_schemas,
                                         **additional_kwargs
                                         )
        else:
            return await self.select_llm(prompts, input_list[0]).agenerate_prompt(
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
                res[self.function_output_key] = self.get_function(function_call["name"]) if function_call else None
        return res

    def get_function(self,function_name):
        return self.functions.get_function(function_name)
    

    def create_outputs(self, response: LLMResult) -> List[Dict[str, str]]:
        """Create outputs from response."""
        
        return [
            self._create_output(generation[0])
            for generation in response.generations
        ]


    