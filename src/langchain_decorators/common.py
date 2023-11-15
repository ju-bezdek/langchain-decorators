
import re
import inspect
import json
import logging
import os
from textwrap import dedent
from langchain.prompts import PromptTemplate
import yaml
from enum import Enum
from typing import Any, Coroutine, Dict, List, Type, Union, Optional, Tuple, get_args, get_origin, TYPE_CHECKING
from langchain.llms.base import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage
from langchain.prompts.chat import ChatMessagePromptTemplate
from typing_inspect import is_generic_type, is_union_type

import pydantic



if pydantic.__version__ <"2.0.0":
    from pydantic import BaseConfig, BaseModel, Extra, Field
    from pydantic.fields import ModelField
else:
    from pydantic.v1 import BaseConfig, BaseModel, Extra, Field
    from pydantic.v1.fields import ModelField
    
if TYPE_CHECKING:
    from .prompt_template import BaseTemplateBuilder


class LlmSelector(BaseModel):

    rules:List[dict]=[]
    llms:Dict[int,BaseLanguageModel]=Field(default_factory=dict)
    streamable_llms_cache:Dict[int,BaseLanguageModel]=Field(default_factory=dict)
    generation_min_tokens:Optional[int]
    prompt_to_generation_ratio:Optional[float]

    def __init__(self, generation_min_tokens:int=None, prompt_to_generation_ratio:float=1/3):
        """ Create a LlmSelector that will select the llm based on the length of the prompt.
        
        Args:
            generation_min_tokens (int, optional): The minimum number of tokens that the llm is expecting generate. Defaults to None (prompt_to_generation_ratio will be used).
            prompt_to_generation_ratio (float, optional): The ratio of the prompt length to the generation length. Defaults to 1/3. 
        """
        super().__init__(generation_min_tokens=generation_min_tokens, prompt_to_generation_ratio=prompt_to_generation_ratio)


    def with_llm(self, llm:BaseLanguageModel,  llm_selector_rule_key:str=None):
        """ this will automatically add a rule with token window based on the model name. Only works for OpenAI and Anthropic models."""
        max_tokens = self.get_model_window(llm.model_name)
        if max_tokens:
            self.with_llm_rule(llm, max_tokens, llm_selector_rule_key=llm_selector_rule_key)
        else:
            raise Exception(f"Could not find a token limit for model {llm.model_name}. Please use `with_llm_rule` and specify the max_tokens for your model.")
 
        return self

    def with_llm_rule(self, llm:BaseLanguageModel, max_tokens:int, llm_selector_rule_key:str=None):
        """ Add a LLM with a selection rule defined by max tokens and llm_selector_rule_key.
        

        Args:
            llm (BaseLanguageModel): The LLM to add
            max_tokens (int): The maximum number of tokens that the LLM can generate / we want it to use it for.
            llm_selector_rule_key (str, optional): Optional selection key to limit the selection by. This allows us to pick LLM only from a subset of LLMs (or even just one). Defaults to None.

        """
        i=len(self.rules)
        self.rules.insert(i, dict(max_tokens=max_tokens, llm_selector_rule_key=llm_selector_rule_key))
        self.llms[i]=llm    
        return self
        
    def get_model_window(self, model_name:str)->int:
        for model_pattern, max_tokens in MODEL_LIMITS.items():
            if re.match(model_pattern, model_name):
                return max_tokens
        

    def get_llm(self, prompt:Union[str,List[BaseMessage]], function_schemas:List[dict]=None, expected_generated_tokens:int=None, streaming=False, llm_selector_rule_key:str=None)->BaseLanguageModel:
        """Picks the best LLM based on the rules and the prompt length.

        Args:
            prompt (Union[str,List[BaseMessage]]): the prompt ... messages or string
            function_schemas (List[dict], optional): openAI function schemas. Defaults to None. (are included in the token limit)
            expected_generated_tokens (int, optional): Number of tokens we expect model to generate. Help for better precision. If None, the prompt_to_generation_ratio will be used (defaults to 1/3 - means 30% above the prompt length)

        """
        if not self.llms:
            raise Exception("No LLMs rules added to the LlmSelector")
        
        result_index = None
        first_rule = self.rules[0]
        first_token_threshold = first_rule.get("max_tokens")
        total_tokens_estimate = self.get_expected_total_tokens(prompt, function_schemas=function_schemas, estimate=True, expected_generated_tokens=expected_generated_tokens)
        if total_tokens_estimate<first_token_threshold and not llm_selector_rule_key:
            result_index = 0
        else:
            total_tokens = self.get_expected_total_tokens(prompt, function_schemas=function_schemas, estimate=False, expected_generated_tokens=expected_generated_tokens) 
            key_match=False
            best_match = None
            best_match_top_tokens = 0
            for i, rule in enumerate(self.rules):
                if llm_selector_rule_key:
                    if rule.get("llm_selector_rule_key") != llm_selector_rule_key:
                        continue
                    else:
                        key_match=True
                max_tokens = rule.get("max_tokens")
                if max_tokens and max_tokens >=total_tokens:
                    result_index = i
                    break
                else:
                    if max_tokens and max_tokens > best_match_top_tokens:
                        best_match_top_tokens = max_tokens
                        best_match = i
                
            # if no condition is met, return the last llm
            if llm_selector_rule_key and not key_match:
                raise Exception(f"Could not find a LLM for key {llm_selector_rule_key}. Valid keys are: {set([rule.get('llm_selector_rule_key') for rule in self.rules])}")
            if result_index == None:
                result_index = best_match
        print_log(f"LLMSelector: Using {'default' if result_index==0 else str(result_index)+'-th'} LLM: {getattr(self.llms[result_index],'model_name', self.llms[result_index].__class__.__name__)}", logging.DEBUG )
        if streaming:
            if result_index not in self.streamable_llms_cache:
                self.streamable_llms_cache[result_index] = make_llm_streamable(self.llms[result_index])
            return self.streamable_llms_cache[result_index]
        else:
            return self.llms[result_index]
    
    def get_expected_total_tokens(self, prompt:Union[str,List[BaseMessage]], function_schemas:List[dict]=None, estimate:bool=True,expected_generated_tokens=None)->int:
        expected_generated_tokens = expected_generated_tokens or self.generation_min_tokens or 0
        prompt_tokens = self.get_token_count(prompt, function_schemas=function_schemas, estimate=estimate) 
        if expected_generated_tokens:
            return prompt_tokens + expected_generated_tokens
        else:
             return prompt_tokens * (1+(self.prompt_to_generation_ratio or 0))
        
    
    def get_token_count(self, prompt:Union[str,List[BaseMessage]], function_schemas:List[dict]=None, estimate:bool=True)->int:
        """Get the number of tokens in the prompt. If estimate is True, it will use a fast estimation, otherwise it will use the llm to count the tokens (slower)"""
        if estimate:
            num_tokens = int(len(prompt)/2)
        else:
            num_tokens = count_tokens(prompt, llm=self.llms[0] ) # note: we will use the first llm to count the tokens... it should be the same general type, and if not, it's ok, should be close enough
        
        if function_schemas:
            num_tokens += self.get_token_count(json.dumps(function_schemas), estimate=estimate)
        return num_tokens
    


class GlobalSettings(BaseModel):
    default_llm: Optional[BaseLanguageModel] = None
    default_streaming_llm: Optional[BaseLanguageModel] = None
    logging_level: int = logging.INFO
    verbose: bool = False
    llm_selector: Optional[LlmSelector] = None

    class Config:
        allow_population_by_field_name = True
        extra = Extra.allow

    @classmethod
    def define_settings(cls,
                        settings_type="default",
                        default_llm:BaseLanguageModel=None,
                        default_streaming_llm:BaseLanguageModel=None,
                        logging_level=logging.INFO,
                        verbose=None,
                        llm_selector: Optional["LlmSelector"] = None,
                        **kwargs
                        ):
        """ Define the global settings for the project.
        
        Args:
            settings_type (str, optional): The name of the settings. Defaults to "default".
            default_llm (BaseLanguageModel, optional): The default language model to use. Defaults to None.
            default_streaming_llm (BaseLanguageModel, optional): The default streaming language model to use. Defaults to None.
            llm_selector (Optional[LlmSelector], optional): The language model selector to use. Defaults to None.
            logging_level (int, optional): The logging level to use. Defaults to logging.INFO.

        """
        if llm_selector is None and default_llm is None and default_streaming_llm is None:
            # only use llm_selector if no default_llm and default_streaming_llm is defined, because than we dont know what rules to set up
            default_llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-1106" if USE_PREVIEW_MODELS else "gpt-3.5-turbo", request_timeout=90) #  '-0613' - has function calling
            default_streaming_llm = make_llm_streamable(default_llm)
            llm_selector = LlmSelector()\
                .with_llm(default_llm, llm_selector_rule_key="chatGPT")\
                .with_llm(ChatOpenAI(temperature=0.0, model="gpt-4-1106-preview"  if USE_PREVIEW_MODELS else "gpt-3.5-turbo-16k",  request_timeout=120), llm_selector_rule_key="GPT4")\
                #.with_llm(ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-1106"), llm_selector_rule_key="chatGPT")\
                #.with_llm(ChatOpenAI(temperature=0.0, model="gpt-4-32k"), llm_selector_rule_key="GPT4") 
        
        else:
            if default_llm is None:
                default_llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-1106" if USE_PREVIEW_MODELS else "gpt-3.5-turbo", request_timeout=90)  #  '-0613' - has function calling
            if default_streaming_llm is None:
                default_streaming_llm = make_llm_streamable(default_llm)
            
        if verbose is None:
            verbose = os.environ.get("LANGCHAIN_DECORATORS_VERBOSE", False) in [True,"true","True","1"]
        settings = cls(default_llm=default_llm, default_streaming_llm=default_streaming_llm,
                       logging_level=logging_level, verbose=verbose, llm_selector=llm_selector, **kwargs)
        if not hasattr(GlobalSettings, "registry"):
            setattr(GlobalSettings, "registry", {})
        GlobalSettings.registry[settings_type] = settings

    @classmethod
    def get_current_settings(cls) -> "GlobalSettings":
        if not hasattr(GlobalSettings, "settings_type"):
            setattr(GlobalSettings, "settings_type", "default")
        if not hasattr(GlobalSettings, "registry"):
            GlobalSettings.define_settings()
        return GlobalSettings.registry[GlobalSettings.settings_type]

    @classmethod
    def switch_settings(cls, project_name):
        GlobalSettings.settings_type = project_name




class LogColors(Enum):
    WHITE_BOLD = "\033[1m"
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    DARK_GRAY = '\033[90m'
    WHITE = '\033[39m'
    BLACK_AND_WHITE = '\033[40m'

    # Define some reset codes to restore the default text color
    RESET = '\033[0m'


def print_log(log_object: Any, log_level: int, color: LogColors = None):
    settings = GlobalSettings.get_current_settings()
    if settings.logging_level <= log_level or settings.verbose:
        if isinstance(log_object, str):
            pass
        elif isinstance(log_object, dict):
            log_object = yaml.safe_dump(log_object)
        elif isinstance(log_object, BaseModel):
            log_object = yaml.safe_dump(log_object.dict())

        if color is None:
            if log_level >= logging.ERROR:
                color = LogColors.RED
            elif log_level >= logging.WARNING:
                color = LogColors.YELLOW
            elif log_level >= logging.INFO:
                color = LogColors.GREEN
            else:
                color = LogColors.DARK_GRAY
        if type(color) is LogColors:
            color = color.value
        reset = LogColors.RESET.value if color else ""
        print(f"{color}{log_object}{reset}\n", flush=True)


class PromptTypeSettings:
    def __init__(self, 
                 llm: BaseLanguageModel = None,  
                 color: LogColors = None, 
                 log_level: Union[int, str] = "info", 
                 capture_stream: bool = False, 
                 llm_selector: "LlmSelector" = None, 
                 prompt_template_builder: "BaseTemplateBuilder" = None):
        self.color = color or LogColors.DARK_GRAY
        if isinstance(log_level, str):
            log_level = getattr(logging, log_level.upper())
        self.log_level = log_level
        self.capture_stream = capture_stream
        self.llm = llm
        self.llm_selector = llm_selector
        
        self._prompt_template_builder = prompt_template_builder 

    
    @property
    def prompt_template_builder(self):
        # lazy init due to circular imports
        if not self._prompt_template_builder:
            from .prompt_template import OpenAITemplateBuilder
            self._prompt_template_builder= OpenAITemplateBuilder()
        return self._prompt_template_builder
            


    def as_verbose(self):
        return PromptTypeSettings(llm=self.llm, color=self.color, log_level=100, capture_stream=self.capture_stream, llm_selector=self.llm_selector, prompt_template_builder=self.prompt_template_builder)

USE_PREVIEW_MODELS = os.environ.get("LANGCHAIN_DECORATORS_USE_PREVIEW_MODELS", True) in [True,"true","True","1"]

class PromptTypes:
    UNDEFINED: PromptTypeSettings = PromptTypeSettings(
        color=LogColors.BLACK_AND_WHITE, log_level=logging.DEBUG)
    
    BIG_CONTEXT: PromptTypeSettings = PromptTypeSettings(
        llm=ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-16k", request_timeout=60), 
        color=LogColors.BLACK_AND_WHITE, log_level=logging.DEBUG)

    GPT4: PromptTypeSettings = PromptTypeSettings(
        llm=ChatOpenAI(temperature=0.0, model="gpt-4-1106-preview" if USE_PREVIEW_MODELS else "gpt-4", request_timeout=90), 
        color=LogColors.BLACK_AND_WHITE, log_level=logging.DEBUG)
     
    BIG_CONTEXT_GPT4: PromptTypeSettings = PromptTypeSettings(
        llm=ChatOpenAI(temperature=0.0, model="gpt-4-1106-preview" if USE_PREVIEW_MODELS else "gpt-4", request_timeout=90), 
        color=LogColors.BLACK_AND_WHITE, log_level=logging.DEBUG)
    
    
    AGENT_REASONING: PromptTypeSettings = PromptTypeSettings(
        color=LogColors.GREEN, log_level=logging.DEBUG)
    TOOL: PromptTypeSettings = PromptTypeSettings(
        color=LogColors.BLUE, log_level=logging.DEBUG)
    FINAL_OUTPUT: PromptTypeSettings = PromptTypeSettings(
        color=LogColors.YELLOW, log_level=logging.DEBUG)


def get_func_return_type(func: callable, with_args:bool=False)->Union[Type, Tuple[Type, List[Type]]]:
    return_type = func.__annotations__.get("return",None)
    if inspect.iscoroutinefunction(func):
        if return_type:
            if is_generic_type(return_type):
                return_type_origin = get_origin(return_type)
                if return_type_origin and issubclass(return_type_origin, Coroutine):
                    return_type_args = getattr(return_type, '__args__', None)
                    if return_type_args and len(return_type_args) == 3:
                        return_type = return_type_args[2]
                    else:
                        raise Exception(f"Invalid Coroutine annotation {return_type}. Expected Coroutine[ any , any, <return_type>] or just <return_type>")
                else:
                    return_type = return_type_origin
        else:
            
            if return_type and issubclass(return_type, Coroutine):
                return None if not with_args else (None, None)
            else:
                return_type = return_type
    
    if return_type and is_union_type(return_type):
        return_type_args = getattr(return_type, '__args__', None)
        if return_type_args and len(return_type_args) == 2 and return_type_args[1] == type(None):
            return return_type_args[0] if not with_args else (return_type_args[0], return_type_args)
        else:
            raise Exception(f"Invalid Union annotation {return_type}. Expected Union[ <return_type>, None] or just <return_type>")
    elif is_generic_type(return_type):
        # this should cover list and dict
        return get_origin(return_type) if not with_args else (get_origin(return_type), get_args(return_type))
    else:
        return return_type if not with_args else (return_type, None)
            
            
def get_function_docs(func: callable)->Tuple:
    if not func.__doc__:
        return None
    fist_line, rest = func.__doc__.split('\n', 1) if '\n' in func.__doc__ else (func.__doc__, "")
    # we dedent the first line separately,because its common that it often starts right after """
    fist_line = fist_line.strip()
    if fist_line:
        fist_line+="\n"
    docs = fist_line + dedent(rest)
    return docs
    

            
def get_function_full_name(func: callable)->str:
    return  f"{func.__module__}.{func.__name__}" if not func.__module__=="__main__" else func.__name__
    


def get_arguments_as_pydantic_fields(func) -> Dict[str, ModelField]:
    argument_types = {}
    model_config = BaseConfig()
    for arg_name, arg_desc in inspect.signature(func).parameters.items():
        if arg_name != "self" and not (arg_name.startswith("_") and arg_desc.default!=inspect.Parameter.empty):
            default = arg_desc.default if arg_desc.default!=inspect.Parameter.empty else None
            if arg_desc.annotation==inspect._empty:
                raise Exception(f"Argument '{arg_name}' of function {func.__name__} has no type annotation")
            argument_types[arg_name] = ModelField(
                class_validators=None,
                model_config=model_config,
                name=arg_name, 
                type_=arg_desc.annotation,
                default=default,
                required= arg_desc.default==inspect.Parameter.empty
                )
            
    return argument_types


def make_llm_streamable(llm:BaseLanguageModel):
    try:
        if hasattr(llm,"lc_kwargs"):
            # older version support
            lc_kwargs = {**llm.lc_kwargs}
        else:
            lc_kwargs = {
                k: getattr(llm, k, v)
                for k, v in llm._lc_kwargs.items()
                if not (llm.__exclude_fields__ or {}).get(k, False)  # type: ignore
            }
        lc_kwargs["streaming"] = True
        return llm.__class__(**lc_kwargs)
    except Exception as e:
        logging.warning(f"Could not make llm {llm} streamable. Error: {e}")
        

def count_tokens(prompt: Union[str,List[BaseMessage]], llm:BaseLanguageModel) -> int:
    """Returns the number of tokens in a text string."""
    if isinstance(prompt,str):
        return llm.get_num_tokens(prompt)
    elif isinstance(prompt,list):
        return llm.get_num_tokens_from_messages(prompt)


MODEL_LIMITS={
    "gpt-3.5-turbo-16k.*": 16_384,
    "gpt-3.5-turbo.*": 4_096,

    "text-davinci-003.*": 4_097,
    "text-davinci-003.*": 4_097,
    "code-davinci-002.*": 8_001,
    "gpt-4-1106-preview": 128_000,
    "gpt-4-32k.*": 32_768,
    "gpt-4.*": 8_192,
    
    "claude-v1":9000,
    r"claude-v\d(\.\d+)?-100k":100_000,
}
