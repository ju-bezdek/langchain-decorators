import inspect
import json
import logging
from functools import wraps
from time import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union, cast

import httpx
import pydantic
from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain.chains import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import PromptTemplate
from langchain.prompts.base import StringPromptValue
from langchain.prompts.chat import ChatPromptValue
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    Generation,
    HumanMessage,
    LLMResult,
)
from langchain.schema.output import LLMResult
from langchain.tools.base import BaseTool
from langchain_core.language_models import BaseLanguageModel, LanguageModelInput

from .common import LlmSelector, LogColors, PromptTypes, PromptTypeSettings, print_log
from .function_decorator import get_function_schema
from .output_parsers import (
    BaseOutputParser,
    OpenAIFunctionsPydanticOutputParser,
    OutputParserExceptionWithOriginal,
)
from .prompt_template import PromptDecoratorTemplate
from .schema import OutputWithFunctionCall

import pydantic
if pydantic.__version__ <"2.0.0":
    from pydantic import BaseModel, PrivateAttr
else:
    from pydantic.v1 import BaseModel as BaseModelV1

    if issubclass(AIMessage, BaseModelV1):
        from pydantic.v1 import BaseModel, PrivateAttr
    else:
        from pydantic import BaseModel, PrivateAttr

CachedChatLLM = None
register_prompt_template = None
try:
    from langchain.tools.convert_to_openai import format_tool_to_openai_function
    from promptwatch import CachedChatLLM, register_prompt_template
except ImportError:
    pass

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None
    pass

MODELS_WITH_JSON_FORMAT_SUPPORT = [
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
    "gpt-3.5-turbo-0125",
    "gpt-4-0125-preview",
]


class FunctionsProvider:

    def __init__(
        self,
        functions: Union[
            List[Union[Callable, BaseTool]], Dict[str, Union[Callable, BaseTool]]
        ],
    ) -> None:
        """Initialize FunctionsProvider with list of funcitons of dictionary where key is the unique function name alias"""
        self.functions = []
        self.aliases = []
        self.function_schemas = []
        self.func_name_map = {}
        if not (isinstance(functions, dict) or isinstance(functions, list)):
            raise ValueError(
                "FunctionsProvider must be initialized with list of functions or dictionary where key is the unique function name alias"
            )

        for i, f in enumerate(functions):

            if isinstance(f, str):
                function_alias = f
                f = functions[f]
            else:
                function_alias = None

            self.add_function(f, function_alias)

    def add_function(self, function: Union[Callable, BaseTool], alias: str = None):
        """Add function to FunctionsProvider. If alias is provided, it will be used as function name in LLM"""
        self.functions.append(function)
        self.aliases.append(alias)
        if isinstance(function, BaseTool):
            self.function_schemas.append(format_tool_to_openai_function(function))
            f_name = alias or function.name
        elif callable(function) and hasattr(function, "get_function_schema"):
            if hasattr(function, "function_name"):
                f_name = alias or function.function_name
            else:
                raise Exception(
                    f"Function {function} does not have function_name attribute. All functions must be marked with @llm_function decorator"
                )
            self.function_schemas.append(
                lambda kwargs, f=function: get_function_schema(f, kwargs)
            )
        else:
            raise ValueError(
                f"Invalid item value in functions. Only Tools or functions decorated with @llm_function are allowed. Got: {function}"
            )
        if f_name in self.func_name_map:
            if alias:
                raise ValueError(f"Invalid alias - duplicate function name: {f_name}.")
            else:
                raise ValueError(
                    f"Duplicate function name: {f_name}. Use unique function names, or use FunctionsProvider and assign a unique alias to each function."
                )
        self.func_name_map[f_name] = function

    def __contains__(self, function):
        return function in self.functions

    def get_function_schemas(self, inputs, _index: int = None):
        if self.function_schemas:
            _f_schemas = []
            for i, (alias, f_schema_builder) in enumerate(
                zip(self.aliases, self.function_schemas)
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
            func = self.func_name_map[function]
        else:
            func = function

        _index = self.functions.index(func)
        return self.get_function_schemas(inputs, _index=_index)[0]

    def get_function(self, function_name: str = None):
        if function_name in self.func_name_map:
            return self.func_name_map[function_name]
        else:
            raise KeyError(f"Invalid function {function_name}")

    def __iter__(self):
        return iter(self.functions)

    def index(self, function):
        return self.functions.index(function)


class LLMDecoratorChain(LLMChain):
    name: str
    llm_selector: LlmSelector = None
    """ Optional LLM selector to pick the right LLM for the job. """
    capture_stream: bool = False
    expected_gen_tokens: Optional[int] = None
    llm_selector_rule_key: Optional[str] = None
    allow_retries: bool = True
    format_instructions_parameter_key: str = ("FORMAT_INSTRUCTIONS",)

    prompt_type: PromptTypeSettings = PromptTypes.UNDEFINED
    default_call_kwargs: Optional[Dict[str, Any]]
    _additional_instruction: Optional[str] = PrivateAttr()
    _is_retry: Optional[str] = PrivateAttr(default=False)

    def __call__(
        self,
        inputs: Union[Dict[str, Any], Any] = None,
        return_only_outputs: bool = False,
        callbacks: Callbacks = None,
        *,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, Any], None] = None,
        include_run_info: bool = False,
        additional_instruction: str = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Call the chain with inputs."""

        # override of __call__ so this can be run preinitialized by the decorator call
        kwargs["inputs"] = inputs
        kwargs["return_only_outputs"] = return_only_outputs
        kwargs["callbacks"] = callbacks
        kwargs["tags"] = tags
        kwargs["metadata"] = metadata
        kwargs["include_run_info"] = include_run_info
        self._additional_instruction = additional_instruction
        if self.default_call_kwargs:
            for k, v in self.default_call_kwargs.items():
                if (
                    kwargs.get(k, None) is None
                    and k in self.default_call_kwargs
                    and k != "llm_kwargs"
                ):
                    kwargs[k] = v
        print_log(
            log_object=f"> Entering {self.name} prompt decorator chain",
            log_level=self.prompt_type.log_level,
            color=LogColors.WHITE_BOLD,
        )
        try:
            result = super().__call__(**kwargs)
        except RequestRetryWithFeedback as e:
            if self._is_retry == True:
                raise Exception(e.feedback)
            self._is_retry = True
            self._additional_instruction = e.feedback
            result = super().__call__(**kwargs)
        print_log(
            log_object=f"> Finished chain",
            log_level=self.prompt_type.log_level,
            color=LogColors.WHITE_BOLD,
        )
        self._additional_instruction = None
        return result

    async def acall(
        self,
        inputs: Union[Dict[str, Any], Any] = None,
        return_only_outputs: bool = False,
        callbacks: Callbacks = None,
        *,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, Any], None] = None,
        include_run_info: bool = False,
        additional_instruction: str = None,
        **kwargs,
    ) -> Coroutine[Any, Any, Dict[str, Any]]:
        """Asynchronously call the chain with inputs."""
        # override of __call__ so this can be run preinitialized by the decorator call
        kwargs["inputs"] = inputs
        kwargs["return_only_outputs"] = return_only_outputs
        kwargs["callbacks"] = callbacks
        kwargs["tags"] = tags
        kwargs["metadata"] = metadata
        kwargs["include_run_info"] = include_run_info
        self._additional_instruction = additional_instruction
        if self.default_call_kwargs:
            for k, v in self.default_call_kwargs.items():
                if (
                    k != "llm_kwargs"
                    and kwargs.get(k, None) is None
                    and k in self.default_call_kwargs
                ):
                    kwargs[k] = v

        try:
            result = await super().acall(**kwargs)
        except RequestRetryWithFeedback as e:
            if self._is_retry == True:
                raise Exception(e.feedback)
            self._is_retry = True
            self._additional_instruction = e.feedback
            result = await super().acall(**kwargs)
        self._additional_instruction = None
        return result

    def execute(self, **kwargs):
        """Execute the chain and return outputs"""
        print_log(
            log_object=f"> Entering {self.name} prompt decorator chain",
            log_level=self.prompt_type.log_level,
            color=LogColors.WHITE_BOLD,
        )

        result_data = self.__call__(**kwargs)

        result = result_data[self.output_key]
        try:
            result = self.postprocess_outputs(result_data, result)
        except OutputParserExceptionWithOriginal as e:
            if self.allow_retries:
                _kwargs = (
                    {**self.default_call_kwargs} if self.default_call_kwargs else {}
                )
                _kwargs.update(kwargs)
                retryChain, call_kwargs = self._get_retry_parse_call_args(
                    self.prompt, e, lambda: self.prompt.format(**_kwargs["inputs"])
                )
                result = retryChain.predict(**call_kwargs)
                print_log(
                    log_object=f"\nResult:\n{result}",
                    log_level=self.prompt_type.log_level if not self.verbose else 100,
                    color=(
                        self.prompt_type.color if self.prompt_type else LogColors.BLUE
                    ),
                )
                return self.postprocess_outputs(result_data, result)
            else:
                raise e

        print_log(
            log_object=f"> Finished chain",
            log_level=self.prompt_type.log_level,
            color=LogColors.WHITE_BOLD,
        )
        return result

    async def aexecute(self, **kwargs):
        """Execute the chain and return outputs"""
        print_log(
            log_object=f"> Entering {self.name} prompt decorator chain",
            log_level=self.prompt_type.log_level,
            color=LogColors.WHITE_BOLD,
        )

        try:
            result_data = await self.acall(**kwargs)

            result = result_data[self.output_key]

            result = self.postprocess_outputs(result_data, result)
        except RequestRetryWithFeedback as e:
            if self._is_retry == True:
                raise Exception(e.feedback)
            self._is_retry = True
            result_data = await self.acall(**kwargs, additional_instruction=e.feedback)

            result = result_data[self.output_key]

            result = self.postprocess_outputs(result_data, result)
        except OutputParserExceptionWithOriginal as e:
            if self.allow_retries:
                _kwargs = (
                    {**self.default_call_kwargs} if self.default_call_kwargs else {}
                )
                _kwargs.update(kwargs)
                retryChain, call_kwargs = self._get_retry_parse_call_args(
                    self.prompt, e, lambda: self.prompt.format(**_kwargs["inputs"])
                )
                result = await retryChain.apredict(**call_kwargs)
                print_log(
                    log_object=f"\nResult:\n{result}",
                    log_level=self.prompt_type.log_level if not self.verbose else 100,
                    color=(
                        self.prompt_type.color if self.prompt_type else LogColors.BLUE
                    ),
                )
                return self.postprocess_outputs(result_data, result)
            else:
                raise e

        print_log(
            log_object=f"> Finished chain",
            log_level=self.prompt_type.log_level,
            color=LogColors.WHITE_BOLD,
        )
        return result

    def _get_retry_parse_call_args(
        self,
        prompt_template: PromptDecoratorTemplate,
        exception: OutputParserExceptionWithOriginal,
        get_original_prompt: Callable,
    ):
        logging.warning(
            msg=f"Failed to parse output for {self.name}: {exception}\nRetrying..."
        )
        if (
            hasattr(self.prompt, "template_string")
            and self.format_instructions_parameter_key
            not in self.prompt.template_string
        ):
            logging.warning(
                f"Please note that we didn't find a {self.format_instructions_parameter_key} parameter in the prompt string. If you don't include it in your prompt template, you need to provide your custom formatting instructions."
            )
        if exception.original_prompt_needed_on_retry:
            original_prompt = get_original_prompt()
        else:
            original_prompt = ""
        retry_parse_template = PromptTemplate.from_template(
            "{original_prompt}This is our original response {original} but it's not in correct format, please convert it into following format:\n{format_instructions}\n\nIf the response doesn't seem to be relevant to the expected format instructions, return 'N/A'"
        )
        register_prompt_template("retry_parse_template", retry_parse_template)

        retryChain = LLMChain(llm=self.llm, prompt=retry_parse_template)
        format_instructions = prompt_template.output_parser.get_format_instructions()
        if not format_instructions:
            raise Exception(
                f"Failed to get format instructions for {self.name} from output parser {prompt_template.output_parser}."
            )
        call_kwargs = {
            "original_prompt": original_prompt,
            "original": exception.original,
            "format_instructions": format_instructions,
        }
        return retryChain, call_kwargs

    def postprocess_outputs(self, result_data, result):
        log_results(
            result_data,
            result,
            is_function_call=False,
            verbose=self.verbose,
            prompt_type=self.prompt_type,
        )
        if self.prompt.output_parser:
            if result:
                try:
                    result = self.prompt.output_parser.parse(result)
                except:
                    result = (
                        False if result and "yes" in result.lower() else False
                    )  # usually its something like "Im sorry..."
        return result

    def select_llm(self, prompts, inputs=None):
        if self.llm_selector:
            # we pick the right LLM based on the first prompt
            first_prompt = prompts[0]
            if isinstance(first_prompt, ChatPromptValue):
                llm = self.llm_selector.get_llm(
                    first_prompt.messages, **self._additional_llm_selector_args(inputs)
                )
            elif isinstance(first_prompt, str):
                self.llm_selector.get_llm(
                    first_prompt, **self._additional_llm_selector_args(inputs)
                )
            else:
                llm = self.llm_selector.get_llm(
                    first_prompt.to_string(),
                    **self._additional_llm_selector_args(inputs),
                )
        else:
            llm = self.llm
        return llm

    def _additional_llm_selector_args(self, inputs):
        return {
            "expected_generated_tokens": self.expected_gen_tokens,
            "streaming": self.capture_stream,
            "llm_selector_rule_key": self.llm_selector_rule_key,
        }

    def __should_use_json_response_format(self, llm):
        if "OpenAI" in type(llm).__name__:
            if (
                llm.model_name in MODELS_WITH_JSON_FORMAT_SUPPORT
                and self.prompt.output_parser
                and (
                    self.prompt.output_parser._type == "json"
                    or self.prompt.output_parser._type == "pydantic"
                )
            ):
                return True

    def generate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> LLMResult:
        """Generate LLM result from inputs."""
        prompts, stop = self.prep_prompts(input_list, run_manager=run_manager)
        llm = self.select_llm(prompts, input_list[0])
        callbacks = run_manager.get_child() if run_manager else None
        additional_kwargs = self.llm_kwargs or {}
        if self.__should_use_json_response_format(llm):
            additional_kwargs["response_format"] = {"type": "json_object"}

        if isinstance(llm, BaseLanguageModel):
            return llm.generate_prompt(
                prompts,
                stop,
                callbacks=callbacks,
                **additional_kwargs,
            )

        else:
            results = llm.bind(stop=stop, **additional_kwargs).batch(
                cast(List, prompts), {"callbacks": callbacks}
            )
            generations: List[List[Generation]] = []
            for res in results:
                if isinstance(res, BaseMessage):
                    generations.append([ChatGeneration(message=res)])
                else:
                    generations.append([Generation(text=res)])
            return LLMResult(generations=generations)

    async def agenerate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> LLMResult:
        """Generate LLM result from inputs."""
        prompts, stop = await self.aprep_prompts(input_list, run_manager=run_manager)
        llm = self.select_llm(prompts, input_list[0])
        callbacks = run_manager.get_child() if run_manager else None
        additional_kwargs = self.llm_kwargs or {}
        if self.__should_use_json_response_format(llm):
            additional_kwargs["response_format"] = {"type": "json_object"}
        if self.llm_kwargs:
            additional_kwargs.update(self.llm_kwargs)
        if isinstance(llm, BaseLanguageModel):
            return await llm.agenerate_prompt(
                prompts,
                stop,
                callbacks=callbacks,
                **additional_kwargs,
            )

        else:
            results = await llm.bind(stop=stop, **additional_kwargs).abatch(
                cast(List, prompts), {"callbacks": callbacks}
            )
            generations: List[List[Generation]] = []
            for res in results:
                if isinstance(res, BaseMessage):
                    generations.append([ChatGeneration(message=res)])
                else:
                    generations.append([Generation(text=res)])
            return LLMResult(generations=generations)


class LLMDecoratorChainWithFunctionSupport(LLMDecoratorChain):

    functions: Union[FunctionsProvider, List[Union[Callable, BaseTool]]]
    func_name_map: dict = None

    function_call_output_key: str = "function_call_info"
    function_output_key: str = "function"
    message_output_key: str = "message"
    _is_retry: Optional[str] = PrivateAttr(default=False)

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [
            self.output_key,
            self.function_output_key,
            self.function_call_output_key,
        ]

    def postprocess_outputs(self, result_data, result):
        log_results(
            result_data,
            result,
            bool(self.functions.functions),
            self.verbose,
            self.prompt_type,
        )

        if self.prompt.output_parser:
            if isinstance(
                self.prompt.output_parser, OpenAIFunctionsPydanticOutputParser
            ):
                # it the output parser is OpenAIFunctionsPydanticOutputParser, it means we should return the regular result, since we've used functions only for structure calling
                # there is no result probably, but if there is, we ignore it... we are interested only in tha data in function_call_info
                result = self.prompt.output_parser.parse(
                    result_data["function_call_info"]["arguments"]
                )
                # we dont want to return  OutputWithFunctionCall in this case
                # TODO: Hardcoded for now...
                return result
            else:
                if result:
                    result = self.prompt.output_parser.parse(result)

        return self._generate_output_with_function_call(result, result_data)

    @root_validator(pre=True)
    def validate_and_prepare_chain(cls, values):
        functions = values.get("functions", None)
        llm = values.get("llm", None)
        if isinstance(functions, list):
            values["functions"] = FunctionsProvider(functions)
        elif isinstance(functions, FunctionsProvider):
            values["functions"] = functions
        elif functions:
            raise ValueError(
                f"functions must be a List[Callable|BaseTool] or FunctionsProvider instance. Got: {functions.__class__}"
            )

        if not llm:
            raise ValueError("llm must be defined")

        # if not "OpenAI" in type(llm).__name__ and (
        #     CachedChatLLM and not isinstance(llm, CachedChatLLM)
        # ):
        #     raise ValueError(f"llm must be a ChatOpenAI instance. Got: {llm}")

        return values

    def get_final_function_schemas(self, inputs):
        return self.functions.get_function_schemas(inputs)

    def _additional_llm_selector_args(self, inputs):
        args = super()._additional_llm_selector_args(inputs)
        args["function_schemas"] = self.get_final_function_schemas(inputs)
        return args

    def preprocess_inputs(self, input_list):
        additional_kwargs = self.llm_kwargs or {}

        final_function_schemas = None
        if self.functions:
            if self.memory is not None:
                # we are sending out more outputs... memory expects only one (AIMessage... so let's set it, becasue user has no way to know these internals)
                if hasattr(self.memory, "output_key") and not self.memory.output_key:
                    self.memory.output_key = "message"
            if len(input_list) != 1:
                raise ValueError("Only one input is allowed when using functions")
            if "function_call" in input_list[0]:
                for input in input_list:
                    function_call = input.pop("function_call")
                # function call should be only one... and the same for all inputs... there shouldn't be more anyway
                if not isinstance(function_call, str):
                    f_name = next(
                        (
                            f_name
                            for f_name, func in self.functions.func_name_map.items()
                            if func == function_call
                        ),
                        None,
                    )
                    if not f_name:
                        raise ValueError(
                            f"Invalid function call. Function {function_call} is not defined in this chain"
                        )
                    function_call = {"name": f_name}
                elif function_call not in ["none", "auto"]:
                    # test if it's a valid function name
                    self.get_function(function_call)

                    function_call = {"name": function_call}

                additional_kwargs["function_call"] = function_call
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
        llm: BaseChatModel = self.select_llm(prompts, input_list[0])
        callbacks = run_manager.get_child() if run_manager else None

        def run(additional_instruction: str = None):
            if final_function_schemas:
                additional_kwargs["functions"] = final_function_schemas
            if not isinstance(prompts, ChatPromptValue):

                if isinstance(llm, BaseLanguageModel):
                    return llm.generate_prompt(
                        prompts,
                        stop,
                        callbacks=callbacks,
                        **additional_kwargs,
                    )

                else:
                    results = llm.bind(stop=stop, **additional_kwargs).batch(
                        cast(List, prompts), {"callbacks": callbacks}
                    )
                    generations: List[List[Generation]] = []
                    for res in results:
                        if isinstance(res, BaseMessage):
                            generations.append([ChatGeneration(message=res)])
                        else:
                            generations.append([Generation(text=res)])
                    return LLMResult(generations=generations)

        try:
            return run(additional_instruction=self._additional_instruction)
        except RequestRetryWithFeedback as e:
            if not isinstance(prompts, ChatPromptValue):
                raise  # supported only for chat
            if not self._is_retry == True:
                self._is_retry = True
                return run(self._additional_instruction)
            else:
                raise Exception(e.feedback)

    async def agenerate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> LLMResult:
        """Generate LLM result from inputs."""
        additional_kwargs, final_function_schemas = self.preprocess_inputs(input_list)

        prompts, stop = await self.aprep_prompts(input_list, run_manager=run_manager)
        llm: BaseChatModel = self.select_llm(prompts, input_list[0])
        callbacks = run_manager.get_child() if run_manager else None

        async def arun(additional_instruction: str = None):
            if final_function_schemas:
                additional_kwargs["functions"] = final_function_schemas
            if not isinstance(prompts, ChatPromptValue):

                if isinstance(llm, BaseLanguageModel):
                    return await llm.agenerate_prompt(
                        prompts,
                        stop,
                        callbacks=callbacks,
                        **additional_kwargs,
                    )

                else:
                    results = await llm.bind(stop=stop, **additional_kwargs).abatch(
                        cast(List, prompts), {"callbacks": callbacks}
                    )
                    generations: List[List[Generation]] = []
                    for res in results:
                        if isinstance(res, BaseMessage):
                            generations.append([ChatGeneration(message=res)])
                        else:
                            generations.append([Generation(text=res)])
                    return LLMResult(generations=generations)

        try:
            return await arun(additional_instruction=self._additional_instruction)
        except RequestRetryWithFeedback as e:
            if not isinstance(prompts, ChatPromptValue):
                raise  # supported only for chat
            if not self._is_retry == True:
                self._is_retry = True
                return await arun(self._additional_instruction)
            else:
                raise Exception(e.feedback)

    def _create_output(self, generation):
        res = {
            self.output_key: generation.text,
            self.function_call_output_key: None,
            self.function_output_key: None,
        }
        if isinstance(generation, ChatGeneration):
            res[self.message_output_key] = generation.message
            # let's make a copy of the function call so that we don't modify the original
            function_call = (
                dict(generation.message.additional_kwargs.get("function_call"))
                if generation.message.additional_kwargs
                else {}
            )
            if function_call:
                if isinstance(function_call["arguments"], str):
                    if function_call["name"] not in self.functions.func_name_map:
                        raise RequestRetryWithFeedback(
                            feedback=f"invalid function '{function_call['name']}', make sure to use only one of these functions: '{', '.join(self.functions.func_name_map.keys())}'"
                        )
                    try:
                        function_call["arguments"] = json.loads(
                            function_call["arguments"]
                        )
                    except json.JSONDecodeError:
                        raise RequestRetryWithFeedback(
                            feedback="(function arguments  have to be a valid JSON)"
                        )

            if (
                generation.message.additional_kwargs
                and generation.message.additional_kwargs.get("function_call")
            ):
                res[self.function_call_output_key] = function_call
                try:
                    function = (
                        self.get_function(function_call["name"])
                        if function_call
                        else None
                    )
                except KeyError:
                    print_log(
                        f"LLM requested function {function_call['name']} which is not defined! Retrying",
                        log_level=logging.WARNING,
                    )
                    valid_func_names = ", ".join(self.functions.func_name_map.keys())
                    raise RequestRetryWithFeedback(
                        feedback=f"(I need to make sure to use only valid functions... from the list: {valid_func_names})"
                    )
                res[self.function_output_key] = function
        return res

    def get_function(self, function_name):
        return self.functions.get_function(function_name)

    def create_outputs(self, response: LLMResult) -> List[Dict[str, str]]:
        """Create outputs from response."""

        return [
            self._create_output(generation[0]) for generation in response.generations
        ]

    def _generate_output_with_function_call(self, result: Any, result_data: dict):
        """get parsed result, function call data from llm and list of functions and build  OutputWithFunctionCall"""
        # find the function first:

        _function = result_data["function"]
        if result_data.get("function_call_info"):
            _tool_arguments = result_data["function_call_info"]["arguments"]
            if isinstance(_function, BaseTool):
                # langchain hack >> "__arg1" as a single argument hack
                _is_single_arg_hack = (
                    "__arg1" in _tool_arguments and len(_tool_arguments) == 1
                )
                tool_input = (
                    _tool_arguments["__arg1"]
                    if _is_single_arg_hack
                    else _tool_arguments
                )
                _tool_arguments = tool_input

                def _sync_function(arguments=tool_input):
                    return _function.run(
                        tool_input=arguments,
                        verbose=self.verbose,
                        callbacks=self.callbacks,
                    )

                async def _async_function(arguments=tool_input):
                    return await _function.arun(
                        tool_input=arguments,
                        verbose=self.verbose,
                        callbacks=self.callbacks,
                    )

            elif callable(_function):
                # TODO: add support for verbose and callbacks

                is_async = inspect.iscoroutinefunction(_function)

                if is_async:
                    _async_function = _function
                    _sync_function = None
                else:
                    _sync_function = _function
                    _async_function = None
            else:
                raise TypeError(
                    f"Invalid function type: {_function} of type {type(_function)}"
                )

            return OutputWithFunctionCall(
                output=result,
                output_text=result_data["text"],
                output_message=result_data["message"],
                function=_sync_function,
                function_async=_async_function,
                function_name=result_data["function_call_info"]["name"],
                function_args=result_data["function_call_info"]["arguments"],
                function_arguments=_tool_arguments,
            )
        else:
            return OutputWithFunctionCall(
                output=result,
                output_message=result_data["message"],
                output_text=result_data["text"],
            )


class FollowupHandle(BaseCallbackHandler):

    def __init__(self) -> None:
        self.last_prompts = None
        self.last_messages = None
        self.last_response_generations = None
        self.last_inputs = None
        self.chain: LLMDecoratorChain = None

    def reset(self):
        self.last_prompts = None
        self.last_messages = None
        self.last_response_generations = None
        self.last_inputs = None
        self.chain = None

    def bind_to_chain(self, chain: LLMDecoratorChain) -> None:
        """Bind callback handler to chain."""
        if self.chain is not None:
            raise Exception("FollowupHandle is already bound to a chain.")
        self.chain = chain

    @property
    def is_bound(self) -> bool:
        """Whether callback handler is bound to a chain."""
        return self.chain is not None

    @property
    def ignore_llm(self) -> bool:
        """Whether to ignore LLM callbacks."""
        return False

    @property
    def ignore_chain(self) -> bool:
        """Whether to ignore chain callbacks."""
        return False

    @property
    def ignore_agent(self) -> bool:
        """Whether to ignore agent callbacks."""
        return True

    @property
    def ignore_retriever(self) -> bool:
        """Whether to ignore retriever callbacks."""
        return True

    @property
    def ignore_chat_model(self) -> bool:
        """Whether to ignore chat model callbacks."""
        return False

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], *args, **kwargs
    ) -> Any:
        self.last_inputs = inputs

    def on_chat_model_start(
        self, serialized: dict, messages: List[List[BaseMessage]], *args, **kwargs
    ):
        if len(messages) != 1:
            raise Exception(
                f"Invalid messages length {len(messages)}. FollowupHandle only supports one prompt at a time."
            )
        self.last_messages = messages
        self.last_prompts = None

    def on_llm_start(self, serialized: dict, prompts: List[str], *args, **kwargs):
        if len(prompts) != 1:
            raise Exception(
                f"Invalid prompts length {len(prompts)}. FollowupHandle only supports one prompt at a time."
            )
        self.last_prompts = prompts
        self.last_messages = None

    def on_llm_end(self, response: LLMResult, *args, **kwargs) -> None:
        self.last_response_generations = response.generations

    def _prepare_followup_chain_with_args(
        self, followup_content: Union[str, BaseMessage], with_functions: bool
    ):
        if self.last_response_generations is None:
            raise Exception(
                "No response from LLM yet. Can't followup before the prompt has been executed"
            )
        if len(self.last_response_generations) != 1:
            raise Exception(
                f"Invalid response generations length {len(self.last_response_generations)}. FollowupHandle only supports one generated response"
            )

        llm = self.chain.select_llm(
            self.last_prompts or [ChatPromptValue(messages=self.last_messages[0])],
            self.last_inputs,
        )

        if self.last_messages:
            msg_list = self.last_messages[0]
            last_response_msg = self.last_response_generations[0][0].message
            msg_list.append(last_response_msg)
            if isinstance(followup_content, str):
                followup_content = HumanMessage(content=followup_content)

            msg_list.append(followup_content)
            new_prompt = ChatPromptValue(messages=msg_list)
        elif self.last_prompts:
            new_prompt = StringPromptValue(
                self.last_prompts[0]
                + self.last_response_generations[0][0].text
                + "\n"
                + followup_content
            )
        else:
            raise Exception("Last generation has not been recorded")

        if with_functions and not isinstance(
            self.chain, LLMDecoratorChainWithFunctionSupport
        ):
            raise Exception(
                "followup can only by used with functions if the the original llm_prompt was called with functions"
            )
        kwargs = {
            "prompts": [new_prompt],
            "stop": None,
            "callbacks": self.chain.callbacks,
        }
        if with_functions:
            kwargs["functions"] = self.chain.get_final_function_schemas(
                self.last_inputs
            )

        return llm, kwargs

    def _process_llm_output(self, llm_result, with_functions, with_output_parser):
        generation = llm_result.generations[0][0]
        if with_output_parser:
            result = with_output_parser.parse(generation.text)
        else:
            if self.chain.prompt.output_parser:
                result = self.chain.prompt.output_parser.parse(generation.text)
            else:
                result = generation.text
        if isinstance(generation, ChatGeneration):
            if with_functions:
                results_data = self.chain.create_outputs(llm_result)

                self.chain._generate_output_with_function_call(
                    result, result_data=results_data[0] if results_data else None
                )
                return self.chain.postprocess_outputs(result, results_data[0])
        else:
            if with_functions:
                raise Exception("LLM does not support functions")

        return result

    def followup(
        self,
        followup_content: Union[str, BaseMessage],
        with_functions: bool = False,
        with_output_parser: BaseOutputParser = None,
    ) -> Union[str, OutputWithFunctionCall, Any]:

        llm, kwargs = self._prepare_followup_chain_with_args(
            followup_content, with_functions=with_functions
        )

        if isinstance(llm, BaseLanguageModel):
            result = llm.generate_prompt(**kwargs)
        else:
            run_result = llm.bind(stop=kwargs.pop("stop", None)).batch(
                cast(List, kwargs["prompts"]),
                {k: v for k, v in kwargs.items() if k not in ["stop", "prompts"]},
            )
            generations = []
            for res in run_result:
                if isinstance(res, BaseMessage):
                    generations.append([ChatGeneration(message=res)])
                else:
                    generations.append([Generation(text=res)])
            result = LLMResult(generations=generations)

        return self._process_llm_output(result, with_functions, with_output_parser)

    async def afollowup(
        self,
        followup_content: Union[str, BaseMessage],
        with_functions: bool = False,
        with_output_parser: BaseOutputParser = None,
    ) -> Union[str, OutputWithFunctionCall, Any]:
        llm, kwargs = self._prepare_followup_chain_with_args(
            followup_content, with_functions=with_functions
        )
        if isinstance(llm, BaseLanguageModel):
            result = await llm.agenerate_prompt(**kwargs)
        else:
            run_result = await llm.bind(stop=kwargs.pop("stop", None)).abatch(
                cast(List, kwargs["prompts"]),
                {k: v for k, v in kwargs.items() if k not in ["stop", "prompts"]},
            )
            generations = []
            for res in run_result:
                if isinstance(res, BaseMessage):
                    generations.append([ChatGeneration(message=res)])
                else:
                    generations.append([Generation(text=res)])
            result = LLMResult(generations=generations)

        return self._process_llm_output(result, with_functions, with_output_parser)


class RequestRetryWithFeedback(Exception):

    def __init__(self, feedback: str = None):
        super().__init__()
        self.feedback = feedback


def log_results(
    result_data, result, is_function_call=False, verbose=False, prompt_type=None
):
    if verbose or prompt_type:
        if not prompt_type:
            prompt_type = PromptTypes.UNDEFINED
        print_log(
            log_object=f"\nResult:\n{result}",
            log_level=prompt_type.log_level if not verbose else 100,
            color=prompt_type.color if prompt_type else LogColors.BLUE,
        )
        if is_function_call:
            function_call_info_str = json.dumps(
                result_data.get("function_call_info"), indent=4
            )
            print_log(
                log_object=f"\nFunction call:\n{function_call_info_str}",
                log_level=prompt_type.log_level if not verbose else 100,
                color=prompt_type.color if prompt_type else LogColors.BLUE,
            )
