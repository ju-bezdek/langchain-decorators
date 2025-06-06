from contextvars import ContextVar
import inspect
import json
import logging
from functools import wraps
from pdb import run
from time import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Type, Union, cast

from .streaming_context import StreamingContext
from .llm_chat_session import LlmChatSession
from .common import deprecated
from langchain.tools.convert_to_openai import format_tool_to_openai_function
import httpx
import pydantic
from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)

from langchain_core.runnables import Runnable, RunnableConfig, RunnableBinding
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
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import (
    RunnableLambda,
    RunnableSequence,
    RunnableWithFallbacks,
)
from .pydantic_helpers import USE_PYDANTIC_V1

from .common import LlmSelector, LogColors, PromptTypes, PromptTypeSettings, print_log
from .output_parsers import (
    BaseOutputParser,
    OpenAIFunctionsPydanticOutputParser,
    OutputParserExceptionWithOriginal,
)
from .prompt_template import PromptDecoratorTemplate
from .schema import OutputWithFunctionCall, PydanticListTypeWrapper

import pydantic
if pydantic.__version__ <"2.0.0":
    from pydantic import BaseModel, PrivateAttr
    from pydantic import root_validator as model_validator

    model_validator_kwargs = {"pre": True}
else:
    if USE_PYDANTIC_V1:
        from pydantic.v1 import BaseModel, PrivateAttr
        from pydantic.v1 import root_validator as model_validator

        model_validator_kwargs = {"pre": True}
    else:
        from pydantic import BaseModel, PrivateAttr, model_validator

        model_validator_kwargs = {"mode": "before"}

# Backwards compatibility
from .llm_tool_use import ToolsProvider, get_function_schema

FunctionsProvider = ToolsProvider

CachedChatLLM = None
register_prompt_template = None


try:
    from promptwatch import CachedChatLLM, register_prompt_template
except ImportError:
    pass

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None
    pass


class LLMDecoratorChain(Runnable):
    name: str
    llm_selector: Optional[LlmSelector] = None
    """ Optional LLM selector to pick the right LLM for the job. """
    capture_stream: bool = False
    expected_gen_tokens: Optional[int] = None
    llm_selector_rule_key: Optional[str] = None
    allow_retries: bool = True
    format_instructions_parameter_key: str = "FORMAT_INSTRUCTIONS"
    default_config: RunnableConfig
    return_type: Optional[Type] = None

    prompt_type: PromptTypeSettings = PromptTypes.UNDEFINED
    default_call_kwargs: Optional[Dict[str, Any]] = None

    _is_retry: Optional[str] = PrivateAttr(default=False)

    def __init__(
        self,
        name: str,
        prompt: PromptDecoratorTemplate,
        llm: BaseLanguageModel,
        capture_stream: bool = False,
        expected_gen_tokens: Optional[int] = None,
        llm_selector: Optional[LlmSelector] = None,
        llm_selector_rule_key: Optional[str] = None,
        prompt_type: PromptTypeSettings = PromptTypes.UNDEFINED,
        default_call_kwargs: Optional[Dict[str, Any]] = None,
        allow_retries: bool = True,
        format_instructions_parameter_key: str = "FORMAT_INSTRUCTIONS",
        with_structured_output: Union[Type, Dict] = None,
        tools_provider: Optional[ToolsProvider] = None,
        return_type: Optional[Type] = None,
        default_config: RunnableConfig = None,
        **kwargs,
    ) -> None:
        """Initialize LLMDecoratorChain with prompt and LLM."""
        self.name = name
        self.prompt = prompt
        self.llm = llm
        self.capture_stream = capture_stream
        self.expected_gen_tokens = expected_gen_tokens
        self.llm_selector = llm_selector
        self.llm_selector_rule_key = llm_selector_rule_key
        self.prompt_type = prompt_type
        self.default_call_kwargs = default_call_kwargs
        self.allow_retries = allow_retries
        self.format_instructions_parameter_key = format_instructions_parameter_key
        self.with_structured_output = with_structured_output
        self.tools_provider = tools_provider
        self.return_type = return_type
        self.default_config: RunnableConfig = default_config or RunnableConfig()

    def __call__(
        self,
        inputs: Union[Dict[str, Any], Any] = None,
        config: Optional[RunnableConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Call the chain with inputs."""

        print_log(
            log_object=f"> Finished chain",
            log_level=self.prompt_type.log_level,
            color=LogColors.WHITE_BOLD,
        )
        formatted_prompt, llm = self._prepare_execution(inputs, **kwargs)

        result = llm.invoke(
            input=formatted_prompt, config=config or self.default_config, **kwargs
        )

        return result

    def invoke(self, inputs: dict, config=None, **kwargs):
        """Execute the chain and return outputs"""
        return self.__call__(
            inputs=inputs, config=config or self.default_config, **kwargs
        )

    async def ainvoke(self, inputs, config=None, **kwargs):
        """Call the chain with inputs."""
        print_log(
            log_object=f"> Entering {self.name} prompt decorator chain",
            log_level=self.prompt_type.log_level,
            color=LogColors.WHITE_BOLD,
        )

        formatted_prompt, llm = self._prepare_execution(inputs, **kwargs)

        result = await llm.ainvoke(
            input=formatted_prompt, config=config or self.default_config
        )

        print_log(
            log_object=f"> Finished chain",
            log_level=self.prompt_type.log_level,
            color=LogColors.WHITE_BOLD,
        )

        return result

    def find_and_apply_on_llm(
        self,
        runnable: Runnable,
        func: Callable[[Union[BaseLanguageModel, BaseChatModel]], Runnable],
    ) -> Runnable:
        """Find and apply a function on the LLM in the chain."""
        if isinstance(runnable, (BaseLanguageModel, RunnableWithFallbacks)):
            return func(runnable)
        elif isinstance(runnable, RunnableSequence):
            if isinstance(runnable.steps[0], BaseLanguageModel):
                runnable.steps[0] = func(runnable.steps[0])
            return runnable
        elif isinstance(runnable, RunnableBinding):
            if isinstance(runnable.bound, BaseLanguageModel):
                runnable.bound = func(runnable.bound)
            else:
                runnable.bound = self.find_and_apply_on_llm(runnable.bound, func=func)
        else:
            raise ValueError(
                f"Unsupported runnable type: {type(runnable)}. Expected BaseLanguageModel or LLMDecoratorChain."
            )

    def _prepare_execution(
        self,
        inputs: Union[Dict[str, Any], Any] = None,
        **kwargs,
    ):
        """Prepare execution by processing inputs and setting up the LLM chain."""
        _inputs = (
            self.default_call_kwargs.get("inputs", {})
            if self.default_call_kwargs
            else {}
        )
        if inputs:
            _inputs.update(inputs)

        if self.default_call_kwargs:
            for k, v in self.default_call_kwargs.items():
                if (
                    kwargs.get(k, None) is None
                    and k in self.default_call_kwargs
                    and k != "llm_kwargs"
                ):
                    kwargs[k] = v
        llm_kwargs = kwargs.get("llm_kwargs", {})
        kwargs.pop("llm_kwargs", None)
        kwargs.pop("inputs", None)

        formatted_prompt = self.prompt.format_prompt(**(_inputs or {}))
        tools_provider = self.tools_provider
        session = LlmChatSession.get_current_session()
        if session:
            tools_provider = self.tools_provider or session.tools_provider
            if isinstance(formatted_prompt, ChatPromptValue):

                formatted_prompt.messages = [
                    *formatted_prompt.messages,
                    *session.message_history,
                ]
            elif isinstance(formatted_prompt, StringPromptValue):
                formatted_prompt = ChatPromptValue(
                    messages=[
                        HumanMessage(content=formatted_prompt.to_string()),
                        *session.message_history,
                    ]
                )

        llm: BaseChatModel | Runnable = self.select_llm(formatted_prompt, _inputs)
        if self.capture_stream and StreamingContext.get_context():
            kwargs["stream"] = True
            self.default_config["callbacks"] = kwargs.pop("callbacks", [])
            self.default_config["callbacks"].append(
                StreamingContext.StreamingContextCallback()
            )

        if tools_provider:
            llm = self._bind_tools_to_llm(
                llm,
                tools_provider,
                _inputs,
                tool_choice=kwargs.get(
                    "tool_choice", kwargs.pop("function_call", None)
                ),
            )

        llm = llm.bind(**llm_kwargs, **kwargs)

        if self.with_structured_output:

            llm = llm.with_structured_output(
                self.with_structured_output, include_raw=True
            )

        llm = llm | RunnableLambda(func=self.process_llm_output)
        if (
            not self.with_structured_output
            and self.prompt.output_parser
            and not (session and session.suppress_output_parser)
        ):
            llm = llm | self.prompt.output_parser

        if not self.prompt_type:
            log_level = logging.DEBUG
        else:
            log_level = self.prompt_type.log_level

        print_log(
            f"Prompt:\n{formatted_prompt.to_string()}", log_level, LogColors.DARK_GRAY
        )

        return formatted_prompt, llm

    def _bind_tools_to_llm(
        self,
        llm,
        tools_provider: ToolsProvider,
        inputs: dict,
        tool_choice: Optional[str] = None,
        **kwargs,
    ):
        parallel_tool_calls = (
            False if issubclass(self.return_type, OutputWithFunctionCall) else True
        )

        if isinstance(llm, BaseChatModel):
            llm = llm.bind_tools(
                tools_provider.get_function_schemas(inputs),
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
            )
        elif isinstance(llm, Runnable):
            llm = self.find_and_apply_on_llm(
                llm,
                lambda x: x.bind_tools(
                    tools_provider.get_function_schemas(inputs),
                    tool_choice=tool_choice,
                    parallel_tool_calls=parallel_tool_calls,
                ),
            )

        return llm

    def select_llm(self, prompt, inputs=None):
        if self.llm_selector:
            # we pick the right LLM based on the first prompt
            if isinstance(prompt, ChatPromptValue):
                llm = self.llm_selector.get_llm(
                    prompt.messages, **self._additional_llm_selector_args(inputs)
                )
            elif isinstance(prompt, str):
                self.llm_selector.get_llm(
                    prompt, **self._additional_llm_selector_args(inputs)
                )
            else:
                llm = self.llm_selector.get_llm(
                    prompt.to_string(),
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

    def process_llm_output(self, llm_result: Any):
        """Process the output from the LLM."""
        output_message: AIMessage = None
        result = None
        result_str = None
        session = LlmChatSession.get_current_session()
        if isinstance(llm_result, LLMResult):
            generations = llm_result.generations[0]
            if len(generations) == 1:
                generation = generations[0]
                if isinstance(generation, ChatGeneration):
                    output_message = generation.message
                    result = generation.message.content
                else:
                    result = generation.text

            else:
                raise ValueError(f"Expected one generation, got {len(generations)}")
            result_str = result
        elif self.with_structured_output and isinstance(llm_result, dict):
            output_message = llm_result.get("raw")

            if session and session.suppress_output_parser:
                result = llm_result.get("raw", {}).get("content", "")
            else:
                if issubclass(self.with_structured_output, PydanticListTypeWrapper):
                    result = llm_result["parsed"].items
                else:
                    result = llm_result["parsed"]
            result_str = llm_result.get("raw")
        elif isinstance(llm_result, BaseMessage):
            output_message = llm_result
            result = llm_result.content
            result_str = result
        else:
            result = llm_result
            result_str = result

        if session and output_message:
            session.add_message(output_message)
        if issubclass(self.return_type, OutputWithFunctionCall) and self.tools_provider:
            # Backwards compatibility:
            function = None
            function_name = None
            function_arguments = None
            tool_call_id = None
            if output_message.tool_calls:
                if len(output_message.tool_calls) > 1:
                    raise ValueError(
                        "Unable to process parallel function calls... use LLMChatSession instead"
                    )
                tool_call = output_message.tool_calls[0]
                function = self.tools_provider.get_tool_by_name(tool_call["name"])
                function_name = tool_call["name"]
                function_arguments = tool_call.get("args", {})
                tool_call_id = tool_call.get("id", None)
            result = OutputWithFunctionCall(
                tool_call_id=tool_call_id,
                output_text=output_message.content,
                output=result,
                function_arguments=function_arguments,
                function=function,
                function_name=function_name,
                output_message=output_message,
            )
        print_log(
            log_object=(
                result_str
                + (
                    (
                        "\n---\nTool calls:\n"
                        + "\n".join([str(tc) for tc in output_message.tool_calls])
                    )
                    if output_message and output_message.tool_calls
                    else ""
                )
            ),
            log_level=logging.DEBUG,
            color=LogColors.GREEN,
        )
        return result


class FollowupHandle(BaseCallbackHandler):

    def __init__(self) -> None:

        self.chain: LLMDecoratorChain = None
        self.message_history: List[BaseMessage] = []

    def reset(self):
        self.chain = None
        self.message_history: List[BaseMessage] = []

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

    def on_llm_end(self, response: LLMResult, *args, **kwargs) -> None:
        """Handle end of LLM response."""
        if not self.is_bound:
            raise Exception("FollowupHandle is not bound to a chain.")

        if len(response.generations) == 1:
            self.message_history.append(response.generations[0][0].message)
        else:
            raise Exception(
                f"Invalid response generations length {len(response.generations)}. FollowupHandle only supports one generated response"
            )

    def _prepare_followup_chain_with_args(
        self, followup_content: Union[str, BaseMessage], with_tools: bool
    ):
        if isinstance(followup_content, str):
            followup_content = HumanMessage(content=followup_content)

        self.message_history.append(followup_content)

    def followup(
        self,
        followup_content: Union[str, BaseMessage],
        with_tools: bool = False,
        with_output_parser: BaseOutputParser = None,
        **kwargs,
    ) -> Union[str, OutputWithFunctionCall, Any]:
        if kwargs.get("with_functions") is not None:
            with_tools = kwargs.get("with_functions")
        if with_tools:
            tools = self.chain.tools_provider
        else:
            tools = None
        followup_content = (
            HumanMessage(content=followup_content)
            if isinstance(followup_content, str)
            else followup_content
        )
        with LlmChatSession(
            tools=tools,
            message_history=[*self.message_history, followup_content],
            copy_parent_context=False,
            suppress_output_parser=bool(with_output_parser),
        ) as session:
            res = self.chain.invoke(inputs=kwargs)
            self.message_history = session.message_history
            if with_output_parser:
                return with_output_parser(res)
            else:
                return res

    async def afollowup(
        self,
        followup_content: Union[str, BaseMessage],
        with_tools: bool = False,
        with_output_parser: BaseOutputParser = None,
        **kwargs,
    ) -> Union[str, OutputWithFunctionCall, Any]:
        if kwargs.get("with_functions") is not None:
            with_tools = kwargs.get("with_functions")
        if with_tools:
            tools = self.chain.tools_provider
        else:
            tools = None
        followup_content = (
            HumanMessage(content=followup_content)
            if isinstance(followup_content, str)
            else followup_content
        )
        with LlmChatSession(
            tools=tools,
            message_history=[*self.message_history, followup_content],
            copy_parent_context=False,
            suppress_output_parser=bool(with_output_parser),
        ) as session:
            res = await self.chain.ainvoke(inputs=kwargs)
            self.message_history = session.message_history
            if with_output_parser:
                return with_output_parser(res)
            else:
                return res


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
