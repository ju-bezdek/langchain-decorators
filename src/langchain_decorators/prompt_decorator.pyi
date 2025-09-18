from typing import Any, Callable, TypeVar, overload, List, Optional, Union
from langchain.llms.base import BaseLanguageModel
from langchain.schema import BaseOutputParser
from .common import PromptTypeSettings, PromptTypes
from .schema import OutputWithFunctionCall

F = TypeVar("F", bound=Callable[..., Any])

@overload
# decorator used without arguments
def llm_prompt(__func: F) -> F: ...
@overload
# decorator used with keyword arguments
def llm_prompt(
    *,
    prompt_type: PromptTypeSettings = ...,  # type: ignore
    template_format: str = ...,
    output_parser: Union[str, None, BaseOutputParser] = ...,  # type: ignore
    stop_tokens: Optional[List[str]] = ...,  # type: ignore
    template_name: Optional[str] = ...,
    template_version: Optional[str] = ...,  # type: ignore
    capture_stream: Optional[bool] = ...,
    llm: Optional[BaseLanguageModel] = ...,
    format_instructions_parameter_key: str = ...,
    retry_on_output_parsing_error: bool = ...,  # type: ignore
    verbose: Optional[bool] = ...,  # type: ignore
    expected_gen_tokens: Optional[int] = ...,  # type: ignore
    llm_selector_rule_key: Optional[str] = ...,
    llm_selector: Any = ...,
    functions_source: Optional[str] = ...,
    memory_source: Optional[str] = ...,  # type: ignore
    control_kwargs: List[str] = ...,  # type: ignore
) -> Callable[[F], F]: ...
def llm_prompt(*args: Any, **kwargs: Any) -> Any: ...
