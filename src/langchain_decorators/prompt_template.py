import logging
import re
import inspect
from abc import ABC, abstractmethod
from string import Formatter

from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    TypeVar,
    Generic,
    List,
)


from langchain.prompts import StringPromptTemplate, PromptTemplate
from langchain.prompts.chat import  MessagesPlaceholder, ChatMessagePromptTemplate, ChatPromptTemplate, ChatPromptValue
from langchain.schema import PromptValue, BaseOutputParser, BaseMemory, BaseChatMessageHistory

from .schema import OutputWithFunctionCall, PydanticListTypeWrapper
from .common import LogColors, PromptTypeSettings, get_func_return_type, get_function_docs, get_function_full_name, print_log
from .output_parsers import *

try:
    from promptwatch import register_prompt_template
except ImportError:
    register_prompt_template = lambda *args, **kwargs: None

import pydantic
if pydantic.__version__ <"2.0.0":
    from pydantic import BaseModel, Field
else:
    if USE_PYDANTIC_V1:
        from pydantic.v1 import BaseModel as BaseModelV1
        from pydantic.v1 import Field
    else:
        from pydantic import BaseModel, Field


class BaseTemplateBuilder(ABC):

    @abstractmethod
    def build_template(self, template_parts:List[Tuple[str,str]],kwargs:Dict[str,Any])->PromptTemplate:
        """ Function that builds a prompt template from a template string and the prompt block name (which is the the part of ```<prompt:$prompt_block_name> in the decorated function docstring)

        Args:
            template_parts (List[Tuple[str,str]]): list of prompt parts List[(prompt_block_name, template_string)]
            kwargs (Dict[str,Any]): all arguments passed to the decorated function

        Returns:
            PromptTemplate: ChatPromptTemplate or StringPromptTemplate
        """
        pass

class OpenAITemplateBuilder:

    def build_template(self, template_parts:List[Tuple[str,str]],kwargs:Dict[str,Any])->PromptTemplate:
        if len(template_parts)==1 and not template_parts[0][1]:
            template_string=template_parts[0][0]
            return PromptTemplate.from_template(template_string)
        else:
            message_templates=[]
            for template_string, prompt_block_name in template_parts:
                template_string=template_string.strip()
                content_template= PromptTemplate.from_template(template_string)
                if prompt_block_name=="placeholder":
                    message_templates.append(MessagesPlaceholder(variable_name=template_string.strip(" {}")))
                elif prompt_block_name:

                    if "[" in prompt_block_name and  prompt_block_name[-1]=="]":
                        i = prompt_block_name.find("[")
                        name = prompt_block_name[i+1:-1]
                        role=prompt_block_name[:i]
                    else:
                        name=None
                        role=prompt_block_name

                    if name:
                        additional_kwargs={"name":name}
                    elif role == "function":
                        raise Exception(f"Invalid function prompt block. function_name {name} is not set. Use this format: <prompt:function[function_name]>")
                    else:
                        additional_kwargs={}

                    message_templates.append(ChatMessagePromptTemplate(role=role,prompt=content_template,additional_kwargs=additional_kwargs))
            return ChatPromptTemplate(messages=message_templates)


def parse_prompts_from_docs(docs:str):
    prompts = []
    for i, prompt_block in enumerate(re.finditer(r"```[^\S\n]*<prompt(?P<role>:[\w| |\[|\]]+)?>\n(?P<prompt>.*?)\n```[ |\t\n]*", docs, re.MULTILINE | re.DOTALL)):
        role = prompt_block.group("role")
        prompt = prompt_block.group("prompt")
        # remove \ escape before ```
        prompt = re.sub(r"((?<=\s)\\(?=```))|^\\(?=```)", "",prompt, flags=re.MULTILINE )
        prompt.strip()
        if not role:
            if i>1:
                raise ValueError("Only one prompt can be defined in code block. If you intend to define messages, you need to specify a role.\nExample:\n```<prompt:role>\nFoo {bar}\n```")
            else:
                prompts.append(prompt)
        else:
            prompts.append((role[1:], prompt))
    if not prompts:
        # the whole document is a prompt
        prompts.append(docs.strip())

    return prompts


class PromptTemplateDraft(BaseModel):
    role:str=None
    input_variables:List[str]
    template:str
    partial_variables_builders:Optional[Dict[str, Callable[[dict], str]]]

    def finalize_template(self, input_variable_values:dict)->Union[MessagesPlaceholder, ChatMessagePromptTemplate, StringPromptTemplate]:
        if self.role=="placeholder":
            return self.template
        else:
            final_template_string = self.template
            if self.partial_variables_builders:

                for final_partial_key, partial_builder in self.partial_variables_builders.items():
                    final_partial_value = partial_builder(input_variable_values)
                    final_template_string=final_template_string.replace(f"{{{final_partial_key}}}",final_partial_value)

            return final_template_string


def build_template_drafts(template:str, format:str, role:str=None )->PromptTemplateDraft:
    partials_with_params={}

    if role !="placeholder" and format=="f-string-extra":
        optional_blocks_regex = list(re.finditer(r"\{\?(?P<optional_partial>.+?)(?=\?\})\?\}", template, re.MULTILINE | re.DOTALL))
        for optional_block in optional_blocks_regex:
            optional_partial = optional_block.group("optional_partial")
            partial_input_variables = {v for _, v, _, _ in Formatter().parse(optional_partial) if v is not None}

            if not partial_input_variables:
                raise ValueError(f"Optional partial {optional_partial} does not contain any optional variables. Didn't you forget to wrap your parameter in {{}}?")

            # replace  {} with [] and all other non-word characters with underscore
            partial_name = re.sub(
                r"[^\w\[\]]+", "_", optional_partial.replace("{", "[").replace("}", "]")
            )

            partials_with_params[partial_name] = (optional_partial, partial_input_variables)
            # replace optional partial with a placeholder
            template = template.replace(optional_block.group(0), f"{{{partial_name}}}")

        partial_builders = {} # partial_name: a function that takes in a dict of variables and returns a string...

        for partial_name, (partial, partial_input_variables) in partials_with_params.items():
            # create function that will render the partial if all the input variables are present. Otherwise, it will return an empty string...
            # it needs to be unique for each partial, since we check only for the variables that are present in the partial
            def partial_formatter(inputs, _partial=partial, _partial_input_variables=partial_input_variables):
                """ This will render the partial if all the input variables are present. Otherwise, it will return an empty string."""
                missing_param = next((param for param in _partial_input_variables if param not in inputs or not inputs[param]), None)
                if missing_param:
                    return ""
                else:
                    return _partial

            partial_builders[partial_name] = partial_formatter
    try:
        input_variables = [v for _, v, _, _ in Formatter().parse(template) if v is not None and v not in partials_with_params]
    except ValueError as e:
        raise ValueError(f"{e}\nError parsing template: \n```\n{template}\n```")
    for partial_name, (partial, partial_input_variables) in partials_with_params.items():
        input_variables.extend(partial_input_variables)

    input_variables=list(set(input_variables))

    if not partials_with_params:
        partials_with_params=None
        partial_builders=None
    if not role:
        return PromptTemplateDraft(input_variables=input_variables, template=template, partial_variables_builders=partial_builders)
    elif role=="placeholder":
        if len(input_variables)>1:
            raise ValueError(f"Placeholder prompt can only have one input variable, got {input_variables}")
        elif len(input_variables)==0:
            raise ValueError(f"Placeholder prompt must have one input variable, got none.")
        return PromptTemplateDraft(template=template,  input_variables=input_variables,  partial_variables_builders=partial_builders, role="placeholder")
    else:
        return PromptTemplateDraft(role=role, input_variables=input_variables, template=template, partial_variables_builders=partial_builders)


class PromptDecoratorTemplate(StringPromptTemplate):
    template_string:str
    prompt_template_drafts:Union[PromptTemplateDraft, List[PromptTemplateDraft]]
    template_name:str
    template_format:str
    optional_variables:List[str]
    default_values:Dict[str,Any]
    format_instructions_parameter_key:str
    template_version: Union[str, None] = None
    prompt_type: Union[PromptTypeSettings, None] = None
    original_kwargs: Union[dict, None] = None
    return_type: Type = None

    @property
    def OutputType(self):
        return self.return_type

    @classmethod 
    def build(
        cls,
        template_string: str,
        template_name: str,
        template_format: str = "f-string-extra",
        output_parser: Union[None, BaseOutputParser] = None,
        optional_variables: Optional[List[str]] = None,
        default_values: Optional[Dict[str, Any]] = None,
        format_instructions_parameter_key: str = "FORMAT_INSTRUCTIONS",
        template_version: str = None,
        prompt_type: PromptTypeSettings = None,
        original_kwargs: dict = None,
        return_type: Type = None,
    ) -> "PromptDecoratorTemplate":

        if template_format not in ["f-string","f-string-extra"]:
            raise ValueError(
                f"template_format must be one of [f-string, f-string-extra], got {template_format}"
            )

        prompts = parse_prompts_from_docs(template_string)

        if isinstance(prompts,list):
            prompt_template_drafts=[]
            input_variables=[]

        for prompt in prompts:
            if isinstance(prompt,str):
                prompt_template_drafts=build_template_drafts(prompt, format=template_format)
                input_variables=prompt_template_drafts.input_variables
                # there should be only one prompt if it's a string
                break
            else:
                (role, content_template)= prompt
                message_template = build_template_drafts(content_template, format=template_format, role=role)
                input_variables.extend(message_template.input_variables)
                prompt_template_drafts.append(message_template)

        return cls(
            input_variables=input_variables,  # defined in base
            output_parser=output_parser,  # defined in base
            prompt_template_drafts=prompt_template_drafts,
            template_name=template_name,
            template_version=template_version,
            template_string=template_string,
            template_format=template_format,
            optional_variables=optional_variables,
            default_values=default_values,
            format_instructions_parameter_key=format_instructions_parameter_key,
            prompt_type=prompt_type,
            return_type=return_type or str,
            original_kwargs=original_kwargs,
        )

    @classmethod 
    def from_func(cls, 
                  func:Union[Callable, Coroutine], 
                  template_name:str=None, 
                  template_version:str=None, 
                  output_parser:Union[str,None, BaseOutputParser]="auto", 
                  template_format:str = "f-string-extra",
                  format_instructions_parameter_key:str="FORMAT_INSTRUCTIONS",
                  prompt_type:PromptTypeSettings = None,
                  original_kwargs:dict=None
                  )->"PromptDecoratorTemplate":

        template_string = get_function_docs(func)  
        template_name=template_name or get_function_full_name(func)
        return_type = get_func_return_type(func)

        if original_kwargs.get("output_parser"):
            output_parser = original_kwargs.pop("output_parser")
        return_list = False
        if output_parser=="auto":
            if return_type==str or return_type==None:
                output_parser = "str"
            elif return_type==dict:
                output_parser = "json"
            elif return_type==list:
                return_list = True
                _, args = get_func_return_type(func, with_args=True)
                output_parser = "list"
                if args:
                    if issubclass(args[0],BaseModel):
                        output_parser = "pydantic"
                        return_type = args[0]
                    elif issubclass(args[0],dict):
                        return_type = args[0]
                        output_parser = "json"
                    elif issubclass(args[0],str):
                        return_type = args[0]
                        output_parser = "list"
                    else:
                        raise Exception(
                            f"Unsupported item type in annotation of {template_name} -> {return_type}[{args}]"
                        )
                else:
                    return_type = str
            elif return_type==bool:
                output_parser = "boolean"
            elif issubclass(return_type, OutputWithFunctionCall):
                output_parser = "str"
                return_type = str
            elif issubclass(return_type, BaseModel) or issubclass(
                return_type, BaseModelV1
            ):
                output_parser = PydanticOutputParser(model=return_type)
            else:
                raise Exception(f"Unsupported return type {return_type}")
        if isinstance(output_parser,str):
            if output_parser=="str":
                output_parser = None
            elif output_parser=="json":
                output_parser = JsonOutputParser()
            elif output_parser=="boolean":
                output_parser = BooleanOutputParser()
            elif output_parser=="markdown":
                if return_type and return_type!=dict:
                    raise Exception(f"Conflicting output parsing instructions. Markdown output parser only supports return type dict, got {return_type}.")
                else:
                    output_parser = MarkdownStructureParser()
            elif output_parser=="list":
                output_parser = ListOutputParser()
            elif  output_parser == "pydantic":
                if issubclass(return_type,BaseModel):
                    output_parser = PydanticOutputParser(
                        model=return_type, as_list=return_list
                    )
                elif return_type==None:
                    raise Exception(f"You must annotate the return type for pydantic output parser, so that we can infer the model")
                else:
                    raise Exception(f"Unsupported return type {return_type} for pydantic output parser")

                if return_list:
                    output_parser = PydanticOutputParser(model=return_type)
                    return_type = PydanticListTypeWrapper[return_type]
            elif output_parser=="functions":
                if not return_type:
                    raise Exception(f"You must annotate the return type for functions output parser, so that we can infer the model")
                elif not issubclass(return_type,OutputWithFunctionCall):
                    if issubclass(return_type,BaseModel):
                        output_parser = OpenAIFunctionsPydanticOutputParser(model=return_type)
                    else:
                        raise Exception(f"Functions output parser only supports return type pydantic models, got {return_type}")
                else:
                    output_parser=None
            else:
                raise Exception(f"Unsupported output parser {output_parser}")

        default_values = {k:v.default for k,v in inspect.signature(func).parameters.items() if v.default!=inspect.Parameter.empty}

        return cls.build(
            template_string=template_string,
            template_name=template_name,
            template_version=template_version,
            output_parser=output_parser,
            template_format=template_format,
            optional_variables=[*default_values.keys()],
            default_values=default_values,
            format_instructions_parameter_key=format_instructions_parameter_key,
            prompt_type=prompt_type,
            original_kwargs=original_kwargs,
            return_type=return_type,
        )

    def get_final_template(self, **kwargs: Any)->PromptTemplate:
        """Create Chat Messages."""

        prompt_type = self.prompt_type or PromptTypeSettings()

        if self.default_values:
            # if we have default values, we will use them to fill in missing values
            kwargs = {**self.default_values, **kwargs}

        kwargs = {k:v for k,v in kwargs.items() if v is not None}

        parts=[]
        if isinstance(self.prompt_template_drafts,list):

            for message_draft in self.prompt_template_drafts:
                msg_template_final_str = message_draft.finalize_template(kwargs)
                if msg_template_final_str: # skip empty messages / templates
                    parts.append((msg_template_final_str,message_draft.role))

        else:
            msg_template_final_str = self.prompt_template_drafts.finalize_template(kwargs)
            parts.append((msg_template_final_str,""))

        template = prompt_type.prompt_template_builder.build_template(parts, kwargs)
        template.output_parser = self.output_parser

        register_prompt_template(self.template_name, template, self.template_version)
        return template

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        if self.format_instructions_parameter_key in self.input_variables and  not kwargs.get(self.format_instructions_parameter_key)  and self.output_parser :
            # add format instructions to inputs
            kwargs[self.format_instructions_parameter_key] = self.output_parser.get_format_instructions()

        final_template = self.get_final_template(**kwargs)
        kwargs = {k:(v if v is not None else "" ) for k,v in  kwargs.items() if k in  final_template.input_variables}
        if isinstance(final_template,ChatPromptTemplate):

            for msg in list(final_template.messages):
                if isinstance(msg,MessagesPlaceholder):
                    if not kwargs.get(msg.variable_name):
                        kwargs[msg.variable_name] = []

        for key, value in list(kwargs.items()):
            if isinstance(value, BaseMemory):
                memory:BaseMemory = kwargs.pop(key)

                kwargs.update(memory.load_memory_variables(kwargs))
            elif isinstance(value, BaseChatMessageHistory):
                kwargs[key] = value.messages

        formatted =  final_template.format_prompt(**kwargs)
        if isinstance(formatted,ChatPromptValue):
            for msg in list(formatted.messages):
                if (
                    isinstance(msg.content, str)
                    and (not msg.content or not msg.content.strip())
                    and not msg.additional_kwargs
                ):
                    formatted.messages.remove(msg)
        self.on_prompt_formatted(formatted.to_string())

        return formatted

    def format(self, **kwargs: Any) ->str:
        formatted = self.get_final_template(**kwargs).format(**kwargs)
        self.on_prompt_formatted(formatted)
        return formatted

    def on_prompt_formatted(self, formatted:str):
        pass
