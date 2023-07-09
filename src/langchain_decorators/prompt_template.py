
import logging
import re
import inspect

from string import Formatter

from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

from pydantic import BaseModel

from textwrap import dedent

from langchain import PromptTemplate
from langchain.prompts import StringPromptTemplate
from langchain.prompts.chat import  MessagesPlaceholder, ChatMessagePromptTemplate, ChatPromptTemplate, ChatPromptValue
from langchain.schema import PromptValue, BaseOutputParser, BaseMemory, BaseChatMessageHistory

from promptwatch import register_prompt_template
from .schema import OutputWithFunctionCall
from .common import LogColors, PromptTypeSettings, get_func_return_type, get_function_docs, get_function_full_name, print_log
from .output_parsers import *


def parse_prompts_from_docs(docs:str):
    prompts = []
    for i, prompt_block in enumerate(re.finditer(r"```[^\S\n]*<prompt(?P<role>:\w+)?>\n(?P<prompt>.*?)\n```[ |\t]*\n", docs, re.MULTILINE | re.DOTALL)):
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
            return MessagesPlaceholder(variable_name=self.input_variables[0])
        else:
            final_template_string = self.template
            if self.partial_variables_builders:
                
                for final_partial_key, partial_builder in self.partial_variables_builders.items():
                    final_partial_value = partial_builder(input_variable_values)
                    final_template_string=final_template_string.replace(f"{{{final_partial_key}}}",final_partial_value)
        
            final_template_string=final_template_string.strip()
            content_template= PromptTemplate.from_template(final_template_string)
            if self.role:
                return ChatMessagePromptTemplate(role=self.role,prompt=content_template)
            else:
                return content_template
            

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
            partial_name = re.sub(r"[^\w\[\]]+", "_", optional_partial.replace("{","[").replace("}","]"))
            

            partials_with_params[partial_name] = (optional_partial, partial_input_variables)
            # replace optional partial with a placeholder
            template = template.replace(optional_block.group(0), f"{{{partial_name}}}")

        partial_builders = {} # partial_name: a function that takes in a dict of variables and returns a string...
        
        
                
        for partial_name, (partial, partial_input_variables) in partials_with_params.items():
            # create function that will render the partial if all the input variables are present. Otherwise, it will return an empty string... 
            # it needs to be unique for each partial, since we check only for the variables that are present in the partial
            def partial_formatter(inputs, _partial=partial):
                """ This will render the partial if all the input variables are present. Otherwise, it will return an empty string."""
                missing_param = next((param for param in partial_input_variables if param not in inputs or not inputs[param]), None)
                if missing_param:
                    return ""
                else:
                    return _partial
            
            partial_builders[partial_name] = partial_formatter

    input_variables = [v for _, v, _, _ in Formatter().parse(template) if v is not None and v not in partials_with_params]
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
    optional_variables_none_behavior:str
    default_values:Dict[str,Any]
    format_instructions_parameter_key:str
    template_version:str=None
    prompt_type:PromptTypeSettings = None

    

    

    @classmethod 
    def build(cls, 
              template_string:str, 
              template_name:str,
              template_format:str="f-string-extra", 
              output_parser:Union[None, BaseOutputParser]=None, 
              optional_variables:Optional[List[str]]=None,
              optional_variables_none_behavior:str="skip_line", 
              default_values:Optional[Dict[str,Any]]=None,
              format_instructions_parameter_key:str="FORMAT_INSTRUCTIONS",
              template_version:str=None,
              prompt_type:PromptTypeSettings = None
            )->"PromptDecoratorTemplate":
            
        if template_format not in ["f-string","f-string-extra"]:
            raise ValueError(f"template_format must be one of [f-string, f-string-extra], got {template_format}")
        

        prompts = parse_prompts_from_docs(template_string)
        
        if isinstance(prompts,list):
            prompt_template_drafts=[]
            input_variables=[]
            
        for prompt in prompts:
            if isinstance(prompt,str):
                prompt_template_drafts=build_template_drafts(prompt, format=template_format)
                input_variables=prompt_template_drafts.input_variables
                #there should be only one prompt if it's a string
                break
            else:
                (role, content_template)= prompt
                message_template = build_template_drafts(content_template, format=template_format, role=role)
                input_variables.extend(message_template.input_variables)
                prompt_template_drafts.append(message_template)
        



        return cls(
                input_variables=input_variables, #defined in base
                output_parser=output_parser,#defined in base
                prompt_template_drafts=prompt_template_drafts,
                template_name=template_name,
                template_version=template_version,
                template_string=template_string,
                template_format=template_format,
                optional_variables=optional_variables,
                optional_variables_none_behavior=optional_variables_none_behavior,
                default_values=default_values,
                format_instructions_parameter_key=format_instructions_parameter_key,
                prompt_type=prompt_type
            )
        
    @classmethod 
    def from_func(cls, 
                  func:Union[Callable, Coroutine], 
                  template_name:str=None, 
                  template_version:str=None, 
                  output_parser:Union[str,None, BaseOutputParser]="auto", 
                  template_format:str = "f-string-extra",
                  format_instructions_parameter_key:str="FORMAT_INSTRUCTIONS",
                  prompt_type:PromptTypeSettings = None
                  )->"PromptDecoratorTemplate":
        
        template_string = get_function_docs(func)  
        template_name=template_name or get_function_full_name(func)
        return_type = get_func_return_type(func)
      
        if output_parser=="auto":
            if return_type==str or return_type==None:
                output_parser = "str"
            elif return_type==dict:
                output_parser = "json"
            elif return_type==list:
                output_parser = "list"
            elif return_type==bool:
                output_parser = "boolean"
            elif issubclass(return_type, OutputWithFunctionCall):
                return_type = "str"
            elif issubclass(return_type,BaseModel):
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
                    output_parser = PydanticOutputParser(model=return_type)
                elif return_type==None:
                    raise Exception(f"You must annotate the return type for pydantic output parser, so that we can infer the model")
                else:
                    raise Exception(f"Unsupported return type {return_type} for pydantic output parser")
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
            prompt_type=prompt_type
        )


    def get_final_template(self, **kwargs: Any)->PromptTemplate:
        """Create Chat Messages."""
        
        
        if self.default_values:
            # if we have default values, we will use them to fill in missing values
            kwargs = {**self.default_values, **kwargs}

        kwargs = {k:v for k,v in kwargs.items() if v is not None}

        if isinstance(self.prompt_template_drafts,list):
            message_templates=[]
            for message_draft in self.prompt_template_drafts:
                msg_template = message_draft.finalize_template(kwargs)
                
                message_templates.append(msg_template)

            template = ChatPromptTemplate(messages=message_templates, output_parser=self.output_parser)
        else:
            template = self.prompt_template_drafts.finalize_template(kwargs)
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
                if not msg.content or not msg.content.strip():
                    formatted.messages.remove(msg)
        self.on_prompt_formatted(formatted.to_string())

        return formatted
    
    
    def format(self, **kwargs: Any) ->str:
        formatted = self.get_final_template(**kwargs).format(**kwargs)
        self.on_prompt_formatted(formatted)
        return formatted

    def on_prompt_formatted(self, formatted:str):
        if not self.prompt_type :
            log_level = logging.DEBUG
        else:
            log_level = self.prompt_type.log_level
            
        log_color =  LogColors.DARK_GRAY # we dont want to color the prompt, is's misleading... we color only the output
        print_log(f"Prompt:\n{formatted}",  log_level , log_color)

