from functools import wraps
import inspect
from textwrap import dedent
from typing import Callable, Dict, Union, Type
from pydantic import BaseModel
from pydantic.fields import ModelField
import re
from enum import Enum

from .common import  get_arguments_as_pydantic_fields, get_function_docs, get_function_full_name

class DocstringsFormat(Enum):
    AUTO = "auto"
    GOOGLE = "google"
    SPHINX = "sphinx"
    NUMPY = "numpy"


def llm_function(
       argument_schema:Union[str, Type[BaseModel], dict]="auto",
       validate_docstrings:bool=True,
       docstring_format:str="auto",
        ):
    """
    Decorator for functions that take a language model as first argument.

     - argument_schema
        - `auto` (default): the schema is automatically inferred from the function signature. If docstrings are provided, they will be used to enhance function description
        - `pydantic` -  expects a pydantic model as first and only argument ('self' being ignored) - allows for controllable schema
        - `docstring` - parses the schema ONLY from the docstring

        or define pydantic model:
        ```
        @llm_function(argument_schema=MyFunctionPydanticArgsSchema)
        def my_function(self, question:str)->bool:
            ...
        ```

        or define a simple dict with descriptions:
        ```
        @llm_function(argument_schema={"question": "Question to ask"})
        def my_function(self, question:str)->bool:
            ...
        ```
    - validate_docstrings: if True, the docstrings will be parsed and validated against function. If parsing or validation fails, an error will be raised

    - docstring_format: the format of the docstring
        -  `auto` (default): the format is automatically inferred from the docstring
        -  `google`: the docstring is parsed as markdown (see [Google docstring format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html))
        -  `numpy`: the docstring is parsed as markdown (see [Numpy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html))
        -  `sphinx`: the docstring is parsed as sphinx format (see [Sphinx docstring format](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html))

    Examples:
    Google style docstrings:
    ```python
    @llm_function
    def function_with_google_docstrings(self, question:str)->bool:
        \"\"\"
        This is a function with google docstrings
        Args:  # but accepts also Parameters, Arguments
            question (str): Question to ask
        \"\"\"
    ```

    ```python
    @llm_function
    def function_with_numpy_docstrings(self, question:str)->bool:
        \"\"\"
        This is a function with google docstrings
        Parameters # but accepts also Args, Arguments
        ----------
            question :int
                Question to ask
            
        \"\"\"
    ```

    ```python
    @llm_function
    def function_with_sphinx_docstrings(self, question:str)->bool:
        \"\"\"
        This is a function with google docstrings
        :param question: Question to ask
            
        \"\"\"
    ```

    Note: The fact whether the parameter is optional is inferred from the function signature. You can also document it, but it will be just part of the description. If typing annotations are used (i.e. Optional[str]), this will be stripped to native type.
    
    """
    
    if callable(argument_schema):
        # this is the case when the decorator is called without arguments
        # we initialize params with default values
        func = argument_schema
        argument_schema = "auto"
    else:
        func = None
    
    def decorator(func):
        
        is_async = inspect.iscoroutinefunction(func)

        if not is_async:
            @wraps(func)
            def func_wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
        else:
            @wraps(func)
            async def func_wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
        def get_function_schema(_validate_docstrings=validate_docstrings):
            return build_func_description(func, format=docstring_format, validate_docstrings=_validate_docstrings)
        func_wrapper.get_function_schema=get_function_schema
        return func_wrapper
    
    if func:
        return decorator(func)
    else:
        return decorator
    




from pydantic.schema import get_field_info_schema, field_schema
def build_func_description(func:Callable, format:Union[DocstringsFormat,str]="auto", validate_docstrings:bool=True):
    
    if isinstance(format,str):
        format = DocstringsFormat(format)

    func_docs = get_function_docs(func)  
    func_name= get_function_full_name(func)
    

    arguments_fields = get_arguments_as_pydantic_fields(func)
    
    args_schema = None
    if len(arguments_fields)==1:
        first_param:ModelField = list(arguments_fields.values())[0]
        if first_param.required == True and issubclass(first_param.type_, BaseModel):
            # the one and only argument is a pydantic model
            args_schema =  first_param.type_.schema()
            args_schema={
                "type":"object",
                "properties":args_schema["properties"],
                "required":args_schema["required"]
            }
    
    if not args_schema:
        # any other case, that function arguments are not wrapped in pydantic model
        if func_docs:
            docsctrings_param_description = find_and_parse_params_from_docstrings(func_docs,format=format)
            if docsctrings_param_description:
                if validate_docstrings:
                    documented_params=set(docsctrings_param_description.keys()) 
                    implemented_params= set(arguments_fields.keys())
                    not_implemented = documented_params - implemented_params  # Set difference: keys in set1 but not in set2
                    not_documented = implemented_params - documented_params  # Set difference: keys in set2 but not in set1

                    if not_implemented or not_documented:
                        errs = []
                        if not_implemented:
                            errs.append(f"Missing (not implemented): {not_implemented}")
                        if not_documented:
                            errs.append(f"Missing (not documented): {not_documented}")
                        
                        raise ValueError("Docstrings parameters do not match function signature. "+errs.join(", "))
                
                for arg_name, arg_model_field in arguments_fields.items():
                    arg_docs = docsctrings_param_description.get(arg_name)
                    if arg_docs:
                        
                        arg_model_field.field_info.description =arg_docs["description"]
                        enum = parse_enum_from_docstring_param(arg_docs["type"],arg_docs["description"])
                        if enum:
                            arg_model_field.field_info.extra["enum"] = enum

        args_schema={
            "type":"object",
            "properties":{},
            "required":[]

        }
        for arg_name, arg_model_field in arguments_fields.items():
            param_schema, _ , nested_models_schema = field_schema(arg_model_field, model_name_map={}) 
            
            if nested_models_schema:
                raise NotImplementedError("Nested models are not supported yet")
            
            args_schema["properties"][arg_name] = param_schema
            if arg_model_field.required:
                args_schema["required"].append(arg_name)

    
    def pop_prop_title(schema):
        #title is autogenerated by pydantic and will just costs us tokens....
        if "title" in schema:
            del schema["title"]
        return schema
    args_schema["properties"] = {prop:pop_prop_title(prop_schema) for prop, prop_schema in args_schema["properties"].items()}
    
    description = parse_function_description_from_docstrings(func_docs) if func_docs else None
    if not description:
        raise ValueError(f"LLM Function {func_name} has no description in docstrings")
    return {
        "name":func_name,
        "description":description,
        "parameters":args_schema
    }
            



        


        
def parse_function_description_from_docstrings(docstring:str)->str:
    # we will return first text until first empty line
    
    lines = docstring.splitlines()
    description = []
    for line in lines:
        line = line.strip()
        if line:
            description.append(line)
        elif description:
            # if we have already some description, we stop at first empty line ... else continue
            break
    return "\n".join(description)





    


    

def find_and_parse_params_from_docstrings(docstring:str,format:DocstringsFormat)->str:
    """
    Find Args section in docstring.
    """

    args_section = None
    args_section_start = 0
    args_section_end = None
    
    if format == DocstringsFormat.AUTO or format == DocstringsFormat.GOOGLE:
        # auto here means more more free format than google
        args_section_start_regex_pattern = r"(^|\n)(Args|Arguments|Parameters)\s*:?\s*\n"
        args_section_end_regex_pattern = r"(^|\n)([A-Z][a-z]+)\s*:?\s*\n"
        if  format == DocstringsFormat.GOOGLE:
            param_start_parser_regex=r"(^|\n)\s+(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\((?P<type>[^\)]*)\)?\s*:\s*(?=\w+)"
        else:
            param_start_parser_regex=r"(^|\n)\s+(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\((?P<type>[^\)]*)\)?\s*(-|:)\s*(?=\w+)"
    elif format == DocstringsFormat.NUMPY:
        args_section_start_regex_pattern = r"(^|\n)(Args|Arguments|Parameters)\s*\n\s*---+\s*\n"
        args_section_end_regex_pattern = r"(^|\n)([A-Z][a-z]+)\s*\n\s*---+\s*\n"
        param_start_parser_regex=r"(^|\n)\s*(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)(\s*:\s*(?P<type>[^\)]*)\s*)?\n\s+(?=\w+)"
    elif format == DocstringsFormat.SPHINX:
        args_section_start_regex_pattern = None # we will look for :param everywhere
        args_section_end_regex_pattern=r"(\n)\s*:[a-z]"
        param_start_parser_regex=r"(^|\n)\s*:param\s+(?P<type>[^\)]*)\s+(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(?=\w+)"

    if args_section_start_regex_pattern:
        match = re.search(args_section_start_regex_pattern, docstring)
        if match:
            args_section_start = match.end()
            if args_section_end_regex_pattern:
                match = re.search(args_section_end_regex_pattern, docstring[args_section_start:])
                if match:
                    args_section_end = match.start() + args_section_start
            if not args_section_end:
                args_section_end = len(docstring)
            args_section = docstring[args_section_start:args_section_end]
        else:
            args_section=None
    else:
        args_section=docstring
    
    params={}
    if args_section:
        last_param=None
        last_param_end=None
        for param_start_match in re.finditer(param_start_parser_regex, args_section):
            if last_param_end is not None:
                last_param["description"]=args_section[last_param_end:param_start_match.start()].strip()

            param_name = param_start_match.group("name")
            param_type = param_start_match.group("type")
            last_param = {"type":param_type,"description":None}
            last_param_end = param_start_match.end()
            params[param_name]=last_param

        if last_param_end is not None:
            section_end=None
            if args_section_start_regex_pattern is None and args_section_end_regex_pattern:
                # this is handling SPHINX, we didnt parse the start so we cant parse the end until all the params are consumed... now we can parse the end after the last param
                section_end_match= re.search(args_section_end_regex_pattern, docstring[last_param_end:])    
                if section_end_match:
                    section_end=last_param_end + section_end_match.start()
            if not section_end:
                section_end=len(docstring)
            last_param["description"]=args_section[last_param_end:section_end].strip()
        

    if not params and  format == DocstringsFormat.AUTO:
            # try other options            
        options = [DocstringsFormat.NUMPY, DocstringsFormat.SPHINX]
        for option in options:
            result = find_and_parse_params_from_docstrings(docstring,option)
            if result:
                return result
    else:
        return params
        
            
def parse_enum_from_docstring_param(type:str, description)->str:
    enum_pattern=r"\[\s*(?P<value>[\"|'][\w|_|-]+[\"|']\s*(\||\]))+"
    enum_part_match = re.search(enum_pattern, type +" / " +description)
    if not enum_part_match:
        return None
    enum_strings =enum_part_match.group(0).strip("[]").split("|")
    return [enum.strip(" \"'") for enum in enum_strings]
