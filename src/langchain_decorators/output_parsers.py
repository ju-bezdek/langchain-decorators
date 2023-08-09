import datetime
import logging
from textwrap import dedent
from typing import Dict, List, Type, TypeVar, Union
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import BaseOutputParser, OutputParserException

import re
import json
import yaml
from pydantic import BaseModel, ValidationError
from pydantic.fields import ModelField
from .function_decorator import llm_function
from .pydantic_helpers import *


class OutputParserExceptionWithOriginal(OutputParserException):
    """Exception raised when an output parser fails to parse the output of an LLM call."""

    def __init__(self, message: str, original: str, original_prompt_needed_on_retry:bool=False) -> None:
        super().__init__(message)
        self.original = original
        self.original_prompt_needed_on_retry=original_prompt_needed_on_retry

    def __str__(self) -> str:
        return f"{super().__str__()}\nOriginal output:\n{self.original}"


class ListOutputParser(BaseOutputParser):
    """Class to parse the output of an LLM call to a list."""

    @property
    def _type(self) -> str:
        return "list"

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""

        pattern = r"^[ \t]*(?:[\-\*\+]|\d+\.)[ \t]+(.+)$"
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        if not matches and text:
            logging.warning(
                f"{self.__class__.__name__} : LLM returned {text} but we could not parse it into a list")
        return matches

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""
        return "Return result a s bulleted list."

class BooleanOutputParser(BaseOutputParser):
    """Class to parse the output of an LLM call to a boolean."""
    pattern:str
    
    @property
    def _type(self) -> str:
        return "boolean"
    
    def __init__(self, pattern: str = r"((Yes)|(No))([,|.|!]|$)") -> None:
        super().__init__(pattern=pattern)

    def parse(self, text: str) -> bool:
        """Parse the output of an LLM call."""

        
        
        match = re.search(self.pattern, text, flags=re.MULTILINE | re.IGNORECASE)
        
        if not match:
            raise OutputParserExceptionWithOriginal(message=self.get_format_instructions(),original=text, original_prompt_needed_on_retry=True)
        else:
            return match.group(1).lower() == "yes"
        

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""
        return "Reply only Yes or No.\nUse this format: Final decision: Yes/No"

class JsonOutputParser(BaseOutputParser):
    """Class to parse the output of an LLM call to a Json."""

    @property
    def _type(self) -> str:
        return "json"

    def parse(self, text: str) -> List[str]:
        try:
            # Greedy search for 1st json candidate.
            match = re.search(r"(\[\s*)?\{.*\}(\s*\])?", text.strip(),
                              re.MULTILINE | re.IGNORECASE | re.DOTALL)
            json_str = ""
            if match:
                json_str = match.group()
            json_dict = json.loads(json_str, strict=False)
            return json_dict

        except (json.JSONDecodeError) as e:

            msg = f"Invalid JSON\n {text}\nGot: {e}"
            raise OutputParserExceptionWithOriginal(msg, text)

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""
        return "Return result as a valid JSON"


T = TypeVar("T", bound=BaseModel)


class PydanticOutputParser(BaseOutputParser[T]):
    """Class to parse the output of an LLM call to a pydantic object."""
    model: Type[T]
    instructions_as_json_example: bool = True

    def __init__(self, model: Type[T], instructions_as_json_example: bool = True):
        super().__init__(model=model, instructions_as_json_example=instructions_as_json_example)

    @property
    def _type(self) -> str:
        return "pydantic"

    def parse(self, text: str) -> T:
        try:
            # Greedy search for 1st json candidate.
            match = re.search(r"\{.*\}", text.strip(),re.MULTILINE | re.IGNORECASE | re.DOTALL)
            json_str = ""
            if match:
                json_str = match.group()
            json_dict = json.loads(json_str, strict=False)

            return self.model.parse_obj(json_dict)

        except (json.JSONDecodeError) as e:
            msg = f"Invalid JSON\n {text}\nGot: {e}"
            raise OutputParserExceptionWithOriginal(msg, text)

        except ValidationError as e:
            try:
                json_dict_aligned = align_fields_with_model(json_dict, self.model)
                return self.model.parse_obj(json_dict_aligned)
            except ValidationError as e:
                err_msg =humanize_pydantic_validation_error(e)
                raise OutputParserExceptionWithOriginal(f"Data are not in correct format: {text}Errors: {err_msg}",text) 
        
    def get_json_example_description(self, model:Type[BaseModel], indentation_level=0):
        field_descriptions = {}
        for field, field_info in model.__fields__.items():

            _item_type = None

            if field_info.type_ == field_info.outer_type_:
                _type = field_info.type_
            elif list == getattr(field_info.outer_type_, '__origin__', None):
                # is list
                _type = list
                _item_type = field_info.outer_type_.__args__[0]
            elif dict == getattr(field_info.outer_type_, '__origin__', None):
                _type = dict
            else:
                raise Exception(f"Unknown type: {field_info.annotation}")
            _nullable = field_info.allow_none
            _description = field_info.field_info.description

            if issubclass(_type, BaseModel):
                field_descriptions[field] = (
                    self.get_json_example_description(_type, indentation_level+1))
            elif _type == str:
                desc = _get_str_field_description(field_info)
                field_descriptions[field] = (desc)
            elif _type == datetime:
                field_descriptions[field] = (
                    "an ISO formatted datetime string")
            elif _type == list:
                list_desc = f"[ {_description} ... list of {_item_type} ]"
                field_descriptions[field] = (list_desc)
            elif _type == dict:
                dict_desc = f"{{ ... {_description} ... }}"
                field_descriptions[field] = (dict_desc)
            elif _type == int:
                field_descriptions[field] = ("an integer")
            elif _type == float:
                field_descriptions[field] = ("a float number")
            else:
                field_descriptions[field] = "?"            

        lines = []
        for field, field_info in model.__fields__.items():
            desc_lines = "\n".join(
                ("\t"*indentation_level+line for line in field_descriptions[field].splitlines())).strip()

            lines.append("\t"*indentation_level + f"\"{field}\": {desc_lines}")

        return "{\n" + ",\n".join(lines) + "\n}"

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""
        if not self.instructions_as_json_example:
            return "Return result as a valid JSON that matched this json schema definition:\n" + yaml.safe_dump(self.model.schema())
        else:

            return dedent(f"""```json\n{self.get_json_example_description(self.model)}\n```""").strip()

class OpenAIFunctionsPydanticOutputParser(BaseOutputParser[T]):
    model: Type[T]

    @property
    def _type(self) -> str:
        return "opanai_functions_pydantic"
    
    def __init__(self, model: Type[T]):
        super().__init__(model=model)

    def parse(self, function_call_arguments:dict ) -> T:
        try:
            return self.model.parse_obj(function_call_arguments)
        except ValidationError as e:
            err_msg =humanize_pydantic_validation_error(e)
            serialized= json.dumps(function_call_arguments)
            raise OutputParserExceptionWithOriginal(f"Function call arguments are not in correct format: {serialized}Errors: {err_msg}",serialized)
        

    def get_format_instructions(self) -> str:
        return "" # will be handled by openai
    
    
    def build_llm_function(self):
        @llm_function()
        def generate_response( output:self.model) -> T:
            """ Use this to transform the data into desired format. """
            #above is a description for LLM...
            return output
        return generate_response


class CheckListParser(ListOutputParser):
    """Parses list a a dictionary... assume this format:
        - KeyParma1: Value1
        - KeyPara2: Value2
        ...
    """

    def __init__(self, model: Type[T] = None):
        self.model = model

    @property
    def _type(self) -> str:
        return "checklist"

    def get_instructions_for_model(self, model: Type[T]) -> str:
        fields_bullets = []
        for field in model.__fields__.values():
            description = [field.field_info.description]
            if field.field_info.extra.get("one_of"):
                description += "one of these values: [ "
                description += " | ".join(field.field_info.extra.get("one_of"))
                description += " ]"
            if field.field_info.extra.get("example"):
                description += f"e.g. {field.field_info.extra.get('example')}"
            if description:
                description = " ".join(description)
            else:
                description = "?"
            fields_bullets.append(f"- {field.name}: {description}")

    def parse(self, text: str) -> Union[dict, T]:
        """Parse the output of an LLM call."""

        pattern = r"^[ \t]*(?:[\-\*\+]|\d+\.)[ \t]+(.+)$"
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        result = {}
        if not matches:
            raise OutputParserExceptionWithOriginal(message="No matches found", original_output=text)
        for match in matches:
            key, value = match.split(":", 1)
            result[key.strip()] = value.strip()

        return matches

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""
        res = "Return result a s bulleted list in this format:\n"
        if self.model:
            res += self.get_instructions_for_model(self.model)
        else:
            res += "\n- Key1: Value1\n- Key2: Value2\n- ..."


class MarkdownStructureParser(ListOutputParser):
    model: Type[T] = None
    level: int = 1
    sections_parsers: Dict[str, Union[BaseOutputParser, dict]] = None

    def __init__(self,  model: Type[T] = None, sections_parsers: Dict[str, Union[dict, BaseOutputParser]] = None, level=1):

        super().__init__(model=model, sections_parsers=sections_parsers, level=level)
        if model:
            for field, field_info in model.__fields__.items():
                if sections_parsers and field in self.sections_parsers:
                    # if section parser was already provided, skip
                    if not type(self.sections_parsers.get(field)) == dict:
                        continue
                field_type = get_field_type(field_info)
                if get_field_type(field_info) == list:
                    item_type = get_field_item_type(field_info)
                    if item_type == str or item_type is None:
                        self.sections_parsers[field] = ListOutputParser()
                    else:
                        raise ValueError(
                            f"Unsupported item type {item_type} for property {model}.{field}. Only list of strings is supported.")
                elif field_type == dict:
                    self.sections_parsers[field] = CheckListParser()
                elif field_type and issubclass(field_type, BaseModel):

                    all_sub_str = all(True for sub_field_info in field_type.__fields__.values(
                    ) if get_field_type(sub_field_info) == str)

                    if all_sub_str:

                        self.sections_parsers[field] = MarkdownStructureParser(
                                model=field_type, sections_parsers=sections_parsers.get(field), level=level+1
                            )
                    else:
                        self.sections_parsers[field] = PydanticOutputParser(
                                model=field_type
                            )

                elif field_type == str:

                    self.sections_parsers[field] = None
                else:
                    raise ValueError(
                        f"Unsupported type {field_type} for property {field}.")
        elif sections_parsers:
            for property, property_parser in sections_parsers.items():
                if type(property_parser) == dict:
                    sections_parsers[property] = MarkdownStructureParser(
                        model=None, sections_parsers=property_parser, level=level+1)
                elif type(property_parser) == str:
                    sections_parsers[property] = None
                elif isinstance(property_parser, BaseOutputParser):
                    continue
                else:
                    raise ValueError(
                        f"Unsupported type {model.__fields__[property].annotation} for property {property}. Use a dict or a pydantic model.")
        else:
            self.sections_parsers = {}

    @property
    def _type(self) -> str:
        return "checklist"

    def get_instructions_for_sections(self,  model: Type[T] = None, sections_parsers: Dict[str, BaseOutputParser] = None) -> str:
        section_instructions = []
        if model:
            for field, field_info in model.__fields__.items():
                name: str = field_info.field_info.title or field
                section_instructions.append(self.level*"#" + f" {name}")
                if sections_parsers and sections_parsers.get(field):
                    section_instructions.append(
                        sections_parsers.get(field).get_format_instructions())
                    continue
                else:

                    description = _get_str_field_description(field_info)
                    section_instructions.append(description)
        else:
            for section, parser in sections_parsers.items():
                section_instructions.append(self.level*"#" + f" {section}")
                if isinstance(parser, BaseOutputParser):
                    section_instructions.append(
                        parser.get_format_instructions())
                else:
                    section_instructions.append("?")

        return "\n\n".join(section_instructions)

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""

        sections_separators = list(re.finditer(
            r"^#+[ |\t]+(.*)$", text, flags=re.MULTILINE))
        res = {}
        for i, section_separator_match in enumerate(sections_separators):

            section_name = section_separator_match.group(1)
            if self.model:
                section_name = next((field for field, field_info in self.model.__fields__.items() if field_info.field_info.title ==
                                    section_name or field.lower() == section_name.lower() or field_info.alias == section_name), section_name)
            if i < len(sections_separators)-1:
                section_content = text[section_separator_match.end(
                ):sections_separators[i+1].start()]
            else:
                section_content = text[section_separator_match.end():]

            parsed_content = None
            if self.sections_parsers and self.sections_parsers.get(section_name, None) or self.sections_parsers.get(section_separator_match.group(1)):
                parser = self.sections_parsers.get(
                    section_name, None) or self.sections_parsers.get(section_separator_match.group(1))
                if isinstance(parser, BaseOutputParser):
                    parsed_content = parser.parse(section_content)
            if not parsed_content:
                parsed_content = section_content.strip()

            res[section_name] = parsed_content

        if self.model:
            try:
                return self.model(**res)
            except ValidationError as e:
                try:
                    res_aligned = align_fields_with_model(res, self.model)
                    return self.model.parse_obj(res_aligned)
                except ValidationError as e:
                    err_msg =humanize_pydantic_validation_error(e)
                    raise OutputParserExceptionWithOriginal(f"Data are not in correct format: {text}\nGot: {err_msg}",text) 
        else:
            return res

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""
        res = "Return result as a markdown in this format:\n"
        if self.model or self.sections_parsers:
            res += self.get_instructions_for_sections(
                self.model, self.sections_parsers)

        else:
            res += "# Section 1\n\ndescription\n\n#Section 2\n\ndescription\n\..."
        return res


def _get_str_field_description(field_info: ModelField, ignore_nullable: bool = False):
    _nullable = field_info.allow_none
    _description = field_info.field_info.description
    _example = field_info.field_info.extra.get("example")
    _enum = field_info.field_info.extra.get("enum")
    _regex = field_info.field_info.extra.get("regex")
    _one_of = _enum or field_info.field_info.extra.get("one_of")
    _regex = field_info.field_info.extra.get("regex")
    description = []
    if _description:
        description.append(_description)
    if _one_of:
        description.append("one of these values: [ ")
        description.append(" | ".join([f"\"{enum_val}\"" for enum_val in _one_of]))
        description.append(" ]")
    if _example:
        description.append(f"e.g. {_example}")
    if _nullable and not ignore_nullable:
        description.append("... or null if not available")
    if _regex and not _enum:
        description.append(f"... must match this regex: {_regex}")

    if description:
        description = " ".join(description)
    else:
        description = "?"

    return (description if _one_of else f"\" {description} \"")
