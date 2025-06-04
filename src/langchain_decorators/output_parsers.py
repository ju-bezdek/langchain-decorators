import datetime
import logging
from textwrap import dedent
from typing import Callable, Dict, List, Type, TypeVar, Union, get_origin, get_args
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import BaseOutputParser, OutputParserException
import re
import json
import yaml
from .function_decorator import llm_function
from .pydantic_helpers import *


import pydantic
if pydantic.__version__ <"2.0.0":
    from pydantic import BaseModel, ValidationError
    from pydantic.schema import field_schema, get_flat_models_from_fields, get_model_name_map
    from pydantic.fields import ModelField, Field
else:
    if USE_PYDANTIC_V1:
        from pydantic.v1 import BaseModel, ValidationError
        from pydantic.v1.schema import (
            field_schema,
            get_flat_models_from_fields,
            get_model_name_map,
        )
        from pydantic.v1.fields import ModelField
    else:
        from pydantic import BaseModel, ValidationError
        from pydantic import ConfigDict as BaseConfig
        from pydantic.fields import Field, FieldInfo as ModelField


class ErrorCodes:
    UNSPECIFIED = 0
    INVALID_FORMAT = 10
    INVALID_JSON = 15
    DATA_VALIDATION_ERROR = 20
class OutputParserExceptionWithOriginal(OutputParserException):
    """Exception raised when an output parser fails to parse the output of an LLM call."""    

    def __init__(self, message: str, original: str, original_prompt_needed_on_retry:bool=False, error_code:int=0) -> None:
        super().__init__(message)
        self.original = original
        self.observation=message
        self.error_code=error_code
        self.original_prompt_needed_on_retry=original_prompt_needed_on_retry

    def __str__(self) -> str:
        return f"{super().__str__()}\nOriginal output:\n{self.original}"


class ListOutputParser(BaseOutputParser):
    """Class to parse the output of an LLM call in a bullet/numbered list format to a list."""

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

    def __init__(
        self, pattern: str = r"((Yes)|(No)|(True)|(False))([,|.|!]|$)"
    ) -> None:
        super().__init__(pattern=pattern)

    def parse(self, text: str) -> bool:
        """Parse the output of an LLM call."""

        match = re.search(self.pattern, text, flags=re.MULTILINE | re.IGNORECASE)

        if not match:
            raise OutputParserExceptionWithOriginal(message=self.get_format_instructions(),original=text, original_prompt_needed_on_retry=True, error_code=ErrorCodes.INVALID_FORMAT)
        else:
            return match.group(1).lower() == "yes" or match.group(1).lower() == "true"

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""
        return "Reply only Yes or No.\nUse this format: Final decision: Yes/No"

class JsonOutputParser(BaseOutputParser):
    """Class to parse the output of an LLM call to a Json."""

    @property
    def _type(self) -> str:
        return "json"

    def find_json_block(self,text, raise_if_not_found=True):

        start_code_block = list(re.finditer(r"(\n|^)```((json)|\n)", text))
        if start_code_block:
            i_start = start_code_block[0].span()[1]
            end_code_block = list(re.finditer(r"\n```($|\n)", text[i_start:]))
            if end_code_block:
                i_end = end_code_block[0].span()[0]
                text = text[i_start : i_start + i_end]

        match = re.search(r"[\{|\[].*[\}|\]]", text.strip(),
                              re.MULTILINE | re.IGNORECASE | re.DOTALL)
        if not match and raise_if_not_found:
            raise OutputParserExceptionWithOriginal(
                message="No JSON found in the response",
                original=text,
                error_code=ErrorCodes.INVALID_JSON,
            )
        return match

    def replace_json_block(self, text: str, replace_func:Callable[[dict],str]) -> str:
        try:

            match = self.find_json_block(text)
            json_str = match.group()
            i_start = match.start()
            _i_start = ("\n"+text).rfind("\n```", 0, i_start)
            i_end = match.end()
            _i_end = text.find("\n```\n", i_end)
            i_start=_i_start if _i_start>=0 else i_start
            i_end=_i_end+5 if _i_end>=0 else i_end

            json_dict = json.loads(json_str, strict=False)
            replacement = replace_func(json_dict)
            return (text[:i_start] + replacement + text[i_end:]).strip()

        except (json.JSONDecodeError) as e:

            msg = f"Invalid JSON\n {text}\nGot: {e}"
            raise OutputParserExceptionWithOriginal(msg, text, error_code=ErrorCodes.INVALID_JSON)

    def parse(self, text: str) -> dict:
        try:
            # Greedy search for 1st json candidate.
            match = self.find_json_block(text)
            json_str = match.group()
            try:
                json_dict = json.loads(json_str, strict=False)
            except json.JSONDecodeError as e:
                try:
                    from json_repair import repair_json
                    repair_json = repair_json(json_str)
                    json_dict = json.loads(repair_json, strict=False)
                    return json_dict
                except ImportError:
                    logging.warning("We might have been able to fix this output using json_repair. You can try json autorepair by installing json_repair package (`pip install json_repair`)")
                    pass
                raise e
            return json_dict

        except (json.JSONDecodeError) as e:

            msg = f"Invalid JSON\n {text}\nGot: {e}"
            raise OutputParserExceptionWithOriginal(msg, text, error_code=ErrorCodes.INVALID_JSON)

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""
        return "Return result as a valid JSON"


T = TypeVar("T", bound=BaseModel)


class PydanticOutputParser(BaseOutputParser[T]):
    """Class to parse the output of an LLM call to a pydantic object."""
    model: Type[T]
    as_list: bool = False
    instructions_as_json_example: bool = True

    def __init__(self, model: Type[T], instructions_as_json_example: bool = True, as_list: bool = False):
        super().__init__(model=model, instructions_as_json_example=instructions_as_json_example,as_list=as_list)

    @property
    def _type(self) -> str:
        return "pydantic"

    def parse(self, text: str) -> T:
        try:
            # Greedy search for 1st json candidate.
            regex_pattern = r"\[.*\]" if self.as_list else r"\{.*\}"
            match = re.search(regex_pattern, text.strip(),re.MULTILINE | re.IGNORECASE | re.DOTALL)
            json_str = ""
            if match:
                json_str = match.group()
            json_dict = json.loads(json_str, strict=False)
            if self.as_list:
                if isinstance(json_dict, dict) and "items" in json_dict:
                    json_dict = json_dict["items"]
                return [self.model(**item) for item in json_dict]
            else:
                return self.model(**json_dict)

        except (json.JSONDecodeError) as e:
            msg = f"Invalid JSON\n {text}\nGot: {e}"
            raise OutputParserExceptionWithOriginal(msg, text, error_code=ErrorCodes.INVALID_JSON)

        except ValidationError as e:
            try:
                json_dict_aligned = align_fields_with_model(json_dict, self.model)
                return self.model.parse_obj(json_dict_aligned)
            except ValidationError as e:
                err_msg =humanize_pydantic_validation_error(e)
                raise OutputParserExceptionWithOriginal(f"Data are not in correct format: {json_str or text}\nErrors: {err_msg}",text, error_code=ErrorCodes.DATA_VALIDATION_ERROR)

    def get_json_example_description(
        self, model: Type[BaseModel] = None, indentation_level=0
    ):
        field_descriptions = {}
        model = model or self.model

        if USE_PYDANTIC_V1:
            for field, field_info in model.__fields__.items():

                _item_type = None

                if field_info.type_ == field_info.outer_type_:
                    _type = field_info.type_
                elif list == getattr(field_info.outer_type_, "__origin__", None):
                    # is list
                    _type = list
                    _item_type = field_info.outer_type_.__args__[0]
                elif dict == getattr(field_info.outer_type_, "__origin__", None):
                    _type = dict
                else:
                    raise Exception(f"Unknown type: {field_info.annotation}")
                _nullable = field_info.allow_none
                _description = field_info.field_info.description
                if _nullable and "optional" not in _description:
                    _description = "(optional) " + _description
                if get_origin(_type) == Union:
                    alternative_types = [
                        union_type
                        for union_type in get_args(_type)
                        if union_type != type(None)
                    ]
                    _indent = "\t" * (indentation_level + 1)
                    _join = f"\n{_indent}or\n\n"
                    field_descriptions[field] = (_join).join(
                        [
                            self.get_json_example_description(
                                union_type, indentation_level=indentation_level + 1
                            )
                            for union_type in alternative_types
                        ]
                    )
                elif isinstance(_type, Type) and issubclass(_type, BaseModel):
                    field_descriptions[field] = self.get_json_example_description(
                        _type, indentation_level + 1
                    )
                elif _type == datetime:
                    field_descriptions[field] = "an ISO formatted datetime string"
                elif _type == str:
                    desc = _get_str_field_description(field_info)
                    field_descriptions[field] = desc
                elif _type in [bool, int, float]:
                    desc = field_info.field_info.description or "value"
                    field_descriptions[field] = f"{desc} as {_type.__name__}"
                elif _type == dict:
                    desc = _get_str_field_description(field_info)
                    field_descriptions[field] = f"{desc} as valid JSON object"
                elif _type == list:
                    desc = (
                        field_info.field_info.description + " as"
                        if field_info.field_info.description
                        else "a"
                    )
                    if _item_type:
                        if isinstance(_item_type, Type) and issubclass(
                            _item_type, BaseModel
                        ):
                            _item_desc = "\n" + self.get_json_example_description(
                                _item_type, indentation_level + 1
                            )
                        else:
                            _item_desc = f"{_item_type.__name__}"
                    field_descriptions[field] = (f"{desc} valid JSON array") + (
                        f" of {_item_desc}" if _item_desc else ""
                    )
                    field_descriptions[field] = f"[ {field_descriptions[field]} ]"
                else:
                    flat_models = get_flat_models_from_fields([field_info], set())
                    model_name_map = get_model_name_map(flat_models)
                    the_field_schema, sub_models, __ = field_schema(
                        field_info, model_name_map=model_name_map
                    )
                    if sub_models:
                        the_field_schema["definitions"] = sub_models
                        the_field_schema = sanitize_pydantic_schema(the_field_schema)
                        if the_field_schema.get("items") and the_field_schema[
                            "items"
                        ].get("$ref"):
                            the_field_schema["items"] = next(iter(sub_models.values()))

                    example = the_field_schema.get("example")
                    _description = ""
                    if the_field_schema.get("type") == "array":
                        if the_field_schema.get("items", None) and the_field_schema[
                            "items"
                        ].get("properties", None):
                            _item_type_str = "\n" + self.get_json_example_description(
                                _item_type, indentation_level + 1
                            )
                        else:
                            _item_type_str = describe_field_schema(
                                the_field_schema["items"]
                            )
                        _description += ", list of " + _item_type_str

                    if example:
                        _description += ", for example: " + str(example)
                    field_descriptions[field] = _description

            lines = []
            for field, field_info in model.__fields__.items():
                desc_lines = "\n".join(
                    (
                        "\t" * indentation_level + line
                        for line in field_descriptions[field].splitlines()
                    )
                ).strip()

                lines.append("\t" * indentation_level + f'"{field}": {desc_lines}')

            return (
                "\t" * indentation_level
                + "{\n"
                + ",\n".join(lines)
                + "\n"
                + "\t" * indentation_level
                + "}\n"
            )
        else:
            logging.warning(
                "get_json_example_description: This functionality is DEPRECATED and not fully supported in Pydantic v2.0.0... modern models support json schema natively."
            )
            for field_name, field_info in model.model_fields.items():
                _item_type = None
                _type = field_info.annotation
                _nullable = field_info.is_required()
                _description = field_info.description or ""

                if _nullable and "optional" not in _description:
                    _description = "(optional) " + _description

                # Handle Union types
                if get_origin(_type) == Union:
                    alternative_types = [
                        union_type
                        for union_type in get_args(_type)
                        if union_type != type(None)
                    ]
                    _indent = "\t" * (indentation_level + 1)
                    _join = f"\n{_indent}or\n\n"
                    field_descriptions[field_name] = (_join).join(
                        [
                            self.get_json_example_description(
                                union_type, indentation_level=indentation_level + 1
                            )
                            for union_type in alternative_types
                            if isinstance(union_type, type)
                            and issubclass(union_type, BaseModel)
                        ]
                    )

                # Handle nested BaseModel
                elif isinstance(_type, type) and issubclass(_type, BaseModel):
                    field_descriptions[field_name] = self.get_json_example_description(
                        _type, indentation_level + 1
                    )

                # Handle datetime
                elif _type == datetime.datetime:
                    field_descriptions[field_name] = "an ISO formatted datetime string"

                # Handle string
                elif _type == str:
                    desc = _get_str_field_description(field_info)
                    field_descriptions[field_name] = desc

                # Handle basic types
                elif _type in [bool, int, float]:
                    desc = field_info.description or "value"
                    field_descriptions[field_name] = f"{desc} as {_type.__name__}"

                # Handle dict
                elif _type == dict or get_origin(_type) == dict:
                    desc = field_info.description or "data"
                    field_descriptions[field_name] = f"{desc} as valid JSON object"

                # Handle list
                elif _type == list or get_origin(_type) == list:
                    desc = (
                        field_info.description + " as"
                        if field_info.description
                        else "a"
                    )
                    _item_type = get_args(_type)[0] if get_args(_type) else None

                    _item_desc = ""
                    if _item_type:
                        if isinstance(_item_type, type) and issubclass(
                            _item_type, BaseModel
                        ):
                            _item_desc = "\n" + self.get_json_example_description(
                                _item_type, indentation_level + 1
                            )
                        else:
                            _item_desc = f"{_item_type.__name__}"

                    field_descriptions[field_name] = (f"{desc} valid JSON array") + (
                        f" of {_item_desc}" if _item_desc else ""
                    )
                    field_descriptions[field_name] = (
                        f"[ {field_descriptions[field_name]} ]"
                    )

                # Handle other types using schema information
                else:
                    # Use Pydantic v2's JSON schema capabilities
                    schema = model.model_json_schema()
                    properties = schema.get("properties", {})
                    field_schema = properties.get(field_name, {})

                    example = field_schema.get("example")
                    _description = ""

                    if field_schema.get("type") == "array":
                        items = field_schema.get("items", {})
                        if items.get("properties"):
                            if _item_type:
                                _item_type_str = (
                                    "\n"
                                    + self.get_json_example_description(
                                        _item_type, indentation_level + 1
                                    )
                                )
                            else:
                                _item_type_str = "objects"
                        else:
                            _item_type_str = items.get("type", "values")

                        _description += ", list of " + _item_type_str

                    if example:
                        _description += ", for example: " + str(example)

                    field_descriptions[field_name] = (
                        _description or field_info.description or "value"
                    )

            # Format the output
            lines = []
            for field_name in model.model_fields.keys():
                desc_lines = "\n".join(
                    (
                        "\t" * indentation_level + line
                        for line in field_descriptions[field_name].splitlines()
                    )
                ).strip()

                lines.append("\t" * indentation_level + f'"{field_name}": {desc_lines}')

            return (
                "\t" * indentation_level
                + "{\n"
                + ",\n".join(lines)
                + "\n"
                + "\t" * indentation_level
                + "}\n"
            )

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""
        if not self.instructions_as_json_example:
            return "Return result as a valid JSON that matched this json schema definition:\n" + yaml.safe_dump(self.model.schema())
        else:
            json_example = self.get_json_example_description(self.model)
            if self.as_list:
                json_example = f"[\n{json_example}\n...\n]"

            return dedent(f"""```json\n{json_example}```""").strip()

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
            raise OutputParserExceptionWithOriginal(
                f"Function call arguments are not in correct format: {serialized}Errors: {err_msg}",
                serialized,
                error_code=ErrorCodes.DATA_VALIDATION_ERROR,
            )

    def get_format_instructions(self) -> str:
        return "" # will be handled by openai

    def build_llm_function(self):
        @llm_function(arguments_schema=self.model)
        def generate_response( **kwargs) -> T:
            """ Use this to transform the data into desired format. """
            # above is a description for LLM...
            return kwargs
        return generate_response


class MarkdownStructureParser(ListOutputParser):
    model: Union[Type[T], None] = None
    level: int = 1
    sections_parsers: Union[Dict[str, Union[BaseOutputParser, dict]], None] = Field(
        default_factory=dict
    )

    def __init__(
        self,
        model: Type[T] = None,
        sections_parsers: Dict[str, Union[dict, BaseOutputParser]] = None,
        level=1,
    ):
        super().__init__(
            model=model, sections_parsers=sections_parsers or {}, level=level
        )

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
                    raise ValueError("Type not supported: dict.")
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
        encode = lambda s: re.sub(r"([^\w]+)", "_", s.lower()) if s else s
        for i, section_separator_match in enumerate(sections_separators):
            section_name = section_separator_match.group(1)

            if self.model:
                # Handle both Pydantic v1 and v2 field access
                if USE_PYDANTIC_V1:
                    field_items = self.model.__fields__.items()
                    section_name = next(
                        (
                            field
                            for field, field_info in field_items
                            if (
                                field_info.field_info.title == section_name
                                or encode(field) == encode(section_name)
                                or encode(field_info.alias) == encode(section_name)
                            )
                        ),
                        section_name,
                    )
                else:
                    field_items = self.model.model_fields.items()
                    section_name = next(
                        (
                            field
                            for field, field_info in field_items
                            if (
                                encode(getattr(field_info, "title", None))
                                == encode(section_name)
                                or encode(field) == encode(section_name)
                                or encode(getattr(field_info, "alias", None))
                                == encode(section_name)
                            )
                        ),
                        section_name,
                    )

            # Extract section content
            if i < len(sections_separators) - 1:
                section_content = text[
                    section_separator_match.end() : sections_separators[i + 1].start()
                ]
            else:
                section_content = text[section_separator_match.end():]

            # Parse section content
            parsed_content = None
            if self.sections_parsers and (
                self.sections_parsers.get(section_name, None)
                or self.sections_parsers.get(section_separator_match.group(1))
            ):

                parser = self.sections_parsers.get(
                    section_name, None
                ) or self.sections_parsers.get(section_separator_match.group(1))

                if isinstance(parser, BaseOutputParser):
                    parsed_content = parser.parse(section_content)

            if not parsed_content:
                parsed_content = section_content.strip()

            res[section_name] = parsed_content

        # Return parsed result
        if self.model:

            try:
                return self.model(**res)
            except ValidationError as e:
                try:
                    res_aligned = align_fields_with_model(res, self.model)
                    if USE_PYDANTIC_V1:
                        return self.model.parse_obj(res_aligned)
                    else:
                        return self.model.model_validate(res_aligned)
                except ValidationError as e:
                    err_msg = humanize_pydantic_validation_error(e)
                    raise OutputParserExceptionWithOriginal(
                        f"Data are not in correct format: {text}\nGot: {err_msg}",
                        text,
                        error_code=ErrorCodes.DATA_VALIDATION_ERROR,
                    )
        else:
            return res

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""
        res = "Return result as a markdown in this format:\n"
        if self.model or self.sections_parsers:
            res += self.get_instructions_for_sections(
                self.model, self.sections_parsers)

        else:
            res += "# Section 1\n\ndescription\n\n#Section 2\n\ndescription\n\n..."
        return res


def _get_str_field_description(field_info: ModelField, ignore_nullable: bool = False, default="?"):
    if USE_PYDANTIC_V1:

        _nullable = field_info.allow_none
        _description = field_info.field_info.description
        _example = field_info.field_info.extra.get("example")
        _enum = field_info.field_info.extra.get("enum")
        _regex = field_info.field_info.extra.get("regex")
        _one_of = _enum or field_info.field_info.extra.get("one_of")
    else:
        _nullable = not field_info.is_required()
        _description = field_info.description or ""
        _example = field_info.examples
        _enum = (
            field_info.json_schema_extra.get("enum")
            if field_info.json_schema_extra
            else None
        )
        _regex = (
            field_info.json_schema_extra.get("regex")
            if field_info.json_schema_extra
            else None
        )
        _one_of = (
            field_info.json_schema_extra.get("one_of")
            if field_info.json_schema_extra
            else None
        )

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
        description = default

    return (description if _one_of else f"\" {description} \"")

def describe_field_schema(field_schema:dict):
    if "type" in field_schema:
        res = field_schema.pop("type")
        return res + ", " + ", ".join([f"{k}:{v}" for k,v in field_schema.items()])
    else:
        return ""
