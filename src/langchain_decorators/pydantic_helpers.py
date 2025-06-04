import inspect
from typing import Dict, Tuple, Type, get_origin
import os

from .common import get_type_from_annotation
import pydantic
if pydantic.__version__ <"2.0.0":
    from pydantic import BaseModel, ValidationError
    from pydantic.fields import FieldInfo

    USE_PYDANTIC_V1 = True
else:
    USE_PYDANTIC_V1 = os.environ.get("USE_PYDANTIC_V1", "").lower() in ("true", "1")
    if USE_PYDANTIC_V1:
        from pydantic.v1 import BaseModel, ValidationError, BaseConfig
        from pydantic.v1.fields import FieldInfo
    else:
        from pydantic import BaseModel, ValidationError
        from pydantic import ConfigDict as BaseConfig
        from pydantic.fields import FieldInfo as FieldInfo


def get_field_type(field_info: FieldInfo):
    if field_info.annotation:
        # In Pydantic v2, use annotation
        _type = get_type_from_annotation(field_info.annotation)
        return _type

    return str


def is_field_nullable(field_info: FieldInfo):
    # In Pydantic v2, check the annotation for Optional/Union with None
    if hasattr(field_info, "annotation"):
        from typing import Union
        import types

        annotation = field_info.annotation
        origin = get_origin(annotation)

        # Check if it's Union type (which Optional uses)
        if origin is Union:
            args = getattr(annotation, "__args__", ())
            return type(None) in args

        # Check if it's explicitly None
        if annotation is type(None):
            return True

    # Fallback: check if default is None
    return getattr(field_info, "default", ...) is None


def get_field_item_type(field_info: FieldInfo):
    if field_info.annotation:
        # In Pydantic v2, use annotation
        _type, _args = get_type_from_annotation(field_info.annotation, True)
        return _args[0]

    return str


def align_fields_with_model( data:dict, model:Type[BaseModel]) -> dict:
    res = {}
    data_with_compressed_keys=None
    if USE_PYDANTIC_V1:
        field_items = model.__fields__.items()
    else:
        field_items = model.model_fields.items()
    for field, field_info in field_items:
        value=None
        if field in data:
            value=data[field]
        elif field_info.field_info.title is not None:
            if field_info.field_info.title in data:
                value=data[field_info.field_info.title] 
            elif field_info.field_info.title.lower() in data:
                value=data[field_info.field_info.title.lower()]
        elif field_info.field_info.alias:
            if field_info.field_info.alias in data:
                value=data[field_info.field_info.alias]
            elif field_info.field_info.alias.lower() in data:
                value=data[field_info.field_info.alias.lower()]
        else:
            if not data_with_compressed_keys:
                data_with_compressed_keys= {k.lower().replace(" ",""):v for k,v in data.items()}
            compressed_key = field.lower().replace(" ","").replace("_","")
            if compressed_key in data_with_compressed_keys:
                value=data_with_compressed_keys[compressed_key]

        if isinstance(value, dict):
            field_type = get_origin(field_info.type_) if field_info.type_ else None
            if field_info.type_ and isinstance(field_type,type) and issubclass(field_type, BaseModel):
                value = align_fields_with_model(value, field_info.type_)
        elif isinstance(value, list):
            value = [align_fields_with_model(item, field_info.type_) for item in value]
        res[field]=value
    return res


def humanize_pydantic_validation_error(validation_error: "ValidationError"):
    return "\n".join(
        [
            f'{".".join([str(i) for i in err.get("loc")])} - {err.get("msg")} '
            for err in validation_error.errors()
        ]
    )


def sanitize_pydantic_schema(schema:dict):
    """ Pydantic schema uses '$ref' references to definitions, which doesn't go well with LLMs... les fix that  """
    if schema.get("definitions"):
        definitions=schema.pop("definitions")
        for def_key, definition in definitions.items():
            if "title" in definition:
                definition.pop("title") # no need for this
            nested_ref = next((1 for val in definition.get("properties",{}) if isinstance(val,dict) and val.get("$ref") ),None)
            if nested_ref:
                raise Exception(f"Nested $ref not supported! ... probably recursive schema: {def_key}")
        def replace_refs_recursive(schema:dict):
            if isinstance(schema,dict):
                if schema.get("properties"):
                    for k,v in schema.get("properties").items():
                        if v.get("$ref"):
                            schema["properties"][k]=definitions[v["$ref"].split("/")[-1]]
                        elif v.get("properties"):
                            replace_refs_recursive(v)
                        elif v.get("items"):
                            if isinstance(v["items"],dict):
                                ref = v["items"].get("$ref")
                                if  ref:
                                    v["items"]=definitions[ref.split("/")[-1]]
                if schema.get("items") and schema["items"].get("$ref"):
                    ref=schema["items"]["$ref"]
                    ref_key = ref.split("/")[-1]
                    schema["items"]= definitions.get(ref_key)

        replace_refs_recursive(schema)
    return schema


def get_arguments_as_pydantic_fields(func) -> Dict[str, Tuple[Type, pydantic.Field]]:
    argument_types = {}

    for arg_name, arg_desc in inspect.signature(func).parameters.items():
        if arg_name != "self" and not (
            arg_name.startswith("_") and arg_desc.default != inspect.Parameter.empty
        ):
            default = (
                arg_desc.default
                if arg_desc.default != inspect.Parameter.empty
                else None
            )
            if arg_desc.annotation == inspect._empty:
                raise Exception(
                    f"Argument '{arg_name}' of function {func.__name__} has no type annotation"
                )
            model_kwargs = {}
            if arg_desc.default == inspect.Parameter.empty:
                model_kwargs["required"] = True
            else:
                model_kwargs["default"] = default

            argument_types[arg_name] = (
                arg_desc.annotation,
                pydantic.Field(name=arg_name, **model_kwargs),
            )

    return argument_types
