
from typing import Type
from pydantic import BaseModel, ValidationError
from pydantic.fields import ModelField



def get_field_type(field_info: ModelField):
    _item_type=None
                
    if field_info.type_==field_info.outer_type_:
        _type=field_info.type_
    elif list == getattr(field_info.outer_type_, '__origin__', None):
        #is list
        _type = list
        
    elif dict == getattr(field_info.outer_type_, '__origin__', None):
        _type=dict
    else:
        raise Exception(f"Unknown type: {field_info.annotation}")
   
    return _type



def is_field_nullable(field_info: ModelField):
    _nullable=field_info.allow_none
    

def get_field_item_type(field_info: ModelField):
    
    if list == getattr(field_info.outer_type_, '__origin__', None):
        return field_info.outer_type_.__args__[0]
    
def align_fields_with_model( data:dict, model:Type[BaseModel]) -> dict:
    res = {}
    data_with_compressed_keys=None
    for field,field_info in model.__fields__.items():
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
            if field_info.type_ and issubclass(field_info.type_, BaseModel):
                value = align_fields_with_model(value, field_info.type_)
        elif isinstance(value, list):
            value = [align_fields_with_model(item, field_info.type_) for item in value]
        res[field]=value
    return res
    

def humanize_pydantic_validation_error(validation_error:ValidationError):
     return "\n".join([ f'{".".join([str(i) for i in err.get("loc")])} - {err.get("msg")} ' for err in  validation_error.errors()])


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

        replace_refs_recursive(schema)
    return schema
