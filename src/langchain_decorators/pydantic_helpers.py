
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
    