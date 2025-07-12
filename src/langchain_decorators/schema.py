import asyncio

import logging
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Union, List, Literal
from langchain.schema import AIMessage
from langchain.schema import FunctionMessage
import json
import base64
from urllib.parse import urlparse

from .llm_chat_session import LlmChatSession
import pydantic

from .pydantic_helpers import USE_PYDANTIC_V1

if pydantic.__version__ <"2.0.0":
    from pydantic import BaseModel, PrivateAttr
else:
    if USE_PYDANTIC_V1:
        from pydantic.v1 import BaseModel, PrivateAttr
    else:
        from pydantic import BaseModel, PrivateAttr, field_validator


T = TypeVar("T")

# backward compatibility
from .llm_tool_use import OutputWithFunctionCall


class MessageAttachment(BaseModel):
    """Represents an attachment that can be included in messages."""
    
    type: Literal["image", "file", "pdf", "audio"]
    input: Union[str, bytes]
    source_type: Optional[Literal["base64", "url"]] = None
    source: Optional[Dict[str, Any]] = None
    mime_type: Optional[str] = None
    file_name: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

    @field_validator('input')
    @classmethod
    def validate_input(cls, v):
        if not isinstance(v, (str, bytes)):
            raise ValueError("input must be str or bytes")
        return v

    @field_validator('source_type', mode='before')
    @classmethod
    def auto_detect_source_type(cls, v, info):
        if v is not None:
            return v
        
        input_data = info.data.get('input') if info.data else None
        if input_data is None:
            return None
            
        if isinstance(input_data, bytes):
            return "base64"
        elif isinstance(input_data, str):
            parsed = urlparse(input_data)
            if parsed.scheme in ("http", "https"):
                return "url"
            else:
                return "base64"
        return "base64"

    @field_validator('source', mode='before')
    @classmethod
    def build_source(cls, v, info):
        if v is not None:
            return v
        
        data = info.data if info.data else {}
        input_data = data.get('input')
        source_type = data.get('source_type')
        mime_type = data.get('mime_type')
        
        if input_data is None or source_type is None:
            return None
            
        if source_type == "url":
            return {"url": str(input_data)}
        elif source_type == "base64":
            if isinstance(input_data, bytes):
                encoded = base64.b64encode(input_data).decode('utf-8')
            else:
                encoded = str(input_data)
            
            source_dict = {"base64": encoded}
            if mime_type:
                source_dict["mime_type"] = mime_type
            return source_dict
        
        return None


class PydanticListTypeWrapper(BaseModel, Generic[T]):
    items: List[T] = pydantic.Field(default_factory=list)

    @classmethod
    def __class_getitem__(cls, item_type):
        """Creates a new runtime type for each parametrized type."""
        # Create a new class with a meaningful name
        name = f"PydanticListOf{getattr(item_type, '__name__', str(item_type))}"

        # Create a new type dynamically that inherits from our class
        new_cls: BaseModel = type(
            name,
            (PydanticListTypeWrapper,),
            {
                "__origin__": list,
                "__args__": (item_type,),
                "__module__": cls.__module__,
                "__annotations__": {"items": List[item_type]},
            },
        )

        # Add methods needed for pydantic
        def __init__(self, items=None):
            super(new_cls, self).__init__(items or [])

        new_cls.__init__ = __init__

        return new_cls
