import asyncio

import logging
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Union, List
from langchain.schema import AIMessage
from langchain.schema import FunctionMessage
import json

from .llm_chat_session import LlmChatSession
import pydantic

from .pydantic_helpers import USE_PYDANTIC_V1

if pydantic.__version__ <"2.0.0":
    from pydantic import BaseModel, PrivateAttr
else:
    if USE_PYDANTIC_V1:
        from pydantic.v1 import BaseModel, PrivateAttr
    else:
        from pydantic import BaseModel, PrivateAttr


T = TypeVar("T")

# backward compatibility
from .llm_tool_use import OutputWithFunctionCall


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
