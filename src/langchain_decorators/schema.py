from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Union, List, Literal


import pydantic

from .pydantic_helpers import USE_PYDANTIC_V1

if pydantic.__version__ < "2.0.0":
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
    data: Union[str, bytes]
    source_type: Optional[Literal["base64", "url"]] = None
    mime_type: Optional[str] = None
    file_name: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

    @field_validator("data")
    @classmethod
    def validate_input(cls, v):
        if not isinstance(v, (str, bytes)):
            raise ValueError("input must be str or bytes")
        return v

    def __init__(
        self,
        data: Union[str, bytes],
        type: Literal["image", "file", "pdf", "audio"],
        source_type: Optional[Literal["base64", "url"]],
        mime_type: Optional[str] = None,
        file_name: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            data=data,
            type=type,
            source_type=source_type,
            mime_type=mime_type,
            file_name=file_name,
            extra=extra,
        )


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
        def __init__(self, *args, items=None, **kwargs):
            super(new_cls, self).__init__(items=items or [])

        new_cls.__init__ = __init__

        return new_cls
