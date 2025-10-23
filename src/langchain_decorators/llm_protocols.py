"""
LLM Protocols Module

This module contains logic to enable customization and patching of functionality
that is specific to different LLM providers. It provides a flexible architecture
for handling provider-specific behaviors such as attachment encoding/decoding,
message formatting, and other protocol-specific operations.

The module follows a mixin pattern to allow for composable functionality and
easy extension for new LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, TYPE_CHECKING, Type, List, Tuple
from langchain_core.language_models import BaseLanguageModel
import re
import base64

# Additional imports for template building
from langchain_core.prompts import PromptTemplate, StringPromptTemplate, PromptTemplate
from langchain_core.prompts.dict import DictPromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.prompts.chat import (
    MessagesPlaceholder,
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


from .schema import MessageAttachment


# Global registry for LLM protocol handlers
_LLM_PROTOCOL_REGISTRY: Dict[str, Type["LlmProtocolBase"]] = {}


def register_llm_protocol(llm_name: str, handler: Type["LlmProtocolBase"]) -> None:
    """
    Register a protocol handler for a specific LLM name.

    Args:
        llm_name: The name of the LLM (as returned by llm.name)
        handler: The protocol handler class to use for this LLM
    """
    _LLM_PROTOCOL_REGISTRY[llm_name] = handler


class BaseAttachmentMixin(ABC):
    """
    Base mixin for handling attachment encoding and decoding operations.

    This mixin provides the interface for converting between different
    attachment representations used by various LLM providers.
    """

    def get_attachment_inputs(self, kwargs: Dict[str, Any]) -> List[str]:
        """
        Check if the provided kwargs contain any attachments.

        Args:
            kwargs: The keyword arguments to check for attachments

        Returns:
            List[str]: A list of keys corresponding to attachment inputs
        """
        attachment_inputs = [
            key
            for key in kwargs
            if isinstance(kwargs.get(key), MessageAttachment)
            or isinstance(kwargs.get(key), list)
            and all(isinstance(item, MessageAttachment) for item in kwargs.get(key))
        ]
        return attachment_inputs

    @abstractmethod
    def encode_attachment_to_content_dict(
        self, attachment: "MessageAttachment"
    ) -> Dict[str, Any]:
        """
        Encode an attachment object to a content dictionary format.

        Args:
            attachment: The attachment object to encode

        Returns:
            Dict[str, Any]: The encoded attachment as a content dictionary
        """
        pass

    @abstractmethod
    def decode_content_dict_to_attachment(
        self, content_dict: Dict[str, Any]
    ) -> "MessageAttachment":
        """
        Decode a content dictionary back to an attachment object.

        Args:
            content_dict: The content dictionary to decode

        Returns:
            MessageAttachment: The decoded attachment object
        """
        pass


class BaseTemplateBuilderMixin(ABC):
    """
    Base mixin for handling template building operations.

    This mixin provides the interface for building prompt templates
    from template parts used by various LLM providers.
    """

    @abstractmethod
    def build_template(
        self, template_parts: List[Tuple[str, str]], kwargs: Dict[str, Any]
    ) -> PromptTemplate:
        """
        Build a prompt template from template parts and kwargs.

        Args:
            template_parts: List of (template_string, prompt_block_name) tuples
            kwargs: All arguments passed to the decorated function

        Returns:
            PromptTemplate: ChatPromptTemplate or StringPromptTemplate
        """
        pass

    @abstractmethod
    def process_prompt_template_string(prompt: str) -> Union[
        StringPromptTemplate,
        list[Union[StringPromptTemplate, ImagePromptTemplate, DictPromptTemplate]],
    ]:
        raise NotImplementedError(
            "process_prompt_template_string method must be implemented in subclasses"
        )


class OpenAIAttachmentMixin(BaseAttachmentMixin):
    """
    OpenAI-specific attachment handling mixin.

    Implements the attachment encoding/decoding logic specific to OpenAI's
    message format and content structure.
    """

    def encode_attachment_to_content_dict(
        self, attachment: "MessageAttachment"
    ) -> Dict[str, Any]:
        """
        Encode an attachment to OpenAI's content dictionary format.

        Args:
            attachment: The attachment object to encode

        Returns:
            Dict[str, Any]: OpenAI-formatted content dictionary
        """
        if attachment.source_type == "url":
            return {
                "type": attachment.type,
                "source_type": "url",
                "url": attachment.data,
            }
        elif attachment.source_type == "base64":
            # Handle base64 data
            if isinstance(attachment.data, bytes):
                # Convert bytes to base64 string
                data = base64.b64encode(attachment.data).decode("utf-8")
            else:
                # Assume it's already base64 encoded string
                data = str(attachment.data)

            result = {
                "type": attachment.type,
                "source_type": "base64",
                "data": data,
            }

            # Add mime_type if available
            if attachment.mime_type:
                result["mime_type"] = attachment.mime_type

            return result
        else:
            raise ValueError(f"Unsupported source_type: {attachment.source_type}")

    def decode_content_dict_to_attachment(
        self, content_dict: Dict[str, Any]
    ) -> "MessageAttachment":
        """
        Decode OpenAI content dictionary back to attachment object.

        Args:
            content_dict: OpenAI-formatted content dictionary

        Returns:
            MessageAttachment: The decoded attachment object
        """
        from .schema import MessageAttachment

        attachment_type = content_dict.get("type")
        source_type = content_dict.get("source_type")

        if source_type == "url":
            return MessageAttachment(
                type=attachment_type, input=content_dict["url"], source_type="url"
            )
        elif source_type == "base64":

            return MessageAttachment(
                type=attachment_type,
                input=content_dict["data"],
                source_type="base64",
                mime_type=content_dict.get("mime_type"),
            )
        else:
            raise ValueError(f"Unsupported source_type in content_dict: {source_type}")

    def replace_prompt_input_partially(
        self, string: str, replacements: Dict[str, str]
    ) -> str:
        """
        Replace placeholders in the prompt string with actual values.

        Args:
            string: The prompt string with placeholders
            replacements: A dictionary mapping placeholders to their actual values

        Returns:
            str: The prompt string with placeholders replaced
        """
        for placeholder, value in replacements.items():
            string = re.sub(rf"{{\s*{re.escape(placeholder)}\s*}}", value, string)
        return string

    def _get_attachments(
        self, kwargs: Dict[str, Any], attachment_inputs: List[str]
    ) -> list[dict]:
        res = []
        for attachment_input in attachment_inputs:
            if isinstance(kwargs[attachment_input], MessageAttachment):
                res.append(
                    DictPromptTemplate(
                        template=self.encode_attachment_to_content_dict(
                            kwargs[attachment_input]
                        ),
                        template_format="f-string",
                    )
                )
            elif isinstance(kwargs[attachment_input], list):
                for attachment in kwargs[attachment_input]:
                    if isinstance(attachment, MessageAttachment):
                        res.append(
                            DictPromptTemplate(
                                template=self.encode_attachment_to_content_dict(
                                    attachment
                                ),
                                template_format="f-string",
                            )
                        )
                    else:
                        raise ValueError(
                            f"Invalid attachment type in list: {type(attachment)}"
                        )
        return res

    def process_prompt_template_string(self, prompt: str, kwargs: dict) -> Union[
        StringPromptTemplate,
        list[Union[StringPromptTemplate, DictPromptTemplate]],
    ]:
        string_prompt_template = PromptTemplate.from_template(prompt)
        prompt_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in string_prompt_template.input_variables
        }
        attachment_inputs = self.get_attachment_inputs(prompt_kwargs)
        if attachment_inputs:
            return [
                PromptTemplate.from_template(
                    self.replace_prompt_input_partially(
                        prompt, {k: "" for k in attachment_inputs}
                    ).strip()
                ),
                *self._get_attachments(prompt_kwargs, attachment_inputs),
            ]
        else:
            return string_prompt_template


class OpenAITemplateBuilderMixin(BaseTemplateBuilderMixin):
    """
    OpenAI-specific template building mixin.

    Implements the template building logic specific to OpenAI's
    message format and prompt structure.
    """

    def _get_class_by_role(self, role: str) -> Type[ChatMessagePromptTemplate]:
        """
        Get the appropriate ChatMessagePromptTemplate class based on the role.

        Args:
            role: The role of the message (e.g., "user", "assistant", "function")

        Returns:
            Type[ChatMessagePromptTemplate]: The class for the specified role
        """
        if role == "user":
            return HumanMessagePromptTemplate
        elif role == "assistant" or role == "ai":
            return AIMessagePromptTemplate
        elif role == "system":
            return SystemMessagePromptTemplate
        else:
            return None

    def build_template(
        self, template_parts: List[Tuple[str, str]], kwargs: Dict[str, Any]
    ) -> PromptTemplate:
        """
        Build a template using OpenAI's format.

        Args:
            template_parts: List of (template_string, prompt_block_name) tuples
            kwargs: All arguments passed to the decorated function

        Returns:
            PromptTemplate: ChatPromptTemplate or StringPromptTemplate
        """
        if len(template_parts) == 1 and not template_parts[0][1]:
            template_string = template_parts[0][0]
            return PromptTemplate.from_template(template_string)
        else:
            message_templates = []
            for template_string, prompt_block_name in template_parts:
                template_string = template_string.strip()
                content_template = self.process_prompt_template_string(
                    template_string, kwargs
                )
                if prompt_block_name == "placeholder":
                    message_templates.append(
                        MessagesPlaceholder(variable_name=template_string.strip(" {}"))
                    )
                elif prompt_block_name:

                    if "[" in prompt_block_name and prompt_block_name[-1] == "]":
                        i = prompt_block_name.find("[")
                        name = prompt_block_name[i + 1 : -1]
                        role = prompt_block_name[:i]
                    else:
                        name = None
                        role = prompt_block_name

                    if name:
                        additional_kwargs = {"name": name}
                    elif role == "function":
                        raise Exception(
                            f"Invalid function prompt block. function_name {name} is not set. Use this format: <prompt:function[function_name]>"
                        )
                    else:
                        additional_kwargs = {}

                    prompt_cls = self._get_class_by_role(role)
                    if prompt_cls:
                        message_prompt = prompt_cls(
                            prompt=content_template,
                            additional_kwargs=additional_kwargs,
                        )
                    else:
                        message_prompt = ChatMessagePromptTemplate(
                            role=role,
                            prompt=content_template,
                            additional_kwargs=additional_kwargs,
                        )
                    message_templates.append(message_prompt)

            return ChatPromptTemplate(messages=message_templates)


class LlmProtocolBase(BaseAttachmentMixin, BaseTemplateBuilderMixin):
    """
    Base class for LLM protocol implementations.

    This class serves as the foundation for all LLM provider-specific protocols,
    combining the base attachment mixin with template building functionality.
    Subclasses should implement provider-specific behavior while maintaining
    a consistent interface.
    """

    def __init__(self, **kwargs):
        """Initialize the protocol with any provider-specific configuration."""
        super().__init__()
        self.config = kwargs

    @classmethod
    def get_for_llm(cls, llm: BaseLanguageModel) -> "LlmProtocolBase":
        """
        Get the appropriate protocol implementation for a specific LLM.

        Args:
            llm: The LLM instance for which to retrieve the protocol

        Returns:
            LlmProtocolBase: The protocol implementation for the specified LLM
        """
        # Check if we have a registered protocol for this LLM name
        if hasattr(llm, "name") and llm.name in _LLM_PROTOCOL_REGISTRY:
            protocol_class = _LLM_PROTOCOL_REGISTRY[llm.name]
            return protocol_class()

        # Default to OpenAI protocol if no specific protocol is registered
        return OpenAILlmProtocol()


class OpenAILlmProtocol(
    LlmProtocolBase, OpenAIAttachmentMixin, OpenAITemplateBuilderMixin
):
    """
    OpenAI LLM protocol implementation.

    Combines the base protocol functionality with OpenAI-specific attachment
    handling and any other OpenAI-specific protocol requirements.
    """

    def __init__(self, **kwargs):
        """Initialize OpenAI protocol with OpenAI-specific configuration."""
        super().__init__(**kwargs)
        self.provider = "openai"
