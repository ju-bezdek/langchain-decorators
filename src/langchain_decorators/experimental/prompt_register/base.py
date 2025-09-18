from abc import ABC, abstractmethod
from langchain.prompts import BasePromptTemplate


class PromptRegisterBase(ABC):
    """Base class for prompt register."""

    @abstractmethod
    async def register_prompt(self, prompt_id: str, prompt: BasePromptTemplate) -> None:
        """Register a prompt and return its ID."""
        pass

    @abstractmethod
    async def get_prompt(self, prompt_id: str) -> BasePromptTemplate | None:
        """Get a prompt by its ID."""
        pass


AVAILABLE_PROMPT_REGISTERS = {}


def register_prompt_register(name: str, register: PromptRegisterBase):
    """Register a prompt register."""
    AVAILABLE_PROMPT_REGISTERS[name] = register
    return register
