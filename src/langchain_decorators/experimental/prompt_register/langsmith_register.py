from langchain_decorators.experimental.prompt_register.base import (
    PromptRegisterBase,
    register_prompt_register,
)
from langchain.prompts import BasePromptTemplate
from langsmith import AsyncClient


class LangSmithPromptRegister(PromptRegisterBase):

    async def register_prompt(self, prompt_id: str, prompt: BasePromptTemplate) -> str:
        """Register a prompt and return its ID."""
        client = AsyncClient()
        _saved_prompt = await client.pull_prompt(prompt_id)
        response = await client.push_prompt(prompt_id, object=prompt)
        return response.id

    async def get_prompt(self, prompt_id: str) -> BasePromptTemplate | None:
        """Get a prompt by its ID."""
        client = AsyncClient()
        response = await client.pull_prompt(prompt_id)
        return response.prompt if response else None


register_prompt_register(
    "langsmith", LangSmithPromptRegister()
)  # Lazy load LangSmithPromptRegister when needed


if __name__ == "__main__":

    async def main():
        from langchain_decorators.prompt_decorator import llm_prompt
        from langchain.prompts import PromptTemplate

        @llm_prompt
        def test_prompt(name: str) -> str:
            """
            Hello {name}, this is a test prompt.
            """

        # print(test_prompt(name="World"))
        prompt = test_prompt.build_chain(name="my_prompt").prompt.get_final_template(
            name="my_prompt"
        )
        register = LangSmithPromptRegister()
        prompt_id = await register.register_prompt("my_prompt_1", prompt=prompt)
        print(f"Registered prompt ID: {prompt_id}")

        retrieved_prompt = await register.get_prompt(prompt_id)
        print(f"Retrieved prompt: {retrieved_prompt.template}")

    import asyncio

    asyncio.run(main())
