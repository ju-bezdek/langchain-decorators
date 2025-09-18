import asyncio
from langchain_decorators import LlmChatSession
from langchain_decorators.function_decorator import llm_function
from langchain_decorators.llm_tool_use import ToolCall
from langchain_decorators.prompt_decorator import llm_prompt
from pydantic import BaseModel

# Example of  LlmChatSession use
# to show of how to simulate a chat session with the LLM, where the messages are stored in the session


class StructuredAnalysis(BaseModel):
    is_problem: bool
    short_description: str


@llm_prompt
def prompt_example(user_input: str) -> StructuredAnalysis:
    """
    ```<prompt:system>
    Your task is to analyze the circumstances described by the user and determine if they represent a problem or not.
    ```
    ```<prompt:user>
    {user_input}
    ```
    """


async def main_async_with_tools() -> str:

    user_input = "My new coworker replies to emails way too fast… like within a minute"
    with LlmChatSession() as session:
        # This will force llm not to use structured output, although its declared as such... but write unbounded output first
        session.suppress_structured_output()
        print("Thinking:", prompt_example(user_input=user_input))
        # Now we enabled it
        session.suppress_structured_output(False)
        res = prompt_example(user_input=user_input)
        if res.is_problem:
            print("Problem detected:", res.short_description)
        else:
            print("No problem detected:", res.short_description)


if __name__ == "__main__":
    asyncio.run(main_async_with_tools())
