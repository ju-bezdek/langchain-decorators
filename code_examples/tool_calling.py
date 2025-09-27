import asyncio
from datetime import date
import datetime
from langchain_decorators import LlmChatSession
from langchain_decorators.function_decorator import llm_function
from langchain_decorators.llm_tool_use import ToolCall
from langchain_decorators.prompt_decorator import llm_prompt
from pydantic import BaseModel
from langchain.tools import tool


@tool(parse_docstring=True)
def langchain_tool(instructions: str):
    """
    Use this if your are told to use langchain tool

    Args:
        instructions: instructions to pass to the function
    """
    return instructions + " executed"


class Agent(BaseModel):
    customer_name: str
    memory: list = []

    @property
    def current_time(self):
        return datetime.datetime.now().isoformat()

    @llm_function()
    def express_emotion(self, emoji: str) -> str:
        """Use this tool to express your emotion as an emoji

        Args:
            emoji (str): the emoji to express

        """

    @llm_prompt
    def main_prompt(self, user_input: str):
        """
        ```<prompt:system>
        You are a friendly but shy assistant. Try to reply with the least amount of words possible.

        Context:
        customer name is {customer_name}
        current time is {current_time}
        ```
        ```<prompt:placeholder>
        {messages}
        ```
        ```<prompt:user>
        {user_input}
        ```
        """

    def start(self):
        while True:
            print(self.invoke(user_input=input("Enter your message: ")))

    def invoke(self, user_input: str):
        with LlmChatSession(
            tools=[self.express_emotion, langchain_tool], message_history=self.memory
        ) as session:
            res = self.main_prompt(user_input=user_input)
            session.execute_tool_calls()  # This automatically calls the tools and adds the tool responses to the chat history
            return res


if __name__ == "__main__":
    Agent(customer_name="John").start()
