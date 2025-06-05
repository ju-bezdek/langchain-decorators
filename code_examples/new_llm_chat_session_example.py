import asyncio
from langchain_decorators import LlmChatSession
from langchain_decorators.function_decorator import llm_function
from langchain_decorators.llm_tool_use import ToolCall
from langchain_decorators.prompt_decorator import llm_prompt

# Example of  LlmChatSession use
# to show of how to simulate a chat session with the LLM, where the messages are stored in the session

@llm_prompt
async def prompt_example(question: str) -> str:
    """
    ```<prompt:system>
    You are a helpful assistant.
    ```
    ```<prompt:user>
    Now reply as a pirate to the user question.
    ```
    ```<prompt:assistant>
    Arrr matey! I be ready to assist ye with yer question.
    ```
    ```<prompt:user>
    OK, stop being a pirate now.
    ```
    ```<prompt:user>
    Pls answer the question: {question}
    ```
    """
    pass


def optional_tool_error_handler(tool_call:ToolCall, exception: Exception) -> str:
    """Optional custom error handler for tool calls."""
    if tool_call.name == "foo":
        return f"Foo Error: {exception}"
    else:
        return exception # this make the exception reraise 
    

async def main_async_with_tools() -> str:

    @llm_function
    async def search_emails(
        query: str, max_results: int = 5
    ) -> list[str]:
        """
        Search for emails matching the query.
        """
        # Simulate email search
        return [f"Email_id:{i} hidden" for i in range(max_results)]
    with LlmChatSession(tools=[search_emails]) as session:
        for simulated_usr_msg in [
            "Can you search my emails?",
            "Ok, let's do it... search for john doe emails"]:
            response=None
            while not response:
                response = await prompt_example(question=simulated_usr_msg)
                print(response)


                # either handle the tool calls manually ... one by one
                for tool_call in session.last_response_tool_calls:
                    print(f"Tool call: {tool_call.name} with args: {tool_call.args}")
                    # execute the tool call manually: 
                    # await tool_call.execute_async()
                    # or 
                    # tool_call.invoke()
                    # or 
                    # res = tool_call(override_foo_arg="bar")
                    # res = postprocess_the_result(res)
                    # tool_call.set_result(res) 
                    # ... or one could add the result message to session directly
                    # session.add_message(tool_call.to_tool_message())

                # or use one liner to execute all tool calls and get the results
                await session.execute_tool_calls(error_handling="fail_safe", custom_error_handler=optional_tool_error_handler)

if __name__ == "__main__":
    asyncio.run(main_async_with_tools())
 
        


