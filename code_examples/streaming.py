# this code example is complete and should run as it is

import asyncio
from langchain_decorators import StreamingContext, llm_prompt

@llm_prompt(capture_stream=True) # this will mark the prompt for streaming (usefull if we want stream just some prompts)
async def write_me_short_post(topic:str, platform:str="twitter", audience:str = "developers"):
    """
    Write me a short header for my post about {topic} for {platform} platform. 
    It should be for {audience} audience.
    (Max 15 words)
    """
    pass

async def run_prompt():
    return await write_me_short_post(topic="Releasing a new App that can do real magic!")

tokens=[]
def capture_stream_func(new_token:str):
    tokens.append(new_token)


async def main():
    with StreamingContext(stream_to_stdout=True, callback=capture_stream_func):
        result = await run_prompt()
        print("Stream finished ... we can distinguish tokens thanks to alternating colors")
        print("\nWe've captured",len(tokens),"tokens🎉\n")
        print("Here is the result:")
        print(result)

asyncio.run(main())

