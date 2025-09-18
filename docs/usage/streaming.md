# Streaming Features in LangChain Decorators

## Overview

Streaming is a powerful feature in LangChain Decorators that allows you to handle long-running tasks and receive output incrementally. This is particularly useful when working with large datasets or when you want to provide real-time feedback to users.

## Enabling Streaming

To enable streaming in your prompts, you need to define your prompt as an asynchronous function and set the `capture_stream` parameter to `True` in the `@llm_prompt` decorator.

### Example

Here’s a simple example of how to set up a streaming prompt:

```python
from langchain_decorators import StreamingContext, llm_prompt

@llm_prompt(capture_stream=True)
async def write_me_short_post(topic: str, platform: str = "twitter", audience: str = "developers"):
    """
    Write me a short header for my post about {topic} for {platform} platform. 
    It should be for {audience} audience.
    (Max 15 words)
    """
    pass
```

## Capturing the Stream

To capture the streamed output, you need to wrap the execution of your prompt in a `StreamingContext`. This allows you to define a callback function that will handle the incoming tokens.

### Example of Capturing Stream

```python
tokens = []

def capture_stream_func(new_token: str):
    tokens.append(new_token)

with StreamingContext(stream_to_stdout=True, callback=capture_stream_func):
    result = await write_me_short_post(topic="old movies")
    print("Stream finished ... we can distinguish tokens thanks to alternating colors")

print("\nWe've captured", len(tokens), "tokens🎉\n")
print("Here is the result:")
print(result)
```

## Benefits of Streaming

- **Real-time Feedback**: Users can see results as they are generated, improving the interactivity of your application.
- **Efficiency**: Streaming allows for processing large amounts of data without waiting for the entire output to be generated.
- **User Experience**: Enhances user experience by providing immediate responses, especially in applications like chatbots or interactive tools.

## Conclusion

Streaming in LangChain Decorators is a valuable feature that enhances the functionality and user experience of your applications. By following the examples provided, you can easily implement streaming in your projects.