# Settings

In this section, we will cover the various settings available in the LangChain Decorators project. These settings allow you to customize the behavior of the decorators and the overall functionality of the library.

## Global Settings

The `GlobalSettings` class is used to define global configurations for the LangChain Decorators. You can set default values for various parameters that will apply across all prompts and functions.

### Example

```python
from langchain_decorators import GlobalSettings

GlobalSettings.define_settings(
    default_llm=ChatOpenAI(temperature=0.7),
    default_streaming_llm=ChatOpenAI(temperature=0.7, streaming=True),
)
```

## Prompt Types

You can define custom prompt types to specify different settings for various use cases. This allows for more granular control over how prompts are processed.

### Example

```python
from langchain_decorators import PromptTypes, PromptTypeSettings

class MyCustomPromptTypes(PromptTypes):
    GPT4 = PromptTypeSettings(llm=ChatOpenAI(model="gpt-4"))
```

## Custom Settings in Decorators

You can also define settings directly in the decorator for specific functions. This allows you to override global settings for individual use cases.

### Example

```python
from langchain.llms import OpenAI

@llm_prompt(
    llm=OpenAI(temperature=0.5),
    stop_tokens=["\nObservation"],
)
def creative_writer(book_title: str) -> str:
    ...
```

## Memory and Callbacks

You can pass memory and callback functions to your prompts to maintain state or handle specific events during execution.

### Example

```python
@llm_prompt()
async def write_me_short_post(topic: str, platform: str = "twitter", memory: SimpleMemory = None):
    """
    Write me a short header for my post about {topic} for {platform} platform.
    """
    pass
```

## Debugging Settings

For debugging purposes, you can enable verbose logging to get more insights into the execution of your prompts and functions.

### Example

```python
@llm_prompt(verbose=True)
def your_prompt(param1):
    ...
```

By utilizing these settings, you can tailor the LangChain Decorators library to better fit your specific needs and improve the overall functionality of your applications.