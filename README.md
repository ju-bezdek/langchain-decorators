# LangChain Decorators ✨

LangChain Decorators is a lightweight layer on top of LangChain that provides syntactic sugar 🍭 for writing custom prompts and chains.

> Note: This is an unofficial add-on to the LangChain library. It's not trying to compete—just to make using it easier. Lots of ideas here are opinionated.

Here is a simple example written with LangChain Decorators ✨

```python
@llm_prompt
def write_me_short_post(topic: str, platform: str = "twitter", audience: str = "developers") -> str:
    """
    Write me a short header for my post about {topic} for the {platform} platform.
    It should be for a {audience} audience.
    (Max 15 words)
    """
    return

# Run it naturally
write_me_short_post(topic="starwars")
# or
write_me_short_post(topic="starwars", platform="reddit")
```

Main principles and benefits:
- A more Pythonic way of writing prompts
- Write multiline prompts that don't break your code flow with indentation
- Leverage IDE support for hinting, type checking, and doc popups to quickly see prompts and parameters
- Keep the power of the 🦜🔗 LangChain ecosystem
- Add support for optional parameters
- Easily share parameters between prompts by binding them to a class

[Quick start](#quick-start)

- [Installation](#installation)
- [Examples](#examples)

[Prompt declarations](#prompt-declarations)

- [Documenting your prompt](#documenting-your-prompt)
- [Chat messages in prompt](#chat-messages-in-prompt)
- [Optional sections](#optional-sections)
- [Output parsers](#output-parsers)

[Chat sessions / threads](#chat-sessions--threads)

[Tool calling](#tool-calling)

- [Enum arguments](#enum-arguments)

[Simplified streaming](#simplified-streaming)

[More complex structures](#more-complex-structures)

[Binding the prompt to an object](#binding-the-prompt-to-an-object)

[Defining custom settings](#defining-custom-settings)

[Debugging](#debugging)

## Quick start

### Installation

```bash
pip install langchain_decorators
```

### Examples

A good way to start is to review the examples here:

- [Jupyter notebook](example_notebook.ipynb)

## Prompt declarations

Define a prompt by creating a function with arguments as inputs and the function docstring as the prompt template:

```python
@llm_prompt
def write_me_short_post(topic: str, platform: str = "twitter", audience: str = "developers"):
    """
    Write me a short header for my post about {topic} for the {platform} platform.
    It should be for a {audience} audience.
    (Max 15 words)
    """
```

This is the default way to declare a prompt, and it’s translated into a chat with a single user message.

If you want to declare a prompt with multiple messages (which is common), you can declare multiple messages as special code blocks inside the function docstring:

```python
@llm_prompt
def write_me_short_post(topic: str, platform: str = "twitter", audience: str = "developers"):
    """
    ```<prompt:system>
    You are a social media manager.
    ```
    ```<prompt:user>
    Write me a short header for my post about {topic} for the {platform} platform.
    It should be for a {audience} audience.
    (Max 15 words)
    ```
    ```<prompt:assistant>
    I need to think about it.
    ```
    """
```

The pattern is a series of consecutive code blocks with a "language" tag in this format: `<prompt:[message-role]>`.

### Documenting your prompt

You can specify which part of your docstring is the prompt by using a code block with the `<prompt>` tag:

```python
@llm_prompt
def write_me_short_post(topic: str, platform: str = "twitter", audience: str = "developers"):
    """
    Here is a good way to write a prompt as part of a function docstring, with additional documentation for devs.

    It needs to be a code block marked with `<prompt>`.
    ```<prompt:user>
    Write me a short header for my post about {topic} for the {platform} platform.
    It should be for a {audience} audience.
    (Max 15 words)
    ```

    Only the code block above will be used as a prompt; the rest of the docstring will be used as documentation for developers.
    It also has a nice benefit in IDEs (like VS Code), which will display the prompt properly (without trying to parse it as Markdown).
    """
    return
```

### Chat messages in prompt

For chat models it’s useful to define the prompt as a set of message templates. Here’s how:

```python
@llm_prompt
def simulate_conversation(human_input: str, agent_role: str = "a pirate"):
    """
    ## System message
    - Note the `:system` suffix inside the <prompt:_role_> tag

    ```<prompt:system>
    You are a {agent_role} hacker. You must act like one.
    Always reply in code, using Python or JavaScript code blocks only.
    for example:
    ```
    ```<prompt:user>
    Hello, who are you?
    ```
    A reply:

    ```<prompt:assistant>
    \```python
    def hello():
        print("Argh... hello you pesky pirate")
    \```
    ```

    We can also add some history using a placeholder:
    ```<prompt:placeholder>
    {history}
    ```
    ```<prompt:user>
    {human_input}
    ```

    Only the code blocks above will be used as a prompt; the rest of the docstring is documentation for developers.
    """
    pass
```

The roles here are the model’s native roles (assistant, user, system for ChatGPT-compatible models).

## Optional sections

- Define a section of your prompt that should be optional.
- If any referenced input in the section is missing or empty (None or ""), the whole section won’t be rendered.

Syntax:

```python
@llm_prompt
def prompt_with_optional_partials():
    """
    This text will always be rendered, but

    {? anything inside this block will be rendered only if all referenced {value}s
       are not empty (None | "") ?}

    You can also place it in-line:
    this too will be rendered{?, but
        this block will be rendered only if {this_value} and {that_value}
        are not empty ?}!
    """
```

## Output parsers

- The llm_prompt decorator tries to detect the best output parser based on the return type (if not set, it returns the raw string).
- list, dict, and pydantic outputs are supported natively (automatically).

```python
# this example should run as-is

from langchain_decorators import llm_prompt

@llm_prompt
def write_name_suggestions(company_business: str, count: int) -> list:
    """Write {count} good name suggestions for a company that {company_business}."""
    pass

write_name_suggestions(company_business="sells cookies", count=5)
```

## Chat sessions / threads

For agentic workflows, you often need to keep track of messages in a single session/thread. Wrap calls in LlmChatSession:

```python
from langchain_decorators import llm_prompt, LlmChatSession

@llm_prompt
def my_prompt(user_input):
    """
    ```<prompt:system>
    Be a pirate assistant that can reply with 5 words max.
    ```
    ```<prompt:placeholder>
    {messages}
    ```
    ```<prompt:user>
    {user_input}
    ```
    """

with LlmChatSession() as session:
    while True:
        response = my_prompt(user_input=input("Enter your message: "))
        print(response)
```

## Tool calling

Implementing tool calling with LangChain can be a bit of a hassle: you need to manage chat history, collect tool response messages, and add them back to history.

Decorators offer a simplified variant that manages this for you.

You can use either the native LangChain `@tool` decorator or `@llm_function`, which offers a few quirks like handling bound methods, allowing dynamic tool schema generation (especially useful to pass in dynamic argument domain values), etc.

```python
import datetime
from pydantic import BaseModel
from langchain_decorators import llm_prompt, llm_function, LlmChatSession
# from langchain.tools import tool as langchain_tool  # example placeholder if needed

class Agent(BaseModel):
    customer_name: str  # bound properties/fields on instances are accessible in the prompt

    @property
    def current_time(self):
        return datetime.datetime.now().isoformat()

    @llm_function
    def express_emotion(self, emoji: str) -> str:
        """Use this tool to express your emotion as an emoji."""
        return print(emoji)

    @llm_prompt
    def main_prompt(self, user_input: str):
        """
        ```<prompt:system>
        You are a friendly but shy assistant. Try to reply with the least number of words possible.

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
        with LlmChatSession(tools=[self.express_emotion]):  # add extra tools like `langchain_tool` if needed
            while True:
                print(self.main_prompt(user_input=input("Enter your message: ")))
                # Automatically call tools and add tool responses to history:
                # session.execute_tool_calls() is handled by the session context if available
```

### Enum arguments

The simplest way to define an enum is via type annotation using `Literal`:

```python
from typing import Literal

@llm_function
def do_magic(spell: str, strength: Literal["light", "medium", "strong"]):
    """
    Do some kind of magic.

    Args:
        spell (str): spell text
        strength (str): the strength of the spell
    """
```

For dynamic domain values, use:

```python
@llm_function(dynamic_schema=True)
def do_magic(spell: str, strength: Literal["light", "medium", "strong"]):
    """
    Do some kind of magic.

    Args:
        spell (Literal{spells_unlocked}): spell text
        strength (Literal["light","medium","strong"]): the strength of the spell
    """

with LlmChatSession(tools=[do_magic], context={"spells_unlocked": spells_unlocked}):
    my_prompt(user_message="Make it levitate")
```

Info: this works by parsing any list of values ["val1", "val2"]. You can also use `|` as a separator and quotes. The `Literal` prefix in docs is optional and used for clarity.

## Simplified streaming

If you want to leverage streaming:

- Define the prompt as an async function.
- Turn on streaming in the decorator (or via a PromptType).
- Capture the stream using StreamingContext.

This lets you mark which prompts should be streamed without wiring LLMs and callbacks throughout your code. Streaming happens only if the call is executed inside a StreamingContext.

```python
# this example should run as-is

from langchain_decorators import StreamingContext, llm_prompt

# Mark the prompt for streaming (only async functions can be streamed)
@llm_prompt(capture_stream=True)
async def write_me_short_post(topic: str, platform: str = "twitter", audience: str = "developers"):
    """
    Write me a short header for my post about {topic} for the {platform} platform.
    It should be for a {audience} audience.
    (Max 15 words)
    """
    pass

# Simple function to demonstrate streaming; replace with websockets in real apps
tokens = []
def capture_stream_func(new_token: str):
    tokens.append(new_token)

# Capture the stream from prompts marked with capture_stream
async def run():
    with StreamingContext(stream_to_stdout=True, callback=capture_stream_func):
        result = await write_me_short_post(topic="cookies")
        print("Stream finished ... tokens are colorized alternately")
    print("\nWe've captured", len(tokens), "tokens 🎉\n")
    print("Here is the result:")
    print(result)
```

## More complex structures

For dict/pydantic outputs you need formatting instructions. You can let the output parser generate instructions based on a pydantic model.

```python
from langchain_decorators import llm_prompt
from pydantic import BaseModel, Field

class TheOutputStructureWeExpect(BaseModel):
    name: str = Field(description="The name of the company")
    headline: str = Field(description="The description of the company (for the landing page)")
    employees: list[str] = Field(description="5–8 fake employee names with their positions")

@llm_prompt()
def fake_company_generator(company_business: str) -> TheOutputStructureWeExpect:
    """
    Generate a fake company that {company_business}
    {FORMAT_INSTRUCTIONS}
    """
    return

company = fake_company_generator(company_business="sells cookies")

# print the result nicely formatted
print("Company name:", company.name)
print("Company headline:", company.headline)
print("Company employees:", company.employees)
```

## Binding the prompt to an object

```python
from pydantic import BaseModel
from langchain_decorators import llm_prompt

class AssistantPersonality(BaseModel):
    assistant_name: str
    assistant_role: str
    field: str

    @property
    def a_property(self):
        return "whatever"

    def hello_world(self, function_kwarg: str | None = None):
        """
        We can reference any {field} or {a_property} inside our prompt and combine it with {function_kwarg}.
        """

    @llm_prompt
    def introduce_your_self(self) -> str:
        """
        ```<prompt:system>
        You are an assistant named {assistant_name}.
        Your role is to act as {assistant_role}.
        ```
        ```<prompt:user>
        Introduce yourself (in fewer than 20 words).
        ```
        """

personality = AssistantPersonality(assistant_name="John", assistant_role="a pirate", field="N/A")
print(personality.introduce_your_self())
```

## Defining custom settings

Here we mark a function as a prompt with the `llm_prompt` decorator, effectively turning it into an LLMChain.

A standard LLMChain takes more init parameters than just input variables and prompt; in this implementation the decorator hides those details. You can control it in several ways:

1) Global settings:

```python
from langchain_decorators import GlobalSettings
from langchain.chat_models import ChatOpenAI

GlobalSettings.define_settings(
    default_llm=ChatOpenAI(temperature=0.0),  # default for non-streaming prompts
    default_streaming_llm=ChatOpenAI(temperature=0.0, streaming=True),  # default for streaming
)
```

2) Predefined prompt types:

```python
from langchain_decorators import PromptTypes, PromptTypeSettings
from langchain.chat_models import ChatOpenAI

PromptTypes.AGENT_REASONING.llm = ChatOpenAI()

# Or define your own:
class MyCustomPromptTypes(PromptTypes):
    GPT4 = PromptTypeSettings(llm=ChatOpenAI(model="gpt-4"))

@llm_prompt(prompt_type=MyCustomPromptTypes.GPT4)
def write_a_complicated_code(app_idea: str) -> str:
    ...
```

3) Settings directly in the decorator:

```python
from langchain.llms import OpenAI

@llm_prompt(
    llm=OpenAI(temperature=0.7),
    stop_tokens=["\nObservation"],
    # ...
)
def creative_writer(book_title: str) -> str:
    ...
```

## Debugging

### Logging to console

You can control console logging in several ways:
- Set the ENV variable `LANGCHAIN_DECORATORS_VERBOSE=true`
- Define global settings (see Defining custom settings)
- Turn on verbose mode on a specific prompt:

```python
@llm_prompt(verbose=True)
def your_prompt(param1):
    ...
```

### Support for LangSmith

Using `langchain_decorators` turns your prompts into first-class citizens in LangSmith. It creates chains named after your functions, making traces easier to interpret. Additionally, you can add tags:

```python
@llm_prompt(tags=["my_tag"])
def my_prompt(input_arg=...):
    """
    ...
```

## Contributing

Feedback, contributions, and PRs are welcome 🙏