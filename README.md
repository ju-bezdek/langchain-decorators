# LangChain Decorators ✨

lanchchain decorators is a layer on top of LangChain that provides syntactic sugar 🍭 for writing custom langchain prompts and chains

> **Note:** This is an unofficial addon to the langchain library. It's not trying to compete, just to make using it easier. Lot's of ideas here are totally opinionated

Here is a simple example of a code written with **LangChain Decorators ✨**

``` python

@llm_prompt
def write_me_short_post(topic:str, platform:str="twitter", audience:str = "developers")->str:
    """
    Write me a short header for my post about {topic} for {platform} platform. 
    It should be for {audience} audience.
    (Max 15 words)
    """
    return

# run it naturaly
write_me_short_post(topic="starwars")
# or
write_me_short_post(topic="starwars", platform="redit")
```

**Main principles and benefits:**

- more `pythonic` way of writing code
- write multiline prompts that won't break your code flow with indentation
- making use of IDE in-built support for **hinting**, **type checking** and **popup with docs** to quickly peek in the function to see the prompt, parameters it consumes etc.
- leverage all the power of 🦜🔗 LangChain ecosystem
- adding support for **optional parameters**
- easily share parameters between the prompts by binding them to one class

[Quick start](#quick-start)

- [Installation](#installation)
- [Examples](#examples)

[Prompt declarations](#prompt-declarations)

- [Documenting your prompt](#documenting-your-prompt)
- [Chat messages prompt](#chat-messages-prompt)
- [Optional sections](#optional-sections)
- [Output parsers](#output-parsers)

[LLM functions (OpenAI functions)](#llm-functions)

- [Functions provider](#functions-provider)
- [Dynamic function schemas](#dynamic-function-schemas)

[Simplified streaming](#simplified-streaming)

[Automatic LLM selection](#automatic-llm-selection)

[More complex structures](#more-complex-structures)

[Binding the prompt to an object](#binding-the-prompt-to-an-object)

[Defining custom settings](#defining-custom-settings)

[Debugging](#debugging)

[Passing a memory, callback, stop etc.](#passing-a-memory-callback-stop-etc)

[Other](#other)

- [More examples](#more-examples)
- [Contributing](#contributing)

## Quick start

### Installation

```bash
pip install langchain_decorators
```

### Examples

Good idea on how to start is to review the examples here:

- [jupyter notebook](example_notebook.ipynb)
- [colab notebook](https://colab.research.google.com/drive/1no-8WfeP6JaLD9yUtkPgym6x0G9ZYZOG#scrollTo=N4cf__D0E2Yk)

## Prompt declarations

By default the prompt is the whole function docs, unless you mark your prompt

### Documenting your prompt

We can specify what part of our docs is the prompt definition, by specifying a code block with **<prompt>** language tag

``` python
@llm_prompt
def write_me_short_post(topic:str, platform:str="twitter", audience:str = "developers"):
    """
    Here is a good way to write a prompt as part of a function docstring, with additional documentation for devs.

    It needs to be a code block, marked as a `<prompt>` language
    ```<prompt>
    Write me a short header for my post about {topic} for {platform} platform. 
    It should be for {audience} audience.
    (Max 15 words)
    ```

    Now only the code block above will be used as a prompt, and the rest of the docstring will be used as a description for developers.
    (It also has a nice benefit that IDE (like VS code) will display the prompt properly (not trying to parse it as markdown, and thus not showing new lines properly))
    """
    return 
```

### Chat messages prompt

For chat models is very useful to define prompt as a set of message templates... here is how to do it:

``` python
@llm_prompt
def simulate_conversation(human_input:str, agent_role:str="a pirate"):
    """
    ## System message
     - note the `:system` sufix inside the <prompt:_role_> tag
     

    ```<prompt:system>
    You are a {agent_role} hacker. You must act like one.
    You reply always in code, using python or javascript code block...
    for example:
    
    ... do not reply with anything else.. just with code - respecting your role.
    ```

    # human message 
    (we are using the real role that are enforced by the LLM - GPT supports system, assistant, user)
    ``` <prompt:user>
    Helo, who are you
    ```
    a reply:
    

    ``` <prompt:assistant>
    \``` python <<- escaping inner code block with \ that should be part of the prompt
    def hello():
        print("Argh... hello you pesky pirate")
    \```
    ```
    
    we can also add some history using placeholder
    ```<prompt:placeholder>
    {history}
    ```
    ```<prompt:user>
    {human_input}
    ```

    Now only the code block above will be used as a prompt, and the rest of the docstring will be used as a description for developers.
    (It also has a nice benefit that IDE (like VS code) will display the prompt properly (not trying to parse it as markdown, and thus not showing new lines properly))
    """
    pass

```

the roles here are model native roles (assistant, user, system for chatGPT)

## Optional sections

- you can define a whole section of your prompt that should be optional
- if any input in the section is missing, the whole section won't be rendered

the syntax for this is as follows:

``` python
@llm_prompt
def prompt_with_optional_partials():
    """
    this text will be rendered always, but

    {? anything inside this block will be rendered only if all the {value}s parameters are not empty (None | "")   ?}

    you can also place it in between the words
    this too will be rendered{? , but
        this  block will be rendered only if {this_value} and {this_value}
        are not empty?} !
    """
```

## Output parsers

- llm_prompt decorator natively tries to detect the best output parser based on the output type. (if not set, it returns the raw string)
- list, dict and pydantic outputs are also supported natively (automatically)

``` python
# this code example is complete and should run as it is

from langchain_decorators import llm_prompt

@llm_prompt
def write_name_suggestions(company_business:str, count:int)->list:
    """ Write me {count} good name suggestions for company that {company_business}
    """
    pass

write_name_suggestions(company_business="sells cookies", count=5)
```

## LLM functions

- currently supported only for the latest OpenAI chat models

- all you need to do is annotate your function with the @llm_function decorator.
- This will parse the description for LLM (first coherent paragraph is considered as function description)
- and aso parameter descriptions (Google, Numpy and Spihnx notations are supported for now)

- by default the docstring format is automatically resolved, but setting the format of the docstring can speed things up a bit.
        -  `auto` (default): the format is automatically inferred from the docstring
        -  `google`: the docstring is parsed as markdown (see [Google docstring format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html))
        -  `numpy`: the docstring is parsed as markdown (see [Numpy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html))
        -  `sphinx`: the docstring is parsed as sphinx format (see [Sphinx docstring format](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html))

### Enum arguments

The best way how to define enum is through type annotation using `Literal`:
```python
@llm_function
def do_magic(spell:str, strength:Literal["light","medium","strong"]):
    """
    Do some kind of magic

    Args:
        spell (str): spall text
        strength (str): the strength of the spell
    """

```

---
**Enum alternative to Literal**
To annotate an *"enum"* like argument, you can use this "typescript" like format: `["value_a" | "value_b"]` ... if will be parsed out.
This text will be a part of a description too... if you dont want it, you can use this notation as a type notation.
Example:
```
Args:
    message_type (["email" | "sms"]): type of a message  / channel how to send the message
        
```

Then you pass these functions as arguments to and  `@llm_prompt` (the argument must be named `functions` ‼️)
here you can pass any @llm_function there or a native LangChain tool

here is how to use it:

```python
from langchain.agents import load_tools
from langchian_decorators import llm_function, llm_prompt, GlobalSettings

@llm_function
def send_message(message:str, addressee:str=None, message_type:Literal["email", "whatsapp"]="email"):
    """ Use this if user asks to send some message

    Args:
        message (str): message text to send
        addressee (str): email of the addressee... in format firstName.lastName@company.com
        message_type (str, optional): style of message by platform
    """

    if message_type=="email":
        send_email(addressee, message)
    elif message_type=="whatsapp":
        send_whatsapp(addressee, message)
        

# load some other tools from langchain
list_of_other_tools = load_tools(
    tool_names=[...], 
    llm=GlobalSettings.get_current_settings().default_llm)

@llm_prompt
def do_what_user_asks_for(user_input:str, functions:List[Union[Callable,BaseTool]]):
    """ 
    ```<prompt:system>
    Your role is to be a helpful asistant.
    ```
    ```<prompt:user>
    {user_input}
    ```
    """

user_input="Yo, send an email to John Smith that I will be late for the meeting"
result = do_what_user_asks_for(
        user_input=user_input, 
        functions=[send_message, *list_of_other_tools]
    )

if result.is_function_call:
    result.execute()
else:
    print(result.output_text)

```

> Additionally you can also add a `function_call` argument to your LLM prompt to control GPT behavior.
> - if you set the value to "none" - it will disable the function call for the moment, but it can still see them (useful do to some reasoning/planning before calling the function)
> - if you set the value to "auto" - GPT will choose to use or to to use the functions
> - if you set the value to a name of function / or the function it self (decorators will handle resolving the same name as used in schema) it will force GPT to use that function

If you use functions argument, the output will be always `OutputWithFunctionCall`

``` python
class OutputWithFunctionCall(BaseModel):
    output_text:str
    output:T
    function_name:str =None
    function_arguments:Union[Dict[str,Any],str,None]
    function:Callable = None
    function_async:Callable = None
    
    @property
    def is_function_call(self):
        ...
    
    @property
    def support_async(self):
        ...
    
    @property
    def support_sync(self):
        ...

    async def execute_async(self):
       """Executes the function asynchronously."""
       ...
        
    def execute(self):
        """ Executes the function synchronously. 
        If the function is async, it will be executed in a event loop.
        """
        ...
     def to_function_message(self, result=None):
        """
        Converts the result to a FunctionMessage... 
        you can override the result collected via execute with your own
        """
        ...
```

If you want to see how the schema has been build, you can use `get_function_schema` method that is added to the function by the decorator:

```python
from langchain_decorators import get_function_schema
@llm_function
def my_func(arg1:str):
    ...

f_schema = get_function_schema(my_func.get_function_schema) 
print(f_schema)

```

In order to add the result to memory / agent_scratchpad you can use `to_function_message` to generate a FunctionMessage that LLM will interpret as a Tool/Function result

### Functions provider

Functions provider enables you to provide set of llm functions more dynamically, for example list of functions - based on the input.
It also enables you to give a unique name to each function for this LLM run. This might be useful for two reasons:

- avoid naming conflicts, if you are combining multiple general purpose functions
- further guidance/hinting of LLM model

### Dynamic function schemas

Function schemas (and especially their descriptions) are crucial tools to guide LLM. If you enable dynamic function declaration, you can (re)use the same prompt attributes for the main prompt also in the llm_function scheme:

```python

@llm_function(dynamic_schema=True)
def db_search(query_input:str):
    """
    This function is useful to search in our database.
    {?Here are some examples of data available:
    {closest_examples}?}
    """

@llm_prompt
def run_agent(query_input:str, closest_examples:str, functions):
    """
    Help user. Use a function when appropriate
    """

closest_examples = get_closest_examples()
run_agent(query_input, closest_examples, functions=[db_search, ...])
```

this is just for illustration, fully executable example is available [here, in code examples](code_examples/dynamic_function_schema.py)

## Simplified streaming

If we want to leverage streaming:

- we need to define prompt as async function
- turn on the streaming on the decorator, or we can define PromptType with streaming on
- capture the stream using StreamingContext

This way we just mark which prompt should be streamed, not needing to tinker with what LLM should we use, passing around the creating and distribute streaming handler into particular part of our chain... just turn the streaming on/off on prompt/prompt type...

The streaming will happen only if we call it in streaming context ... there we can define a simple function to handle the stream

``` python
# this code example is complete and should run as it is

from langchain_decorators import StreamingContext, llm_prompt

# this will mark the prompt for streaming (useful if we want stream just some prompts in our app... but don't want to pass distribute the callback handlers)
# note that only async functions can be streamed (will get an error if it's not)
@llm_prompt(capture_stream=True) 
async def write_me_short_post(topic:str, platform:str="twitter", audience:str = "developers"):
    """
    Write me a short header for my post about {topic} for {platform} platform. 
    It should be for {audience} audience.
    (Max 15 words)
    """
    pass



# just an arbitrary  function to demonstrate the streaming... wil be some websockets code in the real world
tokens=[]
def capture_stream_func(new_token:str):
    tokens.append(new_token)

# if we want to capture the stream, we need to wrap the execution into StreamingContext... 
# this will allow us to capture the stream even if the prompt call is hidden inside higher level method
# only the prompts marked with capture_stream will be captured here
with StreamingContext(stream_to_stdout=True, callback=capture_stream_func):
    result = await run_prompt()
    print("Stream finished ... we can distinguish tokens thanks to alternating colors")


print("\nWe've captured",len(tokens),"tokens🎉\n")
print("Here is the result:")
print(result)
```

### Automatic LLM selection

In real life there might be situations, where the context would grow over the window of the base model you're using (for example long chat history)...
But since this might happen only some times, it would be great if only in this scenario the (usually more expensive) model with bigger context window would be used, and otherwise we'd use the cheaper one.

Now you can do it with LlmSelector

```python
from langchain_decorators import  LlmSelector
my_llm_selector = LlmSelector(
            generation_min_tokens=0, # how much token at min. I for generation I want to have as a buffer
            prompt_to_generation_ratio=1/3 # what percentage of the prompt length should be used for generation buffer 
        )\
        .with_llm_rule(ChatGooglePalm(),max_tokens=512)\  # ... if you want to use LLM whose window is not defined in langchain_decorators.common.MODEL_LIMITS (only OpenAI and Anthropic are there)
        .with_llm(ChatOpenAI(model = "gpt-3.5-turbo"))\   # these models are known, therefore we can just pass them and the max window will be resolved
        .with_llm(ChatOpenAI(model = "gpt-3.5-turbo-16k-0613"))\ 
        .with_llm(ChatOpenAI(model = "claude-v1.3-100k"))
```

This class allows you to define a sequence of LLMs with a rule based on the length of the prompt, and expected generation length... and only after the threshold will be passed, the more expensive model will be used automatically.

You can define it into GlobalSettings:

``` python
langchain_decorators.GlobalSettings.define_settings(
        llm_selector = my_llm_selector # pass the selector into global settings
    )

```

> **Note:** as of version v0.0.10 you there the LlmSelector is in the default settings predefined.
> You can override it by providing you own, or setting up the default LLM or default streaming LLM

Or into specific prompt type:

```python
from langchain_decorators import PromptTypes

class MyCustomPromptTypes(PromptTypes):
    MY_TUBO_PROMPT=PromptTypeSettings(llm_selector = my_llm_selector)

```

### More complex structures

For dict / pydantic you need to specify the formatting instructions...
this can be tedious, that's why you can let the output parser generate you the instructions based on the model (pydantic)

``` python
from langchain_decorators import llm_prompt
from pydantic import BaseModel, Field


class TheOutputStructureWeExpect(BaseModel):
    name:str = Field (description="The name of the company")
    headline:str = Field( description="The description of the company (for landing page)")
    employees:list[str] = Field(description="5-8 fake employee names with their positions")

@llm_prompt()
def fake_company_generator(company_business:str)->TheOutputStructureWeExpect:
    """ Generate a fake company that {company_business}
    {FORMAT_INSTRUCTIONS}
    """
    return

company = fake_company_generator(company_business="sells cookies")

# print the result nicely formatted
print("Company name: ",company.name)
print("company headline: ",company.headline)
print("company employees: ",company.employees)

```

## Binding the prompt to an object

``` python
from pydantic import BaseModel
from langchain_decorators import llm_prompt

class AssistantPersonality(BaseModel):
    assistant_name:str
    assistant_role:str
    field:str

    @property
    def a_property(self):
        return "whatever"

    def hello_world(self, function_kwarg:str=None):
        """
        We can reference any {field} or {a_property} inside our prompt... and combine it with {function_kwarg} in the method
        """

    
    @llm_prompt
    def introduce_your_self(self)->str:
        """
        ``` <prompt:system>
        You are an assistant named {assistant_name}. 
        Your role is to act as {assistant_role}
        ```
        ```<prompt:user>
        Introduce your self (in less than 20 words)
        ```
        """

    

personality = AssistantPersonality(assistant_name="John", assistant_role="a pirate")

print(personality.introduce_your_self(personality))
```

## Defining custom settings

Here we are just marking a function as a prompt with `llm_prompt` decorator, turning it effectively into a LLMChain. Instead of running it

Standard LLMchain takes much more init parameter than just inputs_variables and prompt... here is this implementation detail hidden in the decorator.
Here is how it works:

1. Using **Global settings**:

    ``` python
    # define global settings for all prompty (if not set - chatGPT is the current default)
    from langchain_decorators import GlobalSettings

    GlobalSettings.define_settings(
        default_llm=ChatOpenAI(temperature=0.0), this is default... can change it here globally
        default_streaming_llm=ChatOpenAI(temperature=0.0,streaming=True), this is default... can change it here for all ... will be used for streaming
    )
    ```

2. Using predefined **prompt types**

    ``` python
    #You can change the default prompt types
    from langchain_decorators import PromptTypes, PromptTypeSettings

    PromptTypes.AGENT_REASONING.llm = ChatOpenAI()

    # Or you can just define your own ones:
    class MyCustomPromptTypes(PromptTypes):
        GPT4=PromptTypeSettings(llm=ChatOpenAI(model="gpt-4"))

    @llm_prompt(prompt_type=MyCustomPromptTypes.GPT4) 
    def write_a_complicated_code(app_idea:str)->str:
        ...

    ```

3. Define the settings **directly in the decorator**

    ``` python
    from langchain.llms import OpenAI

    @llm_prompt(
        llm=OpenAI(temperature=0.7),
        stop_tokens=["\nObservation"],
        ...
        )
    def creative_writer(book_title:str)->str:
        ...
    ```

### Passing a memory, callback, stop, etc

To pass any of these, just declare them in the function (or use kwargs to pass anything)

(They do not necessarily need to be declared, but it is a good practice if you are going to use them)

```python

@llm_prompt()
async def write_me_short_post(topic:str, platform:str="twitter", memory:SimpleMemory = None):
    """
    {history_key}
    Write me a short header for my post about {topic} for {platform} platform. 
    It should be for {audience} audience.
    (Max 15 words)
    """
    pass

await write_me_short_post(topic="old movies")

```

## Debugging

### Logging to console

There are several options how to control the outputs logged into console.
The easiest way is to define ENV variable: `LANGCHAIN_DECORATORS_VERBOSE` and set it to "true"

You can also control this programmatically by defining your global settings as shown [here](#defining-custom-settings)

The last option is to control it per each case, simply by turing on verbose mode on prompt:
```
@llm_prompt(verbose=True)
def your_prompt(param1):
  ...
```

### Using PromptWatch.io

PromptWatch io is a platform to track and trace details about everything that is going on in langchain executions.
It allows a single line drop in integration, just by wrapping your entry point code in
```
with PromptWatch():
    run_your_code()
```
Learn more about PromptWatch here: [www.promptwatch.io](https://www.promptwatch.io)

## Other

- this project is dependant on [langchain](https://github.com/hwchase17/langchain) (obviously)
- as well as on [promptwatch](https://github.com/blip-solutions/promptwatch-client), which make it easy to track and store to logs, track changes in prompts and compare them by running unit tests over the prompts...

### More examples

- these and few more examples are also available in the [examples notebook here](https://colab.research.google.com/drive/1no-8WfeP6JaLD9yUtkPgym6x0G9ZYZOG#scrollTo=N4cf__D0E2Yk)
- including the [ReAct Agent re-implementation](https://colab.research.google.com/drive/1no-8WfeP6JaLD9yUtkPgym6x0G9ZYZOG#scrollTo=3bID5fryE2Yp) using purely langchain decorators

## Contributing

feedback, contributions and PR are welcomed 🙏