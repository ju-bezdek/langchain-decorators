import logging
import requests
import langchain_decorators
from langchain_decorators import llm_prompt, LlmSelector
from langchain_decorators.schema import OutputWithFunctionCall
from langchain.schema import HumanMessage, AIMessage
from langchain.agents import load_tools


langchain_decorators.GlobalSettings.define_settings(
    verbose=True, 
    logging_level=logging.DEBUG,
    # llm_selector=LlmSelector(
    #         generation_min_tokens=0, # how much token at min. I for generation I want to have as a buffer
    #         prompt_to_generation_ratio=1/3 # what percentage of the prompt length should be used for generation buffer 
    #     )\
    #     .with_llm_rule(ChatGooglePalm(),max_tokens=512)\  # ... if you want to use LLM whose window is not defined in langchain_decorators.common.MODEL_LIMITS (only OpenAI and Anthropic are there)
    #     .with_llm(ChatOpenAI())\
    #     .with_llm(ChatOpenAI(model="gpt-3.5-turbo-16k-0613"))\
    #     .with_llm(ChatOpenAI(model="claude-v1.3-100k"))
    # )
    )

@llm_prompt
def get_names_and_sentiment(user_input:str)->str:
    """
    Summarize the key bullet points from this text:
    {user_input}
    """

response = requests.get("https://raw.githubusercontent.com/ju-bezdek/langchain-decorators/main/README.md")
langchain_decorators_readme = response.text[:5000]
get_names_and_sentiment(user_input=langchain_decorators_readme)

# Output:
#
# ... skippet a lot of text (debug mode)...
#
# LLMSelector: Using default LLM: gpt-3.5-turbo-0613 👈 automatically chosen default model based on the final prompt length


# Result:
# - LangChain Decorators is a layer on top of LangChain that provides syntactic sugar for writing custom langchain prompts and chains.
# - It offers a more pythonic way of writing code and allows for writing multiline prompts without breaking the code flow with indentation.
# - It leverages IDE in-built support for hinting, type checking, and popup with docs to quickly peek into the function and see the prompt and parameters.
# - It adds support for optional parameters and allows for easily sharing parameters between prompts by binding them to one class.
# - The package can be installed using pip.
# - Examples and documentation can be found in the provided Jupyter and Colab notebooks.
# - Prompt declarations can be specified using code blocks with the `<prompt>` language tag.
# - Chat messages prompts can be defined using message templates.
# - Optional sections can be defined in the prompt, which will only be rendered if all the specified parameters are not empty.
# - Output parsers are automatically detected based on the output type.

# > Finished chain

# > Entering get_names_and_sentiment prompt decorator chain

response = requests.get("https://raw.githubusercontent.com/hwchase17/chat-your-data/master/state_of_the_union.txt")
state_of_the_union = response.text
get_names_and_sentiment(user_input = state_of_the_union)

# Output:
#
# ... skippet a lot of text (debug mode)...
#
# LLMSelector: Using 1-th LLM: gpt-3.5-turbo-16k-0613 👈 automatically chosen bigger model based on the final prompt length

# Result:
# - The speech begins with acknowledgments to various political figures and the American people.
# - The focus then shifts to the recent conflict between Russia and Ukraine, with an emphasis on the strength and determination of the Ukrainian people.
# - The speaker outlines the actions taken by the United States and its allies to hold Russia accountable, including economic sanctions and military support for Ukraine.
# - The speech then transitions to domestic issues, such as the COVID-19 pandemic and the economic recovery efforts.
# - The speaker highlights the American Rescue Plan and its impact on job growth and economic relief for Americans.
# - Infrastructure investment is discussed as a means to create jobs and improve the country's competitiveness.
# - The need for tax reform and addressing income inequality is emphasized.
# - The speech touches on issues such as climate change, healthcare, voting rights, and immigration reform.
# - The speaker also addresses mental health, support for veterans, and efforts to combat cancer.
# - The speech concludes with a message of unity and optimism for the future of the United States.

# > Finished chain





import langchain_decorators
from langchain_decorators.schema import OutputWithFunctionCall
from langchain.schema import HumanMessage, AIMessage
from langchain.agents import load_tools


class ChatAgent:

    def __init__(self, tools) -> None:
                
        self.tools =tools
        self.max_iterations=8
        self.feedback=None
        self.scratchpad_messages=[]
        
   
    @llm_prompt(stop_tokens=["Tool output"] )
    async def think(self, user_input:str, functions, function_call:str="auto")->OutputWithFunctionCall:
        """
        ## System
        ```<prompt:system>
        Be a good assistant. Use functions to solve the task they are designed for. After that, formulate a reply.

        ```
        ## Reasoning scratchpad
        ```<prompt:placeholder>
        {memory}
        ```
        """
        
    async def run(self, user_input:str)->str:
        self.scratchpad_messages.append(HumanMessage(content=user_input))
        for i in range(self.max_iterations):
            output=await self.think(user_input=user_input, functions=self.tools)
            self.scratchpad_messages.append(output.output_message)
            if output.is_function_call:
                await output.execute_async()
                self.scratchpad_messages.append(output.to_function_message())
                
            else:
                output_text = output.output_text
                self.scratchpad_messages.append(AIMessage(content=output_text))
                return output_text


tools = load_tools(["pal-math", "serpapi"], llm=langchain_decorators.GlobalSettings.default_llm)
agent = ChatAgent(tools)









