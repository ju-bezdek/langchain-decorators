
from langchain_decorators import llm_prompt, llm_function, OutputWithFunctionCall, GlobalSettings
from langchain.schema import AIMessage 
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import LLMMathChain
from langchain.agents import Tool

GlobalSettings.define_settings(verbose=True)

class ChatAgent:

    def __init__(self, tools, conversation_history: ChatMessageHistory=None) -> None:                
        self.tools =tools
        self.max_iterations=8
        
        self.conversation_memory=ConversationBufferMemory(
            memory_key="conversation_memory", 
            chat_memory=conversation_history or ChatMessageHistory(), 
            return_messages=True)
        self.scratchpad=[]  
        
    
    @property
    def functions(self):    
        """ This will provide the final list of tools/functions to `think` prompt """    
        return [*self.tools, self.reply]
    
    @llm_prompt(functions_source="functions", verbose=True)
    def think(self, user_input:str, additional_instructions:str=None)->OutputWithFunctionCall:
        """
        # Conversation history
        ```<prompt:placeholder>
        {conversation_memory}
        ```

        # User input
        ```<prompt:user>
        {user_input}
        ```

         # System message with instructions... doesn't need to be first 😉
        ```<prompt:system>
        First lay down a plan what function we should utilize and in what order, than use the first function. Make sure to plan one calculation per step.
        {?{additional_instructions}?}
        ```

        # Reasoning scratchpad
        ```<prompt:placeholder>
        {scratchpad}
        ```
        """
        
    @llm_function
    def reply(self, reply_message:str)->str:
        """ Use this to reply to the user with final answer
        
        Args:
            reply_message (str): text what to send to the user
        """
        # We use this function to capture the reply, so we would know when to agent stopped its intermediate reasoning steps
        return AIMessage(content=reply_message)
     
        
    def run(self, user_input:str)->str:
        self.scratchpad.clear()

        initial_args={  
            "additional_instructions":"First lay down a complete plan of functions we need to use and how, than use the first function",
            "llm_selector_rule_key": "GPT4"
        }
        for i in range(self.max_iterations):
            output= self.think(user_input=user_input, **initial_args )
            self.scratchpad.append(output.output_message)
            initial_args={} #reset initial args
            if output.is_function_call:
                result = output.execute()
                if isinstance(result,  AIMessage):
                    # reply function returns this
                    self.conversation_memory.save_context({"user_input":user_input},{"reply":result.content})
                    return result
                else:
                    self.scratchpad.append(output.function_output_to_message())
                
            # we use reply function to capture the reply, so this is either reasoning, or smth is not going well




llm_math = LLMMathChain(llm=GlobalSettings.get_current_settings().default_llm)

# initialize the math tool
math_tool = Tool(
name='Calculator',
func=llm_math.run,
description='Useful for when you need to answer questions about math.'
)

agent = ChatAgent(tools=[math_tool])


print(agent.run("A 220 m long train is running at a speed of 60 km/hr. At what time will it pass a man who is running at 6 km/hr in the direction opposite to that in which the train is going?"))
# 1️⃣ agent iteration: (enforce use of GPT4 to make the plan)
#
# LLMSelector: Using 1-th LLM: gpt-4-0613
#
# Result:
# The plan is as follows:
#
# 1. Convert the speed of the train from km/hr to m/s by multiplying it by 5/18.
# 2. Convert the speed of the man from km/hr to m/s by multiplying it by 5/18.
# 3. Calculate the relative speed of the train and the man by adding the two speeds together (since they are moving in opposite directions).
# 4. Calculate the time it takes for the train to pass the man by dividing the length of the train by the relative speed.
#
# Let's start with the first step: converting the speed of the train from km/hr to m/s.
# fx: Calculator(60 * 5/18)
# Answer: 16.666666666666668>

# 2️⃣ agent call (using GPT3.5 just follow the instructions prepared by GPT4)
# LLMSelector: Using default LLM: gpt-3.5-turbo-0613
#
# Result:
# The speed of the train is 16.67 m/s.
#
# Now let's move on to the second step: converting the speed of the man from km/hr to m/s.
# fx: Calculator(6 * 5/18)
# Answer: 1.6666666666666667

# 3️⃣ agent call (using GPT3.5 just follow the instructions prepared by GPT4)
# Result:
# The speed of the man is 1.67 m/s.
#
# Next, let's move on to the third step: calculating the relative speed of the train and the man.
# fx: Calculator(16.67 + 1.67)
# Answer: 18.340000000000003

# 4️⃣ agent call (using GPT3.5 just follow the instructions prepared by GPT4)
# LLMSelector: Using default LLM: gpt-3.5-turbo-0613 
# Result:
# The relative speed of the train and the man is 18.34 m/s.

# Finally, let's move on to the fourth step: calculating the time it takes for the train to pass the man.
# fx: Calculator(220 / 18.34)
# Answer: 11.995637949836423

# 5️⃣ agent call (using GPT3.5 just follow the instructions prepared by GPT4)
# 🏁 Final Result:
# It will take approximately 12 seconds for the train to pass the man.

# Therefore, the train will pass the man at approximately 12 seconds.

