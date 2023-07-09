
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
        
        self.conversation_memory=ConversationBufferMemory(memory_key="conversation_memory", chat_memory=conversation_history or ChatMessageHistory(), return_messages=True)
        # note: We cant use LangChainMemory here just yet, because we are using OpenAI functions and as of now, lanchain doesnt support them (add_ai_message adds only string)
        # list will do just fine for our purposes, but we need to add the output message on our own
        self.scratchpad=[]  #ConversationBufferMemory(memory_key="scratchpad", return_messages=True, input_key="user_input")
        
    
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
print(agent.run("What if the train was going at 100 km/hr?"))
print(agent.run("And what if the man wouldn't be moving?"))