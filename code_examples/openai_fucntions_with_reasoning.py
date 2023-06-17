
from langchain_decorators import llm_prompt, llm_function
from src.langchain_decorators.common import GlobalSettings


from langchain_decorators import llm_prompt

GlobalSettings.define_settings(verbose=True)



@llm_function
def say_yes(reaction:str)->str:
    """ 
    Write an excited message saying Yes

    Args:
        reaction (str): Write and excited reaction saying Yes
    """
    print(reaction)

@llm_function
def say_no(reaction:str)->str:
    """ 
    Write a sad reaction saying No

    Args:
        reaction (str): Write an excuse why we should not do this
    """
    print(reaction)


class NaiveAgent:

    def __init__(self) -> None:
        self.todo_list=[]

    @llm_prompt
    def consider_idea(self, user_input:str, functions=[say_yes, say_no]):
        """ 
        Is this a good idea?
        {user_input}
        """

    def run(self, user_input:str)->str:
        result = self.consider_idea(user_input=user_input)
        result.execute()


class ReasonableAgent(NaiveAgent):

    # with reasoning !!!
    @llm_prompt()
    def consider_idea(user_input:str, functions=[say_yes, say_no]):
        """ 
        Is this a good idea?
        {user_input}
        Before you make the decision what function to use, write a short reasoning.
        """

print("Naive agent:")    
NaiveAgent().run("I have a great idea,let's eat the whole tub of ice cream")
print("\n\nReasonable agent:")
ReasonableAgent().run("I have a great idea,let's eat the whole tub of ice cream")



  

# Naive agent:

# Result:

# Function call:
# {
#     "name": "say_yes",
#     "arguments": {
#         "reaction": "Yes, that sounds amazing! Let's do it!"
#     }
# }

# Yes, that sounds amazing! Let's do it!


# Reasonable agent:

# Result:
# Eating a whole tub of ice cream may sound tempting and enjoyable in the moment, but it is not a good idea for several reasons.
# Firstly, consuming such a large amount of ice cream in one sitting can lead to overeating and potential health issues, such as 
# weight gain and digestive problems. Secondly, excessive consumption of sugary foods can negatively impact blood sugar levels and 
# increase the risk of developing conditions like diabetes. Lastly, indulging in a whole tub of ice cream may provide temporary pleasure, 
# but it can also lead to feelings of guilt and regret afterwards. Therefore, it is best to exercise moderation and enjoy ice cream in 
# reasonable portions.


# Function call:
# {
#     "name": "say_no",
#     "arguments": {
#         "reaction": "No, it's not a good idea to eat the whole tub of ice cream. It can lead to overeating, health issues, and feelings of guilt and regret."
#     }
# }

# No, it's not a good idea to eat the whole tub of ice cream. It can lead to overeating, health issues, and feelings of guilt and regret.