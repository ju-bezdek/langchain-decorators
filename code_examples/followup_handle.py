# This example shows how you can do a simple followup after executing a prompt without having to define a history
# In many cases this approach is more convenient and straightforward than using a history

from langchain_decorators import FollowupHandle, llm_prompt


# we don't need to declare followup_handle parameter... but it can be useful to know that it's available ...
# the name of the parameter must be precisely "followup_handle"
@llm_prompt
def ask(question:str, followup_handle:FollowupHandle=None)->str:
    """
    Answer the question like a pirate: {question} 
    (max 30 words)
    """

handle = FollowupHandle()
answer = ask(question="Where was Schrödinger's cat locked in?", followup_handle=handle)
print("Answer:",answer)
# Answer: Arr, Schrödinger's cat be locked in a mysterious box, matey!

answer = handle.followup("How?")
print("Answer:",answer)
# Answer: Arr, Schrödinger's cat be locked in a box, sealed tight with a devilish contraption that be triggerin' a deadly poison if a radioactive decay be detected, arr!

answer = handle.followup("So is it dead or alive?")
print("Answer:",answer)
# Answer: Arr, that be the mystery, me heartie! Schrödinger's cat be both dead and alive until ye open the box and lay yer eyes upon it.
    
# HINT: Use afollowup to execute the followup asynchroniously
# await handle.afollowup("So is it dead or alive?")