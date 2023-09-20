# this example shows how to convert llm_prompt decorated function into a chain 
# that can be used in combination with the rest of langchain ecosystem

from langchain_decorators import FollowupHandle, llm_prompt


# we don't need to declare followup_handle parameter... but it can be useful to know that it's available ...
# the name of the parameter must be precisely "followup_handle"
@llm_prompt
def ask(question:str, followup_handle:FollowupHandle=None)->str:
    """
    Answer the question like a pirate: {question} 
    (max 30 words)
    """

chain = ask.build_chain(question="Where was Schrödinger's cat locked in?")
print(chain()) # outputs: {'text': "Arr, Schrödinger's cat be locked in a mysterious box, matey!"}


# you can also override the inputs (in a native LangChain way)):
print(chain(inputs={"question":"What is the meaning of life?"}, return_only_outputs=True))
# outputs: {'text': "Arr, the meanin' o' life be a grand adventure on the high seas, seekin' treasure, makin' memories, and enjoyin' every moment, me hearties!"}
