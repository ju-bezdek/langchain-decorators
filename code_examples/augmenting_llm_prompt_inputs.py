import asyncio
from typing import List
from langchain_decorators import llm_prompt



# This demo is showcasing 2 features
# - preprocessing of the input arguments in the function implementation
# automatically parsing the output as dict leveraging the JsonOutputParser and JSON output format which is the default for latest OpenAI models and `dict` - output type
class TestCls:

    @llm_prompt()
    async def ask(self, question:str, choices:dict)->dict:
        """
        Answer the question {question} 
        with one of the following choices:
        {choices}
        
        reply in this format: {{"choice_id": your choice as one of {choices_ids} }}
        """
        # by implementing the @llm_prompt we can preprocess the arguments, which is useful to format them properly for the prompt template
        return {
                "choices": "\n".join((f"{choice_id}) {choice}" for choice_id, choice in choices.items())), # formatting choices as a bullet list
                "choices_ids": " | ".join(choices.keys()) # formatting choices as a comma separated list
            }


result_coro =TestCls().ask(
    question="Who was the first president of the USA?", 
    choices={
        "a":"George Washington", 
        "b":"Abraham Lincoln", 
        "c":"Donald Trump"
    })

print(asyncio.run(result_coro))


# Prompt:
# Answer the question Who was the first president of the USA? 
# with one of the following choices:
# a) George Washington
# b) Abraham Lincoln
# c) Donald Trump
#
# reply in this format: {"choice_id": your choice as one of a | b | c }


# Response:
# Result:
# {"choice_id": "a"}