# This example 


from typing import Any, Dict, List, Tuple
from langchain_decorators import llm_prompt, PromptTypeSettings

from langchain_decorators.prompt_template import BaseTemplateBuilder
from langchain import PromptTemplate

from langchain.llms import DeepInfra
llama70 = DeepInfra(model_id="meta-llama/Llama-2-70b-chat-hf")

class LLama2TemplateBuilder(BaseTemplateBuilder):
    
    def build_template(self, template_parts:List[Tuple[str,str]],kwargs:Dict[str,Any])->PromptTemplate:
        """ Function that builds a prompt template from a template string and the prompt block name (which is the the part of ```<prompt:$prompt_block_name> in the decorated function docstring)

        Args:
            template_parts (List[Tuple[str,str]]): list of prompt parts List[(prompt_block_name, template_string)]
            kwargs (Dict[str,Any]): all arguments passed to the decorated function

        Returns:
            PromptTemplate: ChatPromptTemplate or StringPromptTemplate
        """

        if len(template_parts)==1 and not template_parts[0][1]:
            return PromptTemplate.from_template(template_string)
        else:
            template_parts_final=[]
            for template_string,prompt_block_name in template_parts:
                template_string=template_string.strip()
                if prompt_block_name=="system":
                    template_parts_final.append(f"<<SYS>>\n{template_string}\n<</SYS>>")
                elif prompt_block_name=="inst":
                    template_parts_final.append(f"[INST]\n{template_string}\n[/INST]")
                elif prompt_block_name=="user":
                    template_parts_final.append(f"USER: {template_string}")
                else:
                    template_parts_final.append(template_string)

            return PromptTemplate.from_template( "<s> " + '\n'.join(template_parts_final))

llama70 = DeepInfra(model_id="meta-llama/Llama-2-70b-chat-hf") # define DEEPINFRA_API_TOKEN in your environment variables
LLAMA2_PROMPT_TYPE = PromptTypeSettings(prompt_template_builder=LLama2TemplateBuilder(), llm=llama70)

@llm_prompt(prompt_type=LLAMA2_PROMPT_TYPE)
def test_prompt(question:str):
    """
    ```<prompt:system>
    Act as a smart pirate
    ```

    ```<prompt:inst>
    Answer user question
    ```

    ```<prompt:user>
    {question}
    ```
    """

answer = test_prompt(question="What is the meaning of life?")
print(answer)
# Output:
# Result:
#
#
# PIRATE: Arrrr, me hearty! The meanin' o' life be different for every landlubber and scurvy dog. But if ye ask me, it be findin' yer treasure, matey! Whether it be gold doubloons or the love o' a fine piece o' booty, we all be searchin' fer somethin' that makes our lives worth livin'. So hoist the sails, grab yer trusty cutlass, and set sail fer yer own personal X marks the spot! Yarrr!