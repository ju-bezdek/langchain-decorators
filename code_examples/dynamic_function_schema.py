from langchain_decorators.prompt_decorator import llm_prompt
from langchain_decorators.function_decorator import llm_function

from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.example_selector import (
    MaxMarginalRelevanceExampleSelector,
)
from langchain.vectorstores import FAISS
import requests
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders.text import TextLoader
from langchain.vectorstores import FAISS
from langchain.utilities import SerpAPIWrapper

# Install dependencies
# pip install google-search-results
# pip install faiss-cpu

#################################################### HELPERS ####################################################
def download_file(file_url:str, target_path:str=""):
    file_name = os.path.basename(file_url)
    if target_path:
        file_name = os.path.join(target_path, file_name)

    if not os.path.exists(file_name):
        data = requests.get(file_url).text
        with open(file_name, "w") as f:
            f.write(data)
    return file_name
    
        
def get_file_retriever(file_path):
    file_name = os.path.basename(file_path)
    if not os.path.exists(file_name+".faiss"):
        if file_path.startswith("http"):
            file_path = download_file(file_path)
        documents = TextLoader(file_path).load()
        # for doc in documents:
        #     doc.metadata["file_name"] = file_name


        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        faiss = FAISS.from_documents(texts, OpenAIEmbeddings())
        faiss.save_local(file_name+".faiss")
        return faiss.as_retriever()
    else:
        return FAISS.load_local(file_name+".faiss", OpenAIEmbeddings()).as_retriever()

################################################################################################################
serp_api_search = SerpAPIWrapper()



# lets fetch some example data
retriever = get_file_retriever("https://raw.githubusercontent.com/langchain-ai/langchain/7de6a1b78e9e86ebe7ee99c3194cfd97022ce789/docs/extras/modules/state_of_the_union.txt", "_tmp")


# this is the LLM function tha we expect to be called. Normally it wouldn't because it's definition is too vague
# but since we allowed dynamic schema, the Function definition will be augmented with a preview of closest data and enriched before feeding it to the LLM
@llm_function(dynamic_schema=True)
def search_in_files(input:str):
    """
    This function is useful to search in files
    {?Here are some examples of data available:
    {files_preview}?}

    Args:
        input (str): a hypothetical quote containing the answer to the question
    """
    return {f"file {doc.metadata.get('source')} [{i}]": doc.page_content for i, doc in enumerate(retriever.get_relevant_documents(input))}


# LLM would likely choose internet search function, because its more likely that you would find something about state policy on the internet

@llm_function
def internet_search(query_input:str):
    """
    Search for information on the internet

    Args:
        query_input (str): search query
    """
    return serp_api_search.run(query_input)

# this is just a simplified version of the agent function selection prompt
@llm_prompt
def chat_agent_prompt(user_question:str, closest_examples:str, functions):
    """
    ```<prompt:system>
    Help user. Use a function when appropriate
    ```
    ```<prompt:user>
    {user_question}
    ```
    """

# this is a prompt to generate final answer
@llm_prompt
def formulate_final_answer(question:str,scratchpad:list):
    """
    ```<prompt:system>
    Formulate final answer. Always refer the the source of information you used to answer the question.
    ```
    ```<prompt:user>
    {question}
    ```
    ```<prompt:placeholder>
    {scratchpad}
    ```
    """

# our question
user_question = "what will be the state policy regarding weapons"

closest_examples_docs = retriever.get_relevant_documents(user_question)
files_preview_txt = "\n".join([doc.page_content[:350] for doc in closest_examples_docs][:2])
next_step = chat_agent_prompt(user_question=user_question, files_preview=files_preview_txt, functions=[internet_search, search_in_files])
scratchpad = []
if next_step.is_function_call:
    # this will add AImessage with function call arguments to the scratchpad
    scratchpad.append(next_step.function_call_message)

    # this will execute the function and add the result to the scratchpad
    result_msg = next_step.function_output_to_message()
    scratchpad.append(result_msg)
    
    # we will use this to formulate the final answer
    answer = formulate_final_answer(question=user_question,scratchpad=scratchpad)
else:
    # this shouldn't be used in this example, but just in case
    answer = next_step.output

print(answer)
# Expected output:
# Based on the information provided in the file "state_of_the_union.txt", the state policy regarding weapons will include measures to crack down on gun trafficking and ghost guns, pass universal background checks, ban assault weapons and high-capacity magazines, and repeal the liability shield for gun manufacturers. These laws are aimed at reducing gun violence and do not infringe on the Second Amendment. The source of this information is the State of the Union address.