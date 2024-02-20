import json
from langchain_decorators.prompt_decorator import llm_prompt
from langchain.schema import AIMessage, HumanMessage


@llm_prompt()
def rag_get_sources_prompt(question: str, search_results: list[str]) -> dict:
    """

    ```<prompt:system>
    Find the information that is needed to answer the question

    Reply in JSON using the following format:
    {{
        "#id-[i]": quote information from the source that is relevant to the question,
        ...
    }}
    ```

    ```<prompt:function[search_results]>
    {search_results}
    ```

    ```<prompt:user>
    {question}
    ```
    """
    # we can format the inputs that we need to preprocess here... the returned dict is merged with the inputs
    return {
        "search_results": "\n".join(
            (
                f"# Search result {i} (#id-{i})\n{search_result}"
                for (i, search_result) in enumerate(search_results)
            )
        )
    }


@llm_prompt()
def rag_compose_answer_prompt(question: str, history_messages: list, sources: dict):
    """
    ```<prompt:system>
    Reply to user question, using the information from the sources.
    Add footnote links at the in the appropriate position in the answer to refer to the source of information used.
    ```
    ```<prompt:placeholder>
    {history_messages}
    ```
    ```<prompt:function[sources]>
    {sources}
    ```

    ```<prompt:user>
    {question}
    ```
    """
    return {"sources": json.dumps(sources)}


history = [
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="The capital of France is Paris[^1].\n Source: Wikipedia"),
]
# prepare demo data
question_msg = HumanMessage(content="how to eat healthy?")


def fake_retrieve(search_query):
    return [
        f"""# Healthy eating basics
            A healthy diet includes:

            1. Eating lots of vegetables and fruit

            This is one of the most important diet habits. Vegetables and fruit are packed with nutrients (antioxidants, vitamins, minerals and fibre) and help you maintain a healthy weight by keeping you full longer.
            Fill half your plate with vegetables and fruit at every meal and snack.
            2. Choosing whole grain foods

            Whole grain foods include whole grain bread and crackers, brown or wild rice, quinoa, oatmeal and hulled barley. They are prepared using the entire grain. Whole grain foods have fibre, protein and B vitamins to help you stay healthy and full longer.
            Choose whole grain options instead of processed or refined grains like white bread and pasta.
            Fill a quarter of your plate with whole grain foods.
            3. Eating protein foods

            Protein foods include legumes, nuts, seeds, tofu, fortified soy beverage, fish, shellfish, eggs, poultry, lean red meats including wild game, lower fat milk, lower fat yogurts, lower fat kefir and cheeses lower in fat and sodium.
            Protein helps build and maintain bones, muscles and skin.

            4. Limiting highly and ultra-processed foods

            Highly processed foods — often called ultra-processed — are foods that are changed from their original food source and have many added ingredients. During processing, often important nutrients such as vitamins, minerals and fiber are removed while salt and sugar are added.  Examples of processed food include: fast foods, hot dogs, chips, cookies, frozen pizzas, deli meats, white rice and white bread.

            5. Making water your drink of choice

            Water supports health and promotes hydration without adding calories to the diet. 
            Sugary drinks including energy drinks, fruit drinks, 100% fruit juice, soft drinks and flavored coffees have lots of sugar and little to no nutritional value. It is easy to drink empty calories without realizing, and this leads to weight gain.
            """,
        """ # 10 Simple Tips to Make Your Diet Healthier
            1. Eat from smaller plates
            2. Eat your greens first
            3. Slow down
            4. Drink water
            5. Eat more fruits and vegetables
            6. Eat more fish
            7. Cut down on saturated fat
            8. Don’t shop without a list
            9. Don’t ban foods
            10. Eat regularly
            """,
        # irrelevant example
        """37 Hilarious Food Memes For Anyone Who Just Wants To Eat Everything
        Do you love food? Check out these hilarious food memes that'll make you hungry for more. Memes or food, whatever comes first!
        """,
    ]


references = rag_get_sources_prompt(
    question=question_msg.content, search_results=fake_retrieve(question_msg.content)
)

answer = rag_compose_answer_prompt(
    question=question_msg, history_messages=history, sources=references
)

print(answer)
print("References:")
print("\n".join([f"{k}: {v[:20]}..." for k, v in references.items()]))

# Result:
#
# To eat healthy, it's important to include lots of vegetables and fruit in your diet, choose whole grain foods, eat protein foods, limit highly and ultra-processed foods, and make water your drink of choice[^0]. Additionally, you can make your diet healthier by eating from smaller plates, eating more fruits and vegetables, eating more fish, cutting down on saturated fat, and eating regularly[^1].
# References:
# #id-0: A healthy diet inclu...
# #id-1: 10 simple tips to ma...
