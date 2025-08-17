# Prompts Usage Documentation

This section provides an overview of how to effectively use prompts within the LangChain Decorators framework. Prompts are essential for guiding the behavior of language models and customizing their outputs.

## Defining Prompts

Prompts can be defined using the `@llm_prompt` decorator. This allows you to create functions that encapsulate the prompt logic, making it easier to manage and reuse.

### Example

```python
from langchain_decorators import llm_prompt

@llm_prompt
def generate_summary(text: str) -> str:
    """
    Generate a concise summary of the provided text.
    """
    return
```

## Prompt Structure

When defining prompts, you can include various sections in the docstring to enhance clarity and functionality:

- **Prompt Definition**: The main prompt that the model will use.
- **Parameters**: Descriptions of the function parameters.
- **Return Type**: Information about what the function returns.

### Using Code Blocks

You can specify the prompt using a code block within the docstring:

```python
@llm_prompt
def create_post(topic: str) -> str:
    """
    ```<prompt>
    Write a blog post about {topic}.
    ```
    """
    return
```

## Optional Sections

Prompts can include optional sections that are only rendered if certain conditions are met. This allows for more dynamic prompt generation.

### Example

```python
@llm_prompt
def flexible_prompt(user_input: str, additional_info: str = None) -> str:
    """
    This is a flexible prompt that can include additional information if provided.

    {? This section is optional and will only be included if additional_info is not None ?}
    Here is some extra information: {additional_info}
    """
    return
```

## Conclusion

Using prompts effectively can significantly enhance the interaction with language models. By structuring prompts clearly and utilizing optional sections, you can create more dynamic and responsive applications.