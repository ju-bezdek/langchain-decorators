# API Documentation for LangChain Decorators

This document provides an overview of the API for the LangChain Decorators project. It includes details on the available functions, classes, and their usage.

## Overview

LangChain Decorators is a library that enhances the LangChain framework by providing decorators for easier prompt and function management. This API documentation outlines the key components and their functionalities.

## API Reference

### Decorators

#### `@llm_prompt`

This decorator is used to define a prompt for a function. It allows for the creation of multi-line prompts and integrates seamlessly with IDE features like type hinting and documentation popups.

**Parameters:**
- `function`: The function to be decorated.
- `verbose` (optional): If set to `True`, enables verbose logging for debugging.

**Example:**

```python
@llm_prompt
def example_function(param1: str) -> str:
    """
    This function demonstrates the use of the llm_prompt decorator.
    """
    return "Example output"
```

#### `@llm_function`

This decorator is used to define a function that can be called by the LLM. It parses the function's docstring to create a schema for the LLM.

**Parameters:**
- `function`: The function to be decorated.
- `dynamic_schema` (optional): If set to `True`, enables dynamic function schema generation.

**Example:**

```python
@llm_function
def another_example(param1: int) -> str:
    """
    This function provides another example for the llm_function decorator.
    """
    return "Another example output"
```

### Classes

#### `GlobalSettings`

This class manages global settings for the LangChain Decorators library, including default LLM configurations and logging settings.

**Methods:**
- `define_settings`: Defines global settings for the library.

**Example:**

```python
from langchain_decorators import GlobalSettings

GlobalSettings.define_settings(
    default_llm=ChatOpenAI(temperature=0.5)
)
```

### Usage

For detailed usage examples and further information, please refer to the [usage documentation](../usage/prompts.md).

## Conclusion

This API documentation serves as a reference for developers using the LangChain Decorators library. For more information, please consult the other sections of the documentation or the source code.