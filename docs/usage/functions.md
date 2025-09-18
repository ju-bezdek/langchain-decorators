# Functions Usage in LangChain Decorators

This section provides an overview of how to use functions within the LangChain Decorators framework. Functions are a core component of the LangChain ecosystem, allowing users to define and utilize custom logic in their applications.

## Defining Functions

To define a function in LangChain Decorators, you can use the `@llm_function` decorator. This decorator allows you to specify the function's behavior and how it interacts with the LangChain models.

### Example

```python
from langchain_decorators import llm_function

@llm_function
def example_function(param1: str, param2: int) -> str:
    """
    This function takes a string and an integer and returns a formatted string.

    Args:
        param1 (str): The first parameter.
        param2 (int): The second parameter.

    Returns:
        str: A formatted string combining the parameters.
    """
    return f"{param1} has {param2} items."
```

## Using Functions

Once you have defined a function, you can call it just like any other Python function. The LangChain framework will handle the integration with the underlying models.

### Calling the Function

```python
result = example_function("Apples", 5)
print(result)  # Output: Apples has 5 items.
```

## Function Parameters

Functions can accept various types of parameters, including:

- **Positional Parameters**: Standard parameters that must be provided in the correct order.
- **Keyword Parameters**: Parameters that can be specified by name, allowing for more flexible function calls.
- **Optional Parameters**: Parameters that have default values and can be omitted when calling the function.

### Example with Optional Parameters

```python
@llm_function
def greet(name: str, greeting: str = "Hello") -> str:
    """
    Greets a person with a specified greeting.

    Args:
        name (str): The name of the person.
        greeting (str, optional): The greeting to use. Defaults to "Hello".

    Returns:
        str: A greeting message.
    """
    return f"{greeting}, {name}!"
```

## Output Parsing

LangChain Decorators automatically detects the output type of functions and applies the appropriate output parser. This feature simplifies the process of handling different output formats.

### Example of Output Parsing

```python
@llm_function
def get_user_info(user_id: int) -> dict:
    """
    Retrieves user information based on the user ID.

    Args:
        user_id (int): The ID of the user.

    Returns:
        dict: A dictionary containing user information.
    """
    # Simulated user data retrieval
    return {"id": user_id, "name": "John Doe", "age": 30}
```

## Conclusion

Functions in LangChain Decorators provide a powerful way to extend the capabilities of your applications. By defining custom functions and leveraging the framework's features, you can create more dynamic and responsive applications.