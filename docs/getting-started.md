# Getting Started with LangChain Decorators

Welcome to the LangChain Decorators documentation! This guide will help you get started with using LangChain Decorators effectively.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.7 or higher
- pip (Python package installer)

## Installation

To install LangChain Decorators, you can use pip. Run the following command in your terminal:

```bash
pip install langchain_decorators
```

## Quick Start

Once you have installed the package, you can start using LangChain Decorators in your Python projects. Here’s a simple example to get you started:

```python
from langchain_decorators import llm_prompt

@llm_prompt
def write_me_short_post(topic: str, platform: str = "twitter", audience: str = "developers") -> str:
    """
    Write me a short header for my post about {topic} for {platform} platform. 
    It should be for {audience} audience.
    (Max 15 words)
    """
    return

# Example usage
write_me_short_post(topic="starwars")
```

## Documentation Overview

This documentation is structured as follows:

- **Getting Started**: A guide for new users to set up and start using LangChain Decorators.
- **Usage**: Detailed documentation on how to use prompts, functions, streaming, and settings.
- **Reference**: API documentation for advanced users.
- **Changelog**: A log of updates and changes made to the project.

## Next Steps

After completing this guide, you can explore the following sections for more in-depth information:

- [Usage of Prompts](usage/prompts.md)
- [Usage of Functions](usage/functions.md)
- [Streaming Features](usage/streaming.md)
- [Settings Configuration](usage/settings.md)

If you have any questions or need further assistance, feel free to reach out to the community or check the [FAQ](index.md#faq) section. Happy coding!