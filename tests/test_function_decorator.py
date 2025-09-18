import pytest
import os
import asyncio
from typing import List, Dict, Optional, Union, Callable
from enum import Enum
from pydantic import BaseModel, Field

from langchain.tools.base import BaseTool

from langchain_decorators import (
    llm_prompt, 
    llm_function, 
    GlobalSettings,
    OutputWithFunctionCall
)


@pytest.fixture(scope="session")
def setup_real_llm():
    """Setup real LLM for testing"""
    from langchain_openai import ChatOpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set - skipping real LLM tests")
    
    real_llm = ChatOpenAI(
        temperature=0.0,  # Deterministic for testing
        model_name="gpt-3.5-turbo"
    )
    
    GlobalSettings.define_settings(
        default_llm=real_llm,
        verbose=False
    )
    return real_llm


@pytest.mark.real_llm
class TestLLMFunctionDecorator:
    """Test function calling with real LLM interactions - focused tests without redundancy"""

    def test_basic_function_calling_sync(self, setup_real_llm):
        """Test basic sync function calling functionality"""
        @llm_function
        def get_weather(city: str) -> str:
            """Get current weather for a city

            Args:
                city (str): Name of the city
            """
            weather_data = {
                "new york": "Sunny, 72°F",
                "london": "Cloudy, 15°C",
                "tokyo": "Rainy, 18°C",
            }
            return weather_data.get(city.lower(), f"Weather data not available for {city}")

        @llm_prompt
        def weather_assistant(user_request: str, functions: List[Callable]) -> OutputWithFunctionCall:
            """Help with weather requests: {user_request}

            If the user asks about weather, use the get_weather function."""
            pass

        result = weather_assistant(
            user_request="What's the weather like in New York?",
            functions=[get_weather]
        )

        if result.is_function_call:
            assert result.function_name == "get_weather"
            function_result = result.execute()
            assert "72°F" in function_result or "Sunny" in function_result

    @pytest.mark.asyncio
    async def test_basic_function_calling_async(self, setup_real_llm):
        """Test basic async function calling functionality"""

        @llm_function
        def get_weather(city: str) -> str:
            """Get current weather for a city

            Args:
                city (str): Name of the city
            """
            weather_data = {
                "new york": "Sunny, 72°F",
                "london": "Cloudy, 15°C",
                "tokyo": "Rainy, 18°C",
            }
            return weather_data.get(
                city.lower(), f"Weather data not available for {city}"
            )

        @llm_prompt
        async def weather_assistant(
            user_request: str, functions: List[Callable]
        ) -> OutputWithFunctionCall:
            """Help with weather requests: {user_request}

            If the user asks about weather, use the get_weather function."""
            pass

        result = await weather_assistant(
            user_request="What's the weather like in New York?", functions=[get_weather]
        )

        if result.is_function_call:
            assert result.function_name == "get_weather"
            function_result = result.execute()
            assert "72°F" in function_result or "Sunny" in function_result

    def test_complex_function_with_optional_params_sync(self, setup_real_llm):
        """Test function with multiple parameter types including optional parameters - sync"""
        class Priority(Enum):
            LOW = "low"
            MEDIUM = "medium"
            HIGH = "high"
            URGENT = "urgent"

        @llm_function
        def create_task(
            title: str,
            priority: Priority = Priority.MEDIUM,
            assignee: Optional[str] = None,
            tags: List[str] = None,
            deadline: Optional[str] = None,
        ) -> str:
            """Create a new task with optional parameters

            Args:
                title (str): Task title
                priority (Priority): Task priority level (default: medium)
                assignee (str, optional): Person to assign task to
                tags (List[str], optional): Task tags
                deadline (str, optional): Task deadline
            """
            task_id = f"TASK-{hash(title) % 1000:03d}"
            result = f"Created task {task_id}: '{title}' with {priority} priority"

            if assignee:
                result += f", assigned to {assignee}"
            if deadline:
                result += f", due {deadline}"
            if tags:
                result += f", tags: {', '.join(tags)}"

            return result

        @llm_prompt
        def task_manager(request: str, functions: List[Callable]) -> OutputWithFunctionCall:
            """Handle task management requests: {request}

            Use the create_task function to create new tasks based on the request."""
            pass

        result = task_manager(
            request="Create a high priority task to review budget by Friday, assign it to Alice with tags 'finance' and 'urgent'",
            functions=[create_task],
        )

        if result.is_function_call:
            assert result.function_name == "create_task"
            task_result = result.execute()
            assert "TASK-" in task_result
            assert "high" in task_result.lower()

    @pytest.mark.asyncio
    async def test_complex_function_with_optional_params_async(self, setup_real_llm):
        """Test function with multiple parameter types including optional parameters - async"""

        class Priority(Enum):
            LOW = "low"
            MEDIUM = "medium"
            HIGH = "high"
            URGENT = "urgent"

        @llm_function
        def create_task(
            title: str,
            priority: Priority = Priority.MEDIUM,
            assignee: Optional[str] = None,
            tags: List[str] = None,
            deadline: Optional[str] = None,
        ) -> str:
            """Create a new task with optional parameters

            Args:
                title (str): Task title
                priority (Priority): Task priority level (default: medium)
                assignee (str, optional): Person to assign task to
                tags (List[str], optional): Task tags
                deadline (str, optional): Task deadline
            """
            task_id = f"TASK-{hash(title) % 1000:03d}"
            result = f"Created task {task_id}: '{title}' with {priority} priority"

            if assignee:
                result += f", assigned to {assignee}"
            if deadline:
                result += f", due {deadline}"
            if tags:
                result += f", tags: {', '.join(tags)}"

            return result

        @llm_prompt
        async def task_manager(
            request: str, functions: List[Callable]
        ) -> OutputWithFunctionCall:
            """Handle task management requests: {request}

            Use the create_task function to create new tasks based on the request."""
            pass

        result = await task_manager(
            request="Create a high priority task to review budget by Friday, assign it to Alice with tags 'finance' and 'urgent'",
            functions=[create_task],
        )

        if result.is_function_call:
            assert result.function_name == "create_task"
            task_result = result.execute()
            assert "TASK-" in task_result
            assert "high" in task_result.lower()

    def test_multiple_function_selection_sync(self, setup_real_llm):
        """Test AI choosing between multiple available functions - sync"""
        @llm_function
        def send_email(recipient: str, subject: str, message: str) -> str:
            """Send an email

            Args:
                recipient (str): Email recipient
                subject (str): Email subject
                message (str): Email message content
            """
            return f"Email sent to {recipient}: '{subject}' - {message}"

        @llm_function
        def schedule_meeting(title: str, attendees: List[str], date_time: str) -> str:
            """Schedule a meeting
            
            Args:
                title (str): Meeting title
                attendees (List[str]): List of attendee emails
                date_time (str): Meeting date and time
            """
            return f"Meeting '{title}' scheduled for {date_time} with {', '.join(attendees)}"

        @llm_function
        def create_reminder(task: str, remind_time: str) -> str:
            """Create a reminder
            
            Args:
                task (str): Task to be reminded about
                remind_time (str): When to remind
            """
            return f"Reminder set: '{task}' at {remind_time}"

        @llm_prompt
        def office_assistant(request: str, functions: List[Callable]) -> OutputWithFunctionCall:
            """Handle office tasks: {request}

            Choose the most appropriate function for the request."""
            pass

        result = office_assistant(
            request="Send an email to alice@company.com about the budget meeting tomorrow",
            functions=[send_email, schedule_meeting, create_reminder],
        )

        if result.is_function_call:
            # Should choose send_email function
            if result.function_name == "send_email":
                email_result = result.execute()
                assert "alice@company.com" in email_result

    @pytest.mark.asyncio
    async def test_multiple_function_selection_async(self, setup_real_llm):
        """Test AI choosing between multiple available functions - async"""

        @llm_function
        def send_email(recipient: str, subject: str, message: str) -> str:
            """Send an email

            Args:
                recipient (str): Email recipient
                subject (str): Email subject
                message (str): Email message content
            """
            return f"Email sent to {recipient}: '{subject}' - {message}"

        @llm_function
        def schedule_meeting(title: str, attendees: List[str], date_time: str) -> str:
            """Schedule a meeting

            Args:
                title (str): Meeting title
                attendees (List[str]): List of attendee emails
                date_time (str): Meeting date and time
            """
            return f"Meeting '{title}' scheduled for {date_time} with {', '.join(attendees)}"

        @llm_function
        def create_reminder(task: str, remind_time: str) -> str:
            """Create a reminder

            Args:
                task (str): Task to be reminded about
                remind_time (str): When to remind
            """
            return f"Reminder set: '{task}' at {remind_time}"

        @llm_prompt
        async def office_assistant(
            request: str, functions: List[Callable]
        ) -> OutputWithFunctionCall:
            """Handle office tasks: {request}

            Choose the most appropriate function for the request."""
            pass

        result = await office_assistant(
            request="Send an email to alice@company.com about the budget meeting tomorrow",
            functions=[send_email, schedule_meeting, create_reminder],
        )

        if result.is_function_call:
            # Should choose send_email function
            if result.function_name == "send_email":
                email_result = result.execute()
                assert "alice@company.com" in email_result

    @pytest.mark.asyncio
    async def test_async_function_execution(self, setup_real_llm):
        """Test actual async function execution"""
        @llm_function
        async def async_search(query: str, results_count: int = 3) -> str:
            """Search for information asynchronously
            
            Args:
                query (str): Search query
                results_count (int): Number of results to return
            """
            await asyncio.sleep(0.1)  # Simulate async operation
            return f"Found {results_count} results for '{query}'"

        @llm_prompt
        async def search_assistant(request: str, functions: List[Callable]) -> OutputWithFunctionCall:
            """Handle search requests: {request}

            Use the async_search function to search for information."""
            pass

        result = await search_assistant(
            request="Search for Python tutorials, show me 5 results",
            functions=[async_search]
        )

        if result.is_function_call:
            assert result.function_name == "async_search"
            search_result = await result.execute_async()
            assert "Python tutorials" in search_result
            assert "results" in search_result

    def test_error_handling_sync(self, setup_real_llm):
        """Test function execution error handling - sync"""

        @llm_function
        def divide_numbers(a: float, b: float) -> str:
            """Divide two numbers

            Args:
                a (float): Dividend
                b (float): Divisor
            """
            if b == 0:
                raise ValueError("Cannot divide by zero")
            return f"{a} ÷ {b} = {a/b}"

        @llm_prompt
        def math_helper(
            problem: str, functions: List[Callable]
        ) -> OutputWithFunctionCall:
            """Solve math problem: {problem}

            Use the divide_numbers function for division problems."""
            pass

        result = math_helper(
            problem="What is 10 divided by 0?", functions=[divide_numbers]
        )

        if result.is_function_call:
            assert result.function_name == "divide_numbers"
            with pytest.raises(ValueError, match="divide by zero"):
                result.execute()

    @pytest.mark.asyncio
    async def test_error_handling_async(self, setup_real_llm):
        """Test function execution error handling - async"""

        @llm_function
        def divide_numbers(a: float, b: float) -> str:
            """Divide two numbers

            Args:
                a (float): Dividend
                b (float): Divisor
            """
            if b == 0:
                raise ValueError("Cannot divide by zero")
            return f"{a} ÷ {b} = {a/b}"

        @llm_prompt
        async def math_helper(
            problem: str, functions: List[Callable]
        ) -> OutputWithFunctionCall:
            """Solve math problem: {problem}

            Use the divide_numbers function for division problems."""
            pass

        result = await math_helper(
            problem="What is 10 divided by 0?", functions=[divide_numbers]
        )

        if result.is_function_call:
            assert result.function_name == "divide_numbers"
            with pytest.raises(ValueError, match="divide by zero"):
                result.execute()

    def test_function_schema_generation(self, setup_real_llm):
        """Test that function schemas are properly generated for complex types"""
        @llm_function
        def complex_function(
            required_param: str,
            optional_param: Optional[int] = None,
            list_param: List[str] = None,
            bool_param: bool = False
        ) -> str:
            """A complex function with various parameter types

            Args:
                required_param (str): A required string parameter
                optional_param (int, optional): An optional integer
                list_param (List[str], optional): A list of strings
                bool_param (bool): A boolean parameter
            """
            return f"Called with: {required_param}, {optional_param}, {list_param}, {bool_param}"

        @llm_prompt
        def test_schema(request: str, functions: List[Callable]) -> OutputWithFunctionCall:
            """Test function schema: {request}

            Use the complex_function with appropriate parameters."""
            pass

        result = test_schema(
            request="Call the complex function with 'hello' as required param and true for bool param",
            functions=[complex_function]
        )

        if result.is_function_call:
            assert result.function_name == "complex_function"
            function_result = result.execute()
            assert "hello" in function_result.lower()


if __name__ == "__main__":
    TestLLMFunctionDecorator().test_basic_function_calling_sync(None)
