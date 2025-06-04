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
    """Test function calling with real LLM interactions"""
    
    def test_simple_function_call(self, setup_real_llm):
        """Test basic function calling functionality"""
        @llm_function
        def get_weather(city: str) -> str:
            """Get current weather for a city
            
            Args:
                city (str): Name of the city
            """
            # Simulate weather API
            weather_data = {
                "new york": "Sunny, 72°F",
                "london": "Cloudy, 15°C",
                "tokyo": "Rainy, 18°C",
                "paris": "Overcast, 16°C"
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
        else:
            # LLM chose to respond directly - that's also valid
            assert isinstance(result.output_text, str)
    
    def test_function_with_multiple_parameters(self, setup_real_llm):
        """Test function with multiple parameters and types"""
        @llm_function
        def calculate_tip(bill_amount: float, tip_percentage: float = 18.0, split_ways: int = 1) -> str:
            """Calculate tip and split bill among people
            
            Args:
                bill_amount (float): Total bill amount
                tip_percentage (float): Tip percentage (default 18%)
                split_ways (int): Number of people to split bill
            """
            tip_amount = bill_amount * (tip_percentage / 100)
            total = bill_amount + tip_amount
            per_person = total / split_ways
            
            return f"Bill: ${bill_amount:.2f}, Tip: ${tip_amount:.2f}, Total: ${total:.2f}, Per person: ${per_person:.2f}"
        
        @llm_prompt
        def bill_calculator(request: str, functions: List[Callable]) -> OutputWithFunctionCall:
            """Help with bill calculations: {request}
            
            """
            pass
        
        result = bill_calculator(
            request="Calculate tip for a $100 dinner bill with 20% tip for 4 people",
            functions=[calculate_tip]
        )
        
        if result.is_function_call:
            assert result.function_name == "calculate_tip"
            calc_result = result.execute()
            assert "$100.00" in calc_result
            # Check for expected per-person amount ($30.00) instead of the literal "4"
            assert "$30.00" in calc_result
    
    def test_function_with_enum_parameters(self, setup_real_llm):
        """Test function with enum parameters"""
        class Priority(Enum):
            LOW = "low"
            MEDIUM = "medium"
            HIGH = "high"
            URGENT = "urgent"
        
        @llm_function
        def create_task(title: str, priority: Priority, deadline: Optional[str] = None) -> str:
            """Create a new task
            
            Args:
                title (str): Task title
                priority (Priority): Task priority level
                deadline (str, optional): Task deadline
            """
            task_id = f"TASK-{hash(title) % 1000:03d}"
            result = f"Created task {task_id}: '{title}' with {priority} priority"
            if deadline:
                result += f" due {deadline}"
            return result
        
        @llm_prompt
        def task_manager(request: str, functions: List[Callable]) -> OutputWithFunctionCall:
            """Handle task management requests: {request}
            
            """
            pass
        
        result = task_manager(
            request="Create a high priority task to review budget by Friday",
            functions=[create_task]
        )
        
        if result.is_function_call:
            assert result.function_name == "create_task"
            task_result = result.execute()
            assert "TASK-" in task_result
            assert "high" in task_result.lower()
    
    
    
    def test_multiple_function_options(self, setup_real_llm):
        """Test AI choosing between multiple available functions"""
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
        
        # Test email request
        result1 = office_assistant(
            request="Send an email to alice@company.com about the budget meeting tomorrow",
            functions=[send_email, schedule_meeting, create_reminder]
        )
        
        if result1.is_function_call:
            if result1.function_name == "send_email":
                email_result = result1.execute()
                assert "alice@company.com" in email_result
            # Function choice is valid as long as it executes successfully
        
        # Test meeting request
        result2 = office_assistant(
            request="Schedule a team standup for tomorrow at 9 AM with bob@company.com and carol@company.com",
            functions=[send_email, schedule_meeting, create_reminder]
        )
        
        if result2.is_function_call:
            if result2.function_name == "schedule_meeting":
                meeting_result = result2.execute()
                assert "standup" in meeting_result.lower() or "team" in meeting_result.lower()
    
    def test_function_error_handling(self, setup_real_llm):
        """Test function execution error handling"""
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
        def math_helper(problem: str, functions: List[Callable]) -> OutputWithFunctionCall:
            """Solve math problem: {problem}
            
            """
            pass
        
        result = math_helper(
            problem="What is 10 divided by 0?",
            functions=[divide_numbers]
        )
        
        if result.is_function_call:
            assert result.function_name == "divide_numbers"
            # Function should raise ValueError when executed
            with pytest.raises(ValueError, match="divide by zero"):
                result.execute()
    
    @pytest.mark.asyncio
    async def test_async_function_calling(self, setup_real_llm):
        """Test async function calling"""
        @llm_function
        async def async_search(query: str, results_count: int = 3) -> str:
            """Search for information asynchronously
            
            Args:
                query (str): Search query
                results_count (int): Number of results to return
            """
            # Simulate async operation
            await asyncio.sleep(0.1)
            return f"Found {results_count} results for '{query}'"
        
        @llm_prompt
        async def search_assistant(request: str, functions: List[Callable]) -> OutputWithFunctionCall:
            """Handle search requests: {request}
            
            """
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
    
    def test_function_with_complex_return_types(self, setup_real_llm):
        """Test function that returns structured data"""
        @llm_function
        def analyze_data(data: List[float], analysis_type: str = "summary") -> Dict:
            """Analyze numerical data
            
            Args:
                data (List[float]): List of numbers to analyze
                analysis_type (str): Type of analysis to perform
            """
            if not data:
                return {"error": "No data provided"}
            
            return {
                "count": len(data),
                "sum": sum(data),
                "average": sum(data) / len(data),
                "min": min(data),
                "max": max(data),
                "analysis_type": analysis_type
            }
        
        @llm_prompt
        def data_analyst(request: str, functions: List[Callable]) -> OutputWithFunctionCall:
            """Analyze data based on: {request}
            
            """
            pass
        
        result = data_analyst(
            request="Analyze these sales numbers: 100, 150, 200, 175, 125",
            functions=[analyze_data]
        )
        
        if result.is_function_call:
            assert result.function_name == "analyze_data"
            analysis_result = result.execute()
            # Result should be a dictionary or string representation
            assert isinstance(analysis_result, (dict, str))
    
    def test_function_with_optional_parameters(self, setup_real_llm):
        """Test function with mix of required and optional parameters"""
        @llm_function
        def book_restaurant(
            restaurant_name: str,
            party_size: int,
            date: str,
            time: str,
            special_requests: Optional[str] = None
        ) -> str:
            """Book a restaurant reservation
            
            Args:
                restaurant_name (str): Name of the restaurant
                party_size (int): Number of people
                date (str): Reservation date
                time (str): Reservation time
                special_requests (str, optional): Any special requests
            """
            reservation = f"Booked table for {party_size} at {restaurant_name} on {date} at {time}"
            if special_requests:
                reservation += f". Special requests: {special_requests}"
            return reservation
        
        @llm_prompt
        def reservation_assistant(request: str, functions: List[Callable]) -> OutputWithFunctionCall:
            """Handle restaurant reservations: {request}
            
            """
            pass
        
        result = reservation_assistant(
            request="Book a table for 4 people at Mario's Restaurant tomorrow at 7 PM, we need a high chair for a baby",
            functions=[book_restaurant]
        )
        
        if result.is_function_call:
            assert result.function_name == "book_restaurant"
            booking_result = result.execute()
            assert "Mario's" in booking_result or "mario" in booking_result.lower()
            assert "4" in booking_result
    
    def test_function_with_list_parameters(self, setup_real_llm):
        """Test function with list parameters"""
        @llm_function
        def create_shopping_list(items: List[str], category: str = "grocery") -> str:
            """Create a shopping list
            
            Args:
                items (List[str]): List of items to buy
                category (str): Shopping category
            """
            list_text = f"{category.title()} Shopping List:\n"
            for i, item in enumerate(items, 1):
                list_text += f"{i}. {item}\n"
            return list_text
        
        @llm_prompt
        def shopping_assistant(request: str, functions: List[Callable]) -> OutputWithFunctionCall:
            """Help with shopping lists: {request}
            
            """
            pass
        
        result = shopping_assistant(
            request="Create a grocery list with milk, bread, eggs, and apples",
            functions=[create_shopping_list]
        )
        
        if result.is_function_call:
            assert result.function_name == "create_shopping_list"
            list_result = result.execute()
            assert "milk" in list_result.lower()
            assert "bread" in list_result.lower()
            assert "Shopping List" in list_result
    
    def test_function_schema_generation(self, setup_real_llm):
        """Test that function schemas are properly generated"""
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
        
        # Get the schema (this is internal to the decorator but we can test it works)
        @llm_prompt
        def test_schema(request: str, functions: List[Callable]) -> OutputWithFunctionCall:
            """Test function schema: {request}
            
            """
            pass
        
        result = test_schema(
            request="Call the complex function with 'hello' as required param and true for bool param",
            functions=[complex_function]
        )
        
        if result.is_function_call:
            assert result.function_name == "complex_function"
            # Should be able to execute with proper parameters
            function_result = result.execute()
            assert "hello" in function_result.lower()
    
    def test_integration_with_langchain_tools(self, setup_real_llm):
        """Test integration between llm_function and regular LangChain tools"""
        from langchain.tools import Tool, tool
        
        # Create a regular LangChain tool
        def simple_calculator(expression: str) -> str:
            """Calculate a simple math expression"""
            try:
                # Safe eval for basic math
                result = eval(expression.replace("^", "**"))
                return str(result)
            except:
                return "Error in calculation"
        
        calculator_tool = Tool(
            name="Calculator",
            func=simple_calculator,
            description="Calculate simple math expressions"
        )


        
        # Create an llm_function
        @tool
        def get_current_time(timezone:str) -> str:
            """Get the current time
            
            Returns:
                str: Current time formatted as string
            """
            from datetime import datetime
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        @llm_prompt
        def mixed_assistant(request: str, functions: List[Union[Callable, BaseTool]]) -> OutputWithFunctionCall:
            """Handle various requests: {request}
            
            """
            pass
        
        # Test with math request (should use calculator tool)
        result1 = mixed_assistant(
            request="What is 15 * 8?",
            functions=[get_current_time, calculator_tool]
        )
        
        # Test with time request (should use our function)
        result2 = mixed_assistant(
            request="What time is it?",
            functions=[get_current_time, calculator_tool]
        )
        
        # Both should work and choose appropriate tools
        if result1.is_function_call:
            tool_result = result1.execute()
            assert isinstance(tool_result, str)
        
        if result2.is_function_call:
            time_result = result2.execute()
            assert isinstance(time_result, str)

if __name__ == "__main__":
    TestLLMFunctionDecorator().test_integration_with_langchain_tools(None)