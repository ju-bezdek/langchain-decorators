
import pytest
import os
import asyncio
from langchain.schema import AIMessage, HumanMessage, FunctionMessage
from langchain.chat_models.base import BaseChatModel
from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional, Callable

from langchain_decorators import (
    llm_prompt, 
    llm_function,
    GlobalSettings, 
    OutputWithFunctionCall,
    FollowupHandle,
    ToolsProvider
)







class TestChains:
    """Test chain functionality and integration with real LLM"""
    
  
    def test_chain_with_function_support(self):
        """Test LLMDecoratorChainWithFunctionSupport with real function calls"""

        
        @llm_function
        def get_weather(city: str) -> str:
            """Get current weather for a city
            
            Args:
                city (str): Name of the city
            """
            # Simulate weather API response
            weather_data = {
                "new york": "Sunny, 72°F",
                "london": "Rainy, 12°C", 
                "tokyo": "Cloudy, 18°C"
            }
            return weather_data.get(city.lower(), f"Weather not available for {city}")
        
        @llm_prompt
        def weather_assistant(user_input: str, functions: List[Callable]) -> OutputWithFunctionCall:
            """Help the user with weather requests: {user_input}
            
            """
            pass
        
        result = weather_assistant(
            user_input="What's the weather like in New York?",
            functions=[get_weather]
        )
        
        assert isinstance(result, OutputWithFunctionCall)
        if result.is_function_call:
            assert result.function_name == "get_weather"
            weather_result = result.execute()
            assert "72°F" in weather_result or "Sunny" in weather_result
    
    def test_functions_provider(self):
        """Test ToolsProvider class with real functions"""
        @llm_function
        def calculate(expression: str) -> str:
            """Calculate a mathematical expression
            
            Args:
                expression (str): Math expression like "2 + 3"
            """
            try:
                # Safe eval for basic math
                result = eval(expression)
                return str(result)
            except:
                return "Invalid expression"
        
        @llm_function
        def format_text(text: str, style: str = "uppercase") -> str:
            """Format text in different styles
            
            Args:
                text (str): Text to format
                style (str): Style (uppercase, lowercase, title)
            """
            if style == "uppercase":
                return text.upper()
            elif style == "lowercase":
                return text.lower()
            elif style == "title":
                return text.title()
            return text
        
        provider = ToolsProvider([calculate, format_text])
        
        # Test function access
        assert calculate in provider
        assert format_text in provider
        assert len(provider.tools) == 2
        
        # Test function retrieval
        retrieved_func = provider.get_function("calculate")
        assert retrieved_func == calculate
        
        # Test schema generation
        schemas = provider.get_function_schemas({})
        assert len(schemas) == 2
        assert any(schema["name"] == "calculate" for schema in schemas)
        assert any(schema["name"] == "format_text" for schema in schemas)
    
    def test_followup_handle(self):
        """Test FollowupHandle functionality with real LLM"""
        @llm_prompt
        def tech_expert(question: str, followup_handle: FollowupHandle = None) -> str:
            """You are a tech expert. Answer this question concisely: {question}"""
            pass
        
        handle = FollowupHandle()
        result = tech_expert(question="What is Python?", followup_handle=handle)
        
        # Test that handle is bound to chain
        assert handle.chain is not None
        assert isinstance(result, str)
        assert "python" in result.lower()
        
        # Test followup functionality
        followup_result = handle.followup("Can you give me a specific example?")
        
        assert isinstance(followup_result, str)
        assert len(followup_result) > 0
    
    @pytest.mark.asyncio
    async def test_async_followup_handle(self):
        """Test async FollowupHandle functionality"""
        @llm_prompt
        async def async_assistant(question: str, followup_handle: FollowupHandle = None) -> str:
            """Answer this programming question: {question}"""
            pass
        
        handle = FollowupHandle()
        result = await async_assistant(question="What is async/await?", followup_handle=handle)
        
        assert isinstance(result, str)
        assert "async" in result.lower() or "await" in result.lower()
        
        # Test async followup
        followup_result = await handle.afollowup("Give me a code example")
        
        assert isinstance(followup_result, str)
        assert len(followup_result) > 0
    
    def test_followup_with_functions(self):
        """Test followup with function calling"""
        @llm_function
        def search_docs(query: str, language:Literal["python","javascript"]) -> str:
            """Search documentation for information
            
            Args:
                query (str): Search query
                language (str): Programming language to search in (python, javascript)
            """
            # Simulate documentation search
            docs = {
                "python": "Python is a high-level programming language known for its simplicity",
                "javascript": "JavaScript is a programming language for web development",
            }
          
            return f"No documentation found for '{language}'"
        
        @llm_prompt
        def documentation_helper(question: str, functions: List[Callable], followup_handle: FollowupHandle = None) -> OutputWithFunctionCall:
            """{question}
            Provide answer using documentation
            
            """
            pass
        
        handle = FollowupHandle()
        
        result:OutputWithFunctionCall = documentation_helper(
            question="What are the core functions of Python?",
            functions=[search_docs],
            followup_handle=handle
        )
        
        if result.is_function_call:
            doc_result = result.execute()
            handle.message_history.append(result.function_output_to_message())
            assert "python" in doc_result
        
        # Test followup with functions
        followup_result:OutputWithFunctionCall = handle.followup("Now tell me about JavaScript", with_tools=True)
        
        assert isinstance(followup_result, (OutputWithFunctionCall))
        assert followup_result.is_function_call
        assert followup_result.function_name == "search_docs"
        assert isinstance(followup_result.function_arguments, dict)
        assert followup_result.function_arguments.get("language").lower() == "javascript"
    
    def test_output_with_function_call_execution(self):
        """Test OutputWithFunctionCall functionality"""
        @llm_function
        def send_notification(message: str, urgency: str = "normal") -> str:
            """Send a notification message
            
            Args:
                message (str): Message to send
                urgency (str): Urgency level (low, normal, high)
            """
            return f"Notification sent: '{message}' (urgency: {urgency})"
        
        @llm_prompt
        def notification_manager(request: str, functions: List[Callable]) -> OutputWithFunctionCall:
            """Handle notification requests: {request}
            """
            pass
        
        result = notification_manager(
            request="Send a high priority notification that the server is down",
            functions=[send_notification]
        )
        
        if result.is_function_call:
            # Test properties
            assert result.function_name == "send_notification"
            assert result.support_sync
            
            # Test execution
            notification_result = result.execute()
            assert "Notification sent" in notification_result
            assert result.result == notification_result
    
    @pytest.mark.asyncio
    async def test_output_with_function_call_async(self):
        """Test OutputWithFunctionCall with async function"""
        @llm_function
        async def process_data_async(data: str, format_type: str = "json") -> str:
            """Process data asynchronously
            
            Args:
                data (str): Data to process
                format_type (str): Output format (json, xml, csv)
            """
            await asyncio.sleep(0.1)  # Simulate async processing
            return f"Processed '{data}' as {format_type} format"
        
        @llm_prompt
        async def async_data_processor(request: str, functions: List[Callable]) -> OutputWithFunctionCall:
            """Process data requests: {request}
            
            """
            pass
        
        result = await async_data_processor(
            request="Process user data as JSON format",
            functions=[process_data_async]
        )
        
        if result.is_function_call:
            # Test async execution
            processed_result = await result.execute_async()
            assert "Processed" in processed_result
    
   
    
    def test_conversation_chain_with_context(self):
        """Test conversational chain that maintains context"""
        conversation_history = []
        
        @llm_prompt
        def conversational_assistant(user_message: str, context_history: List[str] = None) -> str:
            """
            ```<prompt:system>
            You are a helpful assistant. Use the conversation history to provide contextual responses.
            
            {? Previous conversation:
            {context_history}
            ?}
            ```
            
            ```<prompt:user>
            {user_message}
            ```
            """
            pass
        
        # First message
        response1 = conversational_assistant(
            user_message="I'm planning a trip to Japan",
            context_history=conversation_history
        )
        conversation_history.append(f"User: I'm planning a trip to Japan\nAssistant: {response1}")
        
        assert "japan" in response1.lower()
        
        # Follow-up message with context
        response2 = conversational_assistant(
            user_message="What's the best time to visit?",
            context_history=conversation_history
        )
        
        assert isinstance(response2, str)
        # Should provide Japan-specific advice due to context
        assert any(month in response2.lower() for month in ["spring", "summer", "fall", "winter", "march", "april", "may"]) or "japan" in response2.lower()
    
    def test_multi_step_workflow(self):
        """Test multi-step workflow with function chaining"""
        @llm_function
        def analyze_text(text: str) -> str:
            """Analyze text for sentiment and topics
            
            Args:
                text (str): Text to analyze
            """
            # Simple sentiment analysis simulation
            positive_words = ["good", "great", "excellent", "amazing", "wonderful"]
            negative_words = ["bad", "terrible", "awful", "horrible", "disappointing"]
            
            text_lower = text.lower()
            sentiment = "neutral"
            
            if any(word in text_lower for word in positive_words):
                sentiment = "positive"
            elif any(word in text_lower for word in negative_words):
                sentiment = "negative"
            
            return f"Sentiment: {sentiment}, Text length: {len(text)} characters"
        
        @llm_function
        def generate_summary(analysis: str, original_text: str) -> str:
            """Generate a summary based on analysis
            
            Args:
                analysis (str): Text analysis results
                original_text (str): Original text
            """
            return f"Summary: Text analysis shows {analysis}. Original text: '{original_text[:50]}...'"
        
        @llm_prompt
        def text_workflow_manager(text_input: str, functions: List[Callable]) -> OutputWithFunctionCall:
            """Manage text processing workflow for: {text_input}
            
            
            
            First analyze the text, then create a summary.
            """
            pass
        
        # Start workflow
        result = text_workflow_manager(
            text_input="This is an amazing product that exceeded my expectations!",
            functions=[analyze_text, generate_summary]
        )
        
        if result.is_function_call:
            analysis_result = result.execute()
            assert "positive" in analysis_result.lower()
    
    def test_error_handling_in_functions(self):
        """Test error handling in function execution"""
        @llm_function
        def risky_operation(value: str) -> str:
            """Perform a risky operation that might fail
            
            Args:
                value (str): Input value
            """
            if value.lower() == "error":
                raise ValueError("Intentional error for testing")
            return f"Successfully processed: {value}"
        
        @llm_prompt
        def error_handling_assistant(request: str, functions: List[Callable]) -> OutputWithFunctionCall:
            """Handle requests carefully: {request}
            
            
            """
            pass
        
        # Test successful operation
        result_success = error_handling_assistant(
            request="Process the value 'success'",
            functions=[risky_operation]
        )
        
        if result_success.is_function_call:
            success_result = result_success.execute()
            assert "Successfully processed" in success_result
        
        # Test error case
        result_error = error_handling_assistant(
            request="Process the value 'error'",
            functions=[risky_operation]
        )
        
        if result_error.is_function_call:
            with pytest.raises(ValueError, match="Intentional error"):
                result_error.execute()
    
    def test_dynamic_function_schemas(self):
        """Test dynamic function schema generation with context"""
        @llm_function(dynamic_schema=True)
        def search_content(query: str) -> str:
            """Search for {content_type} about {topic}
            
            Args:
                query (str): Search query for {search_domain}
            """
            return f"Found content for: {query}"
        
        # Test schema generation with dynamic context
        context = {
            "content_type": "tutorials", 
            "topic": "machine learning",
            "search_domain": "educational resources"
        }
        
        schema = search_content.get_function_schema(search_content, context)
        
        assert "tutorials" in schema["description"]
        assert "machine learning" in schema["description"]
        assert "educational resources" in schema["parameters"]["properties"]["query"]["description"]
    
    def test_complex_prompt_with_multiple_functions(self):
        """Test complex prompt that can choose between multiple functions"""
        @llm_function
        def create_calendar_event(title: str, date: str, time: str) -> str:
            """Create a calendar event
            
            Args:
                title (str): Event title
                date (str): Event date
                time (str): Event time
            """
            return f"Calendar event created: '{title}' on {date} at {time}"
        
        @llm_function
        def send_email_reminder(recipient: str, subject: str, message: str) -> str:
            """Send an email reminder
            
            Args:
                recipient (str): Email recipient
                subject (str): Email subject
                message (str): Email message
            """
            return f"Email sent to {recipient}: {subject} - {message}"
        
        @llm_function
        def set_phone_alarm(time: str, label: str) -> str:
            """Set a phone alarm
            
            Args:
                time (str): Alarm time
                label (str): Alarm label
            """
            return f"Alarm set for {time} with label: {label}"
        
        @llm_prompt
        def personal_assistant(request: str, functions: List[Callable]) -> OutputWithFunctionCall:
            """
            ```<prompt:system>
            You are a personal assistant. Choose the most appropriate function for each request.
            ```
            
            ```<prompt:user>
            {request}
            ```
            
            
            """
            pass
        
        # Test calendar event request
        calendar_result = personal_assistant(
            request="Schedule a meeting with the team for tomorrow at 2 PM",
            functions=[create_calendar_event, send_email_reminder, set_phone_alarm]
        )
        
        if calendar_result.is_function_call:
            if calendar_result.function_name == "create_calendar_event":
                event_result = calendar_result.execute()
                assert "Calendar event created" in event_result
        
        # Test alarm request
        alarm_result = personal_assistant(
            request="Wake me up at 7 AM tomorrow for my workout",
            functions=[create_calendar_event, send_email_reminder, set_phone_alarm]
        )
        
        if alarm_result.is_function_call:
            if alarm_result.function_name == "set_phone_alarm":
                alarm_response = alarm_result.execute()
                assert "Alarm set" in alarm_response


if __name__ == "__main__":
    TestChains().test_multi_step_workflow()