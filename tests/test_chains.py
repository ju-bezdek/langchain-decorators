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

# Test with and without fallbacks
WITH_FALLBACKS = pytest.param(True, id="with_fallbacks"), pytest.param(
    False, id="without_fallbacks"
)


def configure_llm(with_fallbacks: bool) -> BaseChatModel:
    """Setup real LLM for testing"""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set - skipping real LLM tests")

    real_llm = ChatOpenAI(
        temperature=0.0, model_name="gpt-3.5-turbo"  # Deterministic for testing
    )

    # Use the parameterized value to determine if fallbacks should be used

    if with_fallbacks:
        real_llm = real_llm.with_fallbacks([ChatOpenAI(model_name="gpt-4o-mini")])

    GlobalSettings.define_settings(default_llm=real_llm, verbose=False)
    return real_llm


@pytest.fixture(
    scope="session", params=[True, False], ids=["with_fallbacks", "without_fallbacks"]
)
def setup_real_llm(request):
    configure_llm(request.param)


@pytest.mark.real_llm
class TestChains:
    """Test chain functionality and integration - focused on unique chain features"""

    def test_tools_provider_functionality(self, setup_real_llm):
        """Test ToolsProvider class with real functions"""
        @llm_function
        def calculate(expression: str) -> str:
            """Calculate a mathematical expression
            
            Args:
                expression (str): Math expression like "2 + 3"
            """
            try:
                # Safe eval for basic math
                result = eval(expression.replace("^", "**"))
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

    def test_followup_handle_sync(self, setup_real_llm):
        """Test FollowupHandle functionality - sync"""
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
    async def test_followup_handle_async(self, setup_real_llm):
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

    def test_followup_with_functions_sync(self, setup_real_llm):
        """Test followup with function calling - sync"""

        @llm_function
        def search_docs(query: str, language: Literal["python", "javascript"]) -> str:
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
            return docs.get(language, f"No documentation found for '{language}'")

        @llm_prompt
        def documentation_helper(question: str, functions: List[Callable], followup_handle: FollowupHandle = None) -> OutputWithFunctionCall:
            """{question}
            Provide answer using documentation search.
            """
            pass

        handle = FollowupHandle()

        result: OutputWithFunctionCall = documentation_helper(
            question="What are the core functions of Python?",
            functions=[search_docs],
            followup_handle=handle,
        )

        if result.is_function_call:
            doc_result = result.execute()
            handle.message_history.append(result.function_output_to_message())
            assert "python" in doc_result.lower()

        # Test followup with functions
        followup_result: OutputWithFunctionCall = handle.followup(
            "Now tell me about JavaScript", with_tools=True
        )

        assert isinstance(followup_result, OutputWithFunctionCall)
        if followup_result.is_function_call:
            assert followup_result.function_name == "search_docs"
            assert isinstance(followup_result.function_arguments, dict)
            assert (
                followup_result.function_arguments.get("language", "").lower()
                == "javascript"
            )

    @pytest.mark.asyncio
    async def test_followup_with_functions_async(self, setup_real_llm):
        """Test followup with function calling - async"""

        @llm_function
        def search_docs(query: str, language: Literal["python", "javascript"]) -> str:
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
            return docs.get(language, f"No documentation found for '{language}'")

        @llm_prompt
        async def documentation_helper(
            question: str,
            functions: List[Callable],
            followup_handle: FollowupHandle = None,
        ) -> OutputWithFunctionCall:
            """{question}
            Provide answer using documentation search.
            """
            pass

        handle = FollowupHandle()

        result: OutputWithFunctionCall = await documentation_helper(
            question="What are the core functions of Python?",
            functions=[search_docs],
            followup_handle=handle,
        )

        if result.is_function_call:
            doc_result = result.execute()
            handle.message_history.append(result.function_output_to_message())
            assert "python" in doc_result.lower()

        # Test async followup with functions
        followup_result: OutputWithFunctionCall = await handle.afollowup(
            "Now tell me about JavaScript", with_tools=True
        )

        assert isinstance(followup_result, OutputWithFunctionCall)
        if followup_result.is_function_call:
            assert followup_result.function_name == "search_docs"
            assert isinstance(followup_result.function_arguments, dict)
            assert (
                followup_result.function_arguments.get("language", "").lower()
                == "javascript"
            )

    def test_conversation_chain_with_context_sync(self, setup_real_llm):
        """Test conversational chain that maintains context - sync"""
        conversation_history = []

        @llm_prompt
        def conversational_assistant(
            user_message: str, context_history: List[str] = None
        ) -> str:
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
            context_history=conversation_history,
        )
        conversation_history.append(
            f"User: I'm planning a trip to Japan\nAssistant: {response1}"
        )

        assert "japan" in response1.lower()

        # Follow-up message with context
        response2 = conversational_assistant(
            user_message="What's the best time to visit?",
            context_history=conversation_history,
        )

        assert isinstance(response2, str)
        # Should provide Japan-specific advice due to context
        assert (
            any(
                month in response2.lower()
                for month in [
                    "spring",
                    "summer",
                    "fall",
                    "winter",
                    "march",
                    "april",
                    "may",
                ]
            )
            or "japan" in response2.lower()
        )

    @pytest.mark.asyncio
    async def test_conversation_chain_with_context_async(self, setup_real_llm):
        """Test conversational chain that maintains context - async"""
        conversation_history = []

        @llm_prompt
        async def conversational_assistant(
            user_message: str, context_history: List[str] = None
        ) -> str:
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
        response1 = await conversational_assistant(
            user_message="I'm planning a trip to Japan",
            context_history=conversation_history,
        )
        conversation_history.append(f"User: I'm planning a trip to Japan\nAssistant: {response1}")

        assert "japan" in response1.lower()

        # Follow-up message with context
        response2 = await conversational_assistant(
            user_message="What's the best time to visit?",
            context_history=conversation_history,
        )

        assert isinstance(response2, str)
        # Should provide Japan-specific advice due to context
        assert any(month in response2.lower() for month in ["spring", "summer", "fall", "winter", "march", "april", "may"]) or "japan" in response2.lower()

    def test_multi_step_workflow_sync(self, setup_real_llm):
        """Test multi-step workflow with function chaining - sync"""
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

    @pytest.mark.asyncio
    async def test_multi_step_workflow_async(self, setup_real_llm):
        """Test multi-step workflow with function chaining - async"""

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
        async def text_workflow_manager(
            text_input: str, functions: List[Callable]
        ) -> OutputWithFunctionCall:
            """Manage text processing workflow for: {text_input}

            First analyze the text, then create a summary.
            """
            pass

        # Start workflow
        result = await text_workflow_manager(
            text_input="This is an amazing product that exceeded my expectations!",
            functions=[analyze_text, generate_summary],
        )

        if result.is_function_call:
            analysis_result = result.execute()
            assert "positive" in analysis_result.lower()

    def test_dynamic_function_schemas(self, setup_real_llm):
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
        assert (
            "educational resources"
            in schema["parameters"]["properties"]["query"]["description"]
        )


if __name__ == "__main__":
    configure_llm(True)
    TestChains().test_followup_with_functions_sync(None)
