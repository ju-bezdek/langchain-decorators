from langchain_openai import ChatOpenAI
import pytest
import os
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain.schema import AIMessage, HumanMessage
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory

from langchain_decorators import (
    llm_prompt, 
    GlobalSettings, 
    PromptTypes, 
    PromptTypeSettings,
    OutputWithFunctionCall,
    StreamingContext,
    FollowupHandle
)


@pytest.fixture(scope="session")
def setup_real_llm():
    """Setup real LLM for testing"""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set - skipping real LLM tests")

    real_llm = ChatOpenAI(
        temperature=0.0, model_name="gpt-3.5-turbo"  # Deterministic for testing
    )

    GlobalSettings.define_settings(default_llm=real_llm, verbose=False)
    return real_llm


@pytest.mark.real_llm
class TestLLMPromptDecorator:
    """Test the core @llm_prompt decorator functionality - focused tests without redundancy"""

    def test_basic_prompt_sync(self, setup_real_llm):
        """Test basic prompt decorator functionality - sync"""
        @llm_prompt
        def write_haiku(topic: str) -> str:
            """Write a haiku about {topic}. Keep it simple and traditional."""
            pass

        result = write_haiku(topic="morning coffee")

        # Verify we got a string response
        assert isinstance(result, str)
        assert len(result.strip()) > 0
        # Should have multiple lines for haiku
        assert '\n' in result.strip()

    @pytest.mark.asyncio
    async def test_basic_prompt_async(self, setup_real_llm):
        """Test basic prompt decorator functionality - async"""

        @llm_prompt
        async def write_haiku(topic: str) -> str:
            """Write a haiku about {topic}. Keep it simple and traditional."""
            pass

        result = await write_haiku(topic="morning coffee")

        # Verify we got a string response
        assert isinstance(result, str)
        assert len(result.strip()) > 0
        # Should have multiple lines for haiku
        assert "\n" in result.strip()

    def test_optional_template_parameters_sync(self, setup_real_llm):
        """Test optional template parameters with complex conditional logic - sync"""

        @llm_prompt
        def email_generator(
            recipient: str,
            purpose: str,
            urgency: Optional[str] = None,
            deadline: Optional[str] = None,
            attachments: Optional[List[str]] = None,
            tone: str = "professional",
        ) -> str:
            """
            ```<prompt:system>
            You are an email writing assistant. Write professional emails based on the provided parameters.
            ```

            ```<prompt:user>
            Write an email to {recipient} about {purpose}.

            {? The tone should be {tone}. ?}

            {? This is {urgency} priority. ?}

            {? The deadline is {deadline}. ?}

            {? Please mention these attachments: {attachments} ?}

            Keep it concise and professional.
            ```
            """
            pass

        # Test with all optional parameters
        result1 = email_generator(
            recipient="John Smith",
            purpose="project update",
            urgency="high",
            deadline="Friday",
            attachments=["report.pdf", "data.xlsx"],
            tone="urgent",
        )

        # Test with minimal parameters (most optional ones None)
        result2 = email_generator(recipient="Jane Doe", purpose="meeting request")

        assert isinstance(result1, str)
        assert isinstance(result2, str)
        assert "John Smith" in result1 or "John" in result1
        assert "Jane" in result2 or "Jane Doe" in result2
        assert len(result1.strip()) > 0
        assert len(result2.strip()) > 0

    @pytest.mark.asyncio
    async def test_optional_template_parameters_async(self, setup_real_llm):
        """Test optional template parameters with complex conditional logic - async"""

        @llm_prompt
        async def email_generator(
            recipient: str,
            purpose: str,
            urgency: Optional[str] = None,
            deadline: Optional[str] = None,
            attachments: Optional[List[str]] = None,
            tone: str = "professional",
        ) -> str:
            """
            ```<prompt:system>
            You are an email writing assistant. Write professional emails based on the provided parameters.
            ```

            ```<prompt:user>
            Write an email to {recipient} about {purpose}.

            {? The tone should be {tone}. ?}

            {? This is {urgency} priority. ?}

            {? The deadline is {deadline}. ?}

            {? Please mention these attachments: {attachments} ?}

            Keep it concise and professional.
            ```
            """
            pass

        # Test with all optional parameters
        result1 = await email_generator(
            recipient="John Smith",
            purpose="project update",
            urgency="high",
            deadline="Friday",
            attachments=["report.pdf", "data.xlsx"],
            tone="urgent",
        )

        # Test with minimal parameters (most optional ones None)
        result2 = await email_generator(recipient="Jane Doe", purpose="meeting request")

        assert isinstance(result1, str)
        assert isinstance(result2, str)
        assert "John Smith" in result1 or "John" in result1
        assert "Jane" in result2 or "Jane Doe" in result2
        assert len(result1.strip()) > 0
        assert len(result2.strip()) > 0

    def test_chat_message_templates_sync(self, setup_real_llm):
        """Test chat message template syntax - sync"""
        @llm_prompt
        def roleplay_conversation(character: str, situation: str) -> str:
            """
            ```<prompt:system>
            You are {character}. Stay in character and respond naturally.
            ```

            ```<prompt:user>
            {situation}

            Please respond with just one sentence as the character would.
            ```
            """
            pass

        result = roleplay_conversation(
            character="a friendly librarian", 
            situation="Someone asks for help finding a book about space"
        )

        assert isinstance(result, str)
        assert len(result.strip()) > 0
        # Should be relatively short (one sentence as requested)
        assert len(result) < 200

    @pytest.mark.asyncio
    async def test_chat_message_templates_async(self, setup_real_llm):
        """Test chat message template syntax - async"""

        @llm_prompt
        async def roleplay_conversation(character: str, situation: str) -> str:
            """
            ```<prompt:system>
            You are {character}. Stay in character and respond naturally.
            ```

            ```<prompt:user>
            {situation}

            Please respond with just one sentence as the character would.
            ```
            """
            pass

        result = await roleplay_conversation(
            character="a friendly librarian",
            situation="Someone asks for help finding a book about space",
        )

        assert isinstance(result, str)
        assert len(result.strip()) > 0
        # Should be relatively short (one sentence as requested)
        assert len(result) < 200

    def test_output_parser_list_sync(self, setup_real_llm):
        """Test automatic list output parsing - sync"""
        @llm_prompt
        def brainstorm_ideas(topic: str, count: int = 3) -> List[str]:
            """Brainstorm {count} creative ideas for {topic}. Return as a list."""
            pass

        result = brainstorm_ideas(topic="indoor rainy day activities", count=3)

        assert isinstance(result, list)
        assert len(result) >= 2  # Should get at least a couple
        assert all(isinstance(item, str) for item in result)
        assert all(len(item.strip()) > 0 for item in result)

    @pytest.mark.asyncio
    async def test_output_parser_list_async(self, setup_real_llm):
        """Test automatic list output parsing - async"""

        @llm_prompt
        async def brainstorm_ideas(topic: str, count: int = 3) -> List[str]:
            """Brainstorm {count} creative ideas for {topic}. Return as a list."""
            pass

        result = await brainstorm_ideas(topic="indoor rainy day activities", count=3)

        assert isinstance(result, list)
        assert len(result) >= 2  # Should get at least a couple
        assert all(isinstance(item, str) for item in result)
        assert all(len(item.strip()) > 0 for item in result)

    def test_pydantic_output_parser_sync(self, setup_real_llm):
        """Test automatic pydantic model output parsing - sync"""
        class BookInfo(BaseModel):
            title: str = Field(description="Book title")
            author: str = Field(description="Author name")
            genre: str = Field(description="Book genre")
            year_published: int = Field(description="Year published")
            rating: int = Field(description="Rating 1-5 stars")

        @llm_prompt
        def create_fictional_book(theme: str) -> BookInfo:
            """Create a fictional book about {theme}. Make up realistic details."""
            pass

        result = create_fictional_book(theme="time travel")

        assert isinstance(result, BookInfo)
        assert isinstance(result.title, str) and len(result.title) > 0
        assert isinstance(result.author, str) and len(result.author) > 0
        assert isinstance(result.genre, str) and len(result.genre) > 0
        assert isinstance(result.year_published, int)
        assert isinstance(result.rating, int) and 1 <= result.rating <= 5

    @pytest.mark.asyncio
    async def test_pydantic_output_parser_async(self, setup_real_llm):
        """Test automatic pydantic model output parsing - async"""

        class BookInfo(BaseModel):
            title: str = Field(description="Book title")
            author: str = Field(description="Author name")
            genre: str = Field(description="Book genre")
            year_published: int = Field(description="Year published")
            rating: int = Field(description="Rating 1-5 stars")

        @llm_prompt
        async def create_fictional_book(theme: str) -> BookInfo:
            """Create a fictional book about {theme}. Make up realistic details."""
            pass

        result = await create_fictional_book(theme="time travel")

        assert isinstance(result, BookInfo)
        assert isinstance(result.title, str) and len(result.title) > 0
        assert isinstance(result.author, str) and len(result.author) > 0
        assert isinstance(result.genre, str) and len(result.genre) > 0
        assert isinstance(result.year_published, int)
        assert isinstance(result.rating, int) and 1 <= result.rating <= 5

    def test_followup_handle_sync(self, setup_real_llm):
        """Test followup functionality - sync"""
        @llm_prompt
        def start_conversation(topic: str, followup_handle: FollowupHandle = None) -> str:
            """Let's discuss {topic}. Give me a brief overview to start our conversation."""
            pass

        handle = FollowupHandle()
        result = start_conversation(topic="space exploration", followup_handle=handle)

        # Verify the handle was bound to the chain
        assert handle.chain is not None
        assert isinstance(result, str)

        # Test followup functionality
        followup_result = handle.followup("Tell me more about Mars missions")
        assert isinstance(followup_result, str)
        assert len(followup_result.strip()) > 0

    @pytest.mark.asyncio
    async def test_followup_handle_async(self, setup_real_llm):
        """Test followup functionality - async"""

        @llm_prompt
        async def start_conversation(
            topic: str, followup_handle: FollowupHandle = None
        ) -> str:
            """Let's discuss {topic}. Give me a brief overview to start our conversation."""
            pass

        handle = FollowupHandle()
        result = await start_conversation(
            topic="space exploration", followup_handle=handle
        )

        # Verify the handle was bound to the chain
        assert handle.chain is not None
        assert isinstance(result, str)

        # Test followup functionality - note: followup might need to be awaited depending on implementation
        followup_result = handle.followup("Tell me more about Mars missions")
        assert isinstance(followup_result, str)
        assert len(followup_result.strip()) > 0

    @pytest.mark.asyncio
    async def test_streaming_integration(self, setup_real_llm):
        """Test streaming integration with async prompt"""
        collected_tokens = []

        def token_collector(token: str):
            collected_tokens.append(token)

        @llm_prompt(capture_stream=True)
        async def streaming_writer(topic: str) -> str:
            """Write a short paragraph about {topic}."""
            pass

        # Test streaming response
        with StreamingContext(stream_to_stdout=False, callback=token_collector):
            result = await streaming_writer(topic="artificial intelligence")

        # Verify streaming worked
        assert len(collected_tokens) > 0
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_custom_prompt_types_sync(self, setup_real_llm):
        """Test custom prompt types and settings - sync"""
        from langchain_openai import ChatOpenAI

        custom_llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo")

        custom_prompt_type = PromptTypeSettings(llm=custom_llm, capture_stream=False)

        @llm_prompt(prompt_type=custom_prompt_type)
        def creative_writing(prompt: str) -> str:
            """Write something creative about {prompt}. Be imaginative and artistic."""
            pass

        result = creative_writing(prompt="a magical forest")

        assert isinstance(result, str)
        assert len(result.strip()) > 0
        # Should contain forest-related content
        assert any(
            word in result.lower() for word in ["forest", "tree", "magic", "wood"]
        )

    @pytest.mark.asyncio
    async def test_custom_prompt_types_async(self, setup_real_llm):
        """Test custom prompt types and settings - async"""
        from langchain_openai import ChatOpenAI

        custom_llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo")

        custom_prompt_type = PromptTypeSettings(llm=custom_llm, capture_stream=False)

        @llm_prompt(prompt_type=custom_prompt_type)
        async def creative_writing(prompt: str) -> str:
            """Write something creative about {prompt}. Be imaginative and artistic."""
            pass

        result = await creative_writing(prompt="a magical forest")

        assert isinstance(result, str)
        assert len(result.strip()) > 0
        # Should contain forest-related content
        assert any(
            word in result.lower() for word in ["forest", "tree", "magic", "wood"]
        )


if __name__ == "__main__":
    TestLLMPromptDecorator().test_pydantic_output_parser_sync(None)
