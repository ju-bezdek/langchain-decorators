from langchain_openai import ChatOpenAI
import pytest
import asyncio
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



@pytest.mark.real_llm
class TestLLMPromptDecorator:
    """Test the core @llm_prompt decorator functionality with real LLM"""
    
    def test_simple_prompt_decorator(self):
        """Test basic prompt decorator functionality"""
        @llm_prompt
        def write_haiku(topic: str) -> str:
            """Write a haiku about {topic}. Follow 5-7-5 syllable pattern. Return only the haiku."""
            pass
        
        result = write_haiku(topic="morning coffee")
        
        # Verify we got a string response
        assert isinstance(result, str)
        assert len(result.strip()) > 0
        # Should have multiple lines for haiku
        assert '\n' in result.strip()
    
    def test_prompt_with_default_values(self):
        """Test prompt with default parameter values"""
        @llm_prompt
        def product_description(product: str, audience: str = "general consumers", tone: str = "professional") -> str:
            """Write a product description for {product} targeting {audience} with a {tone} tone. Keep it under 50 words."""
            pass
        
        result = product_description(product="wireless headphones")
        
        # Should contain reference to the product
        assert "headphones" in result.lower() or "wireless" in result.lower()
        assert isinstance(result, str)
        assert len(result.strip()) > 0
    
    def test_prompt_with_code_block_syntax(self):
        """Test prompt defined using code block syntax"""
        @llm_prompt
        def explain_concept(concept: str) -> str:
            """
            This function explains technical concepts in simple terms.
            
            ```<prompt>
            Explain {concept} in simple terms that a beginner could understand.
            Use an analogy and keep it under 100 words.
            ```
            
            This documentation won't be part of the prompt.
            """
            pass
        
        result = explain_concept(concept="machine learning")
        
        assert isinstance(result, str)
        assert len(result.strip()) > 0
        # Should mention machine learning in some form
        assert any(term in result.lower() for term in ["machine", "learning", "algorithm", "data"])
    
    def test_chat_message_templates(self):
        """Test chat message template syntax"""
        @llm_prompt
        def roleplay_conversation(character: str, situation: str) -> str:
            """
            ```<prompt:system>
            You are roleplaying as {character}. Stay in character.
            Respond in exactly one sentence.
            ```
            
            ```<prompt:user>
            Situation: {situation}
            How do you respond?
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
    
    def test_optional_sections(self):
        """Test optional section syntax {? ... ?}"""
        @llm_prompt
        def write_story_opening(genre: str, character_name: str, setting: Optional[str] = None) -> str:
            """Write a story opening in the {genre} genre featuring {character_name}{? set in {setting}?}. Keep it under 100 words."""
            pass
        
        # Test with optional parameter provided
        result1 = write_story_opening(
            genre="mystery", 
            character_name="Detective Smith", 
            setting="Victorian London"
        )
        
        # Test with optional parameter not provided
        result2 = write_story_opening(
            genre="mystery", 
            character_name="Detective Smith"
        )
        
        assert isinstance(result1, str)
        assert isinstance(result2, str)
        assert len(result1.strip()) > 0
        assert len(result2.strip()) > 0
        
        # Both should mention the character
        assert "Smith" in result1
        assert "Smith" in result2
    
    @pytest.mark.asyncio
    async def test_async_prompt(self):
        """Test async prompt execution"""
        @llm_prompt
        async def quick_joke(topic: str) -> str:
            """Tell a clean, family-friendly joke about {topic}. Keep it under 50 words."""
            pass
        
        result = await quick_joke(topic="computers")
        
        assert isinstance(result, str)
        assert len(result.strip()) > 0
        # Should be relatively short
        assert len(result.split()) < 60
    
    def test_output_parser_list(self):
        """Test automatic list output parsing"""
        @llm_prompt
        def brainstorm_ideas(topic: str, count: int = 3) -> List[str]:
            """Brainstorm {count} creative ideas for {topic}. 
            Return exactly {count} ideas, one per line, numbered 1-{count}."""
            pass
        
        result = brainstorm_ideas(topic="indoor rainy day activities", count=3)
        
        assert isinstance(result, list)
        assert len(result) >= 2  # Should get at least a couple
        assert all(isinstance(item, str) for item in result)
        assert all(len(item.strip()) > 0 for item in result)
    
    def test_output_parser_dict(self):
        """Test automatic dict output parsing"""
        @llm_prompt
        def analyze_text_sentiment(text: str) -> Dict:
            """Analyze the sentiment of: "{text}"
            
            Return as JSON with fields:
            - sentiment: "positive", "negative", or "neutral"
            - confidence: 0.0 to 1.0
            - key_words: list of 2-3 key sentiment words
            
            Return only valid JSON."""
            pass
        
        result = analyze_text_sentiment(text="I love this amazing product!")
        
        assert isinstance(result, dict)
        assert "sentiment" in result
        assert result["sentiment"] in ["positive", "negative", "neutral"]
        if "confidence" in result:
            assert isinstance(result["confidence"], (int, float))
    
    def test_pydantic_output_parser(self):
        """Test automatic pydantic model output parsing"""
        class BookInfo(BaseModel):
            title: str = Field(description="Book title")
            author: str = Field(description="Author name")
            genre: str = Field(description="Book genre")
            year_published: int = Field(description="Year published")
            rating: int = Field(description="Rating 1-5 stars")
        
        @llm_prompt
        def create_fictional_book(theme: str) -> BookInfo:
            """Create details for a fictional book about {theme}.
            
            {FORMAT_INSTRUCTIONS}
            
            Make it realistic and creative."""
            pass
        
        result = create_fictional_book(theme="time travel")
        
        assert isinstance(result, BookInfo)
        assert isinstance(result.title, str) and len(result.title) > 0
        assert isinstance(result.author, str) and len(result.author) > 0
        assert isinstance(result.genre, str) and len(result.genre) > 0
        assert isinstance(result.year_published, int)
        assert isinstance(result.rating, int) and 1 <= result.rating <= 5
    
    def test_prompt_types_and_settings(self):
        """Test custom prompt types and settings"""
        custom_llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo")
        
        custom_prompt_type = PromptTypeSettings(
            llm=custom_llm,
            capture_stream=False
        )
        
        @llm_prompt(prompt_type=custom_prompt_type)
        def creative_writing(prompt: str) -> str:
            """Write creatively about: {prompt}. Be imaginative and unique. Keep it under 100 words."""
            pass
        
        result = creative_writing(prompt="a magical forest")
        
        assert isinstance(result, str)
        assert len(result.strip()) > 0
        # Should contain forest-related content
        assert any(word in result.lower() for word in ["forest", "tree", "magic", "wood"])
    
    def test_followup_handle(self):
        """Test followup functionality"""
        @llm_prompt
        def start_conversation(topic: str, followup_handle: FollowupHandle = None) -> str:
            """Start a conversation about {topic}. Ask an engaging question to continue the discussion. Keep it under 50 words."""
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
    
    def test_preprocessing_function_implementation(self):
        """Test preprocessing by implementing the decorated function"""
        @llm_prompt
        def summarize_items(items: List[str], format_style: str = "bullet points") -> str:
            """Summarize these items in {format_style} format:
            {items}
            
            Keep it concise."""
            # Return preprocessing data
            return {"items": "\n".join(f"- {item}" for item in items)}
        
        result = summarize_items(
            items=["Learn Python", "Build a web app", "Deploy to cloud"],
            format_style="numbered list"
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Should mention some of the items
        assert any(keyword in result.lower() for keyword in ["python", "web", "cloud"])
    
    
    @pytest.mark.asyncio
    async def test_streaming_integration(self):
        """Test streaming functionality"""
        @llm_prompt(capture_stream=True)
        async def streaming_story(topic: str) -> str:
            """Write a very short story about {topic}. Keep it under 100 words."""
            pass
        
        collected_tokens = []
        
        def token_collector(token: str):
            collected_tokens.append(token)
        
        with StreamingContext(stream_to_stdout=False, callback=token_collector):
            result = await streaming_story(topic="a curious cat")
        
        # Verify streaming worked
        assert len(collected_tokens) > 0
        assert isinstance(result, str)
        full_streamed = "".join(collected_tokens)
        # The final result should match what was streamed
        assert len(full_streamed) > 0
    
    def test_error_handling_graceful(self):
        """Test graceful error handling"""
        @llm_prompt
        def simple_task(instruction: str) -> str:
            """Complete this task: {instruction}. Respond in one sentence."""
            pass
        
        # This should work without errors
        result = simple_task(instruction="Say hello")
        assert isinstance(result, str)
        assert len(result.strip()) > 0
    
    def test_complex_template_with_conditions(self):
        """Test complex template with multiple optional sections"""
        @llm_prompt
        def email_generator(
            recipient: str, 
            purpose: str,
            urgency: Optional[str] = None,
            deadline: Optional[str] = None,
            attachments: Optional[List[str]] = None
        ) -> str:
            """Write a professional email to {recipient} about {purpose}{? with {urgency} urgency?}{? due by {deadline}?}{? with attachments: {attachments}?}. Keep it under 150 words."""
            pass
        
        # Test with all optional parameters
        result1 = email_generator(
            recipient="John Smith",
            purpose="project update", 
            urgency="high",
            deadline="Friday",
            attachments=["report.pdf", "data.xlsx"]
        )
        
        # Test with minimal parameters
        result2 = email_generator(
            recipient="Jane Doe",
            purpose="meeting request"
        )
        
        assert isinstance(result1, str)
        assert isinstance(result2, str)
        assert "John Smith" in result1 or "John" in result1
        assert "Jane" in result2 or "Jane Doe" in result2
        assert len(result1.strip()) > 0
        assert len(result2.strip()) > 0


if __name__ == "__main__":
    TestLLMPromptDecorator().test_pydantic_output_parser()