import pytest
import logging
from unittest.mock import Mock, patch
from typing import List, Dict, Optional
from langchain.chat_models.base import BaseChatModel

from langchain_decorators import (
    GlobalSettings,
    PromptTypes,
    PromptTypeSettings,
    LlmSelector,
    LogColors,
    print_log
)
from langchain_decorators.common import (
    get_function_docs,
    get_function_full_name,
    get_func_return_type,
    init_chat_model,
    MODEL_LIMITS
)


class TestCommonUtilities:
    """Test common utilities and global settings"""
    
    def setup_method(self):
        """Setup test environment"""
        # Reset global settings for each test
        GlobalSettings._instance = None
    
    def test_global_settings_singleton(self):
        """Test GlobalSettings singleton pattern"""
        settings1 = GlobalSettings.get_current_settings()
        settings2 = GlobalSettings.get_current_settings()
        
        assert settings1 is settings2  # Should be the same instance
    
    def test_global_settings_define_settings(self):
        """Test GlobalSettings.define_settings"""
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI()
        
        GlobalSettings.define_settings(
            default_llm=llm,
            verbose=True,
            default_streaming=True
        )
        
        settings = GlobalSettings.get_current_settings()
        assert settings.default_llm is llm
        assert settings.verbose == True
        assert settings.default_streaming == True
    
    def test_prompt_types_built_in(self):
        """Test built-in prompt types"""
        # Test that built-in prompt types exist
        assert hasattr(PromptTypes, 'UNDEFINED')
        assert hasattr(PromptTypes, 'AGENT_REASONING')
        
        # Test that they are PromptTypeSettings instances
        assert isinstance(PromptTypes.UNDEFINED, PromptTypeSettings)
        assert isinstance(PromptTypes.AGENT_REASONING, PromptTypeSettings)
        
        # Test different colors for different types
        assert PromptTypes.UNDEFINED.color != PromptTypes.AGENT_REASONING.color
    
    def test_custom_prompt_type_settings(self):
        """Test creating custom PromptTypeSettings"""
        custom_llm = Mock(spec=BaseChatModel)
        
        custom_type = PromptTypeSettings(
            llm=custom_llm,
            color=LogColors.GREEN,
            log_level="debug",
            capture_stream=True
        )
        
        assert custom_type.llm == custom_llm
        assert custom_type.color == LogColors.GREEN
        assert custom_type.log_level == logging.DEBUG
        assert custom_type.capture_stream == True
    
    def test_prompt_type_settings_as_verbose(self):
        """Test PromptTypeSettings.as_verbose() method"""
        original_type = PromptTypeSettings(
            color=LogColors.BLUE,
            log_level="info"
        )
        
        verbose_type = original_type.as_verbose()
        
        assert verbose_type.color == original_type.color
        assert verbose_type.log_level == 100  # Verbose log level
        assert verbose_type.llm == original_type.llm
    
   
    
    def test_llm_selector_model_limits(self):
        """Test LlmSelector model window limits"""
        selector = LlmSelector()
        
        # Test known model limits
        assert selector.get_model_window("gpt-3.5-turbo") == 4_096
        assert selector.get_model_window("gpt-4-32k") == 32_768
        assert selector.get_model_window("gpt-4-1106-preview") == 128_000
        
        # Test unknown model (should return None or default)
        unknown_limit = selector.get_model_window("unknown-model")
        assert unknown_limit is None or isinstance(unknown_limit, int)
    
    
    def test_print_log_function(self):
        """Test print_log function"""
        with patch('builtins.print') as mock_print:
            print_log(
                log_object="Test message",
                log_level=logging.INFO,
                color=LogColors.GREEN
            )
            
            # Verify print was called
            mock_print.assert_called_once()
            
            # Check that the message contains our text
            call_args = mock_print.call_args[0][0]
            assert "Test message" in call_args
    
    def test_get_function_docs(self):
        """Test get_function_docs utility"""
        def test_function():
            """This is a test function.
            
            It has multiple lines of documentation.
            And some more details here.
            """
            pass
        
        docs = get_function_docs(test_function)
        
        assert "This is a test function." in docs
        assert "multiple lines" in docs
        assert "more details" in docs
    
    def test_get_function_full_name(self):
        """Test get_function_full_name utility"""
        def test_function():
            pass
        
        # Test function in main module
        full_name = get_function_full_name(test_function)
        assert "test_function" in full_name
        
        # For functions in __main__, should just return function name
        if test_function.__module__ == "__main__":
            assert full_name == "test_function"
        else:
            assert "." in full_name  # Should include module name
    
    def test_get_func_return_type(self):
        """Test get_func_return_type utility"""
        def string_return() -> str:
            return "string"
        
        def int_return() -> int:
            return 42
        
        def list_return() -> List[str]:
            return []
        
        def no_annotation():
            return None
        
        # Test with type annotations
        assert get_func_return_type(string_return) == str
        assert get_func_return_type(int_return) == int
        assert get_func_return_type(list_return) == list
        assert get_func_return_type(list_return, with_args=True) == (list, [str])
        
        # Test without annotation
        assert get_func_return_type(no_annotation) is None
    
    def test_init_chat_model(self):
        """Test init_chat_model utility"""
        # Test that init_chat_model can create a model
        # This might be mocked depending on availability
        try:
            model = init_chat_model("gpt-3.5-turbo", temperature=0.7)
            assert model is not None
        except ImportError:
            # If langchain.chat_models.init_chat_model is not available
            # The fallback should create a ChatOpenAI instance
            with patch('langchain_decorators.common.ChatOpenAI') as mock_chat_openai:
                model = init_chat_model("gpt-3.5-turbo", temperature=0.7)
                mock_chat_openai.assert_called_once_with(model_name="gpt-3.5-turbo", temperature=0.7)
    
    def test_model_limits_constants(self):
        """Test MODEL_LIMITS constants"""
        # Test that MODEL_LIMITS contains expected models
        assert isinstance(MODEL_LIMITS, dict)
        assert len(MODEL_LIMITS) > 0
        
        # Test some known models
        gpt_35_pattern = next((k for k in MODEL_LIMITS.keys() if "gpt-3.5-turbo" in k), None)
        assert gpt_35_pattern is not None
        assert isinstance(MODEL_LIMITS[gpt_35_pattern], int)
        
        gpt_4_pattern = next((k for k in MODEL_LIMITS.keys() if "gpt-4" in k and "32k" not in k), None)
        assert gpt_4_pattern is not None
        assert isinstance(MODEL_LIMITS[gpt_4_pattern], int)
    
    def test_custom_prompt_types_inheritance(self):
        """Test creating custom prompt types through inheritance"""
        class CustomPromptTypes(PromptTypes):
            GPT4 = PromptTypeSettings(
                color=LogColors.PURPLE if hasattr(LogColors, 'PURPLE') else LogColors.BLUE,
                log_level="debug"
            )
            CREATIVE = PromptTypeSettings(
                color=LogColors.YELLOW,
                log_level="info",
                capture_stream=True
            )
        
        # Test that custom types are accessible
        assert hasattr(CustomPromptTypes, 'GPT4')
        assert hasattr(CustomPromptTypes, 'CREATIVE')
        assert hasattr(CustomPromptTypes, 'UNDEFINED')  # Should inherit from parent
        
        # Test that custom types have correct properties
        assert CustomPromptTypes.GPT4.log_level == logging.DEBUG
        assert CustomPromptTypes.CREATIVE.capture_stream == True
    
    def test_global_settings_with_custom_prompt_types(self):
        """Test GlobalSettings with custom prompt types"""
        class MyPromptTypes(PromptTypes):
            CUSTOM = PromptTypeSettings(
                color=LogColors.GREEN,
                log_level="warning"
            )
        
        GlobalSettings.define_settings(
            verbose=True,
            default_prompt_type=MyPromptTypes.CUSTOM
        )
        
        settings = GlobalSettings.get_current_settings()
        assert settings.verbose == True
        # The exact handling of default_prompt_type depends on implementation
    
   
    def test_prompt_type_settings_lazy_init(self):
        """Test PromptTypeSettings lazy initialization"""
        settings = PromptTypeSettings()
        
        # Test that prompt_template_builder is lazily initialized
        builder = settings.prompt_template_builder
        assert builder is not None
        
        # Test that accessing it again returns the same instance
        builder2 = settings.prompt_template_builder
        assert builder is builder2



if __name__ == "__main__":
    TestCommonUtilities().test_global_settings_define_settings()