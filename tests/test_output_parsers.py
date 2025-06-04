from textwrap import dedent
import pytest
import json
from unittest.mock import Mock
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

from langchain_decorators.output_parsers import (
    PydanticOutputParser,
    ListOutputParser,
    JsonOutputParser,
    BooleanOutputParser,
    MarkdownStructureParser,
    OpenAIFunctionsPydanticOutputParser,
    OutputParserExceptionWithOriginal,
    ErrorCodes
)


class TestOutputParsers:
    """Test output parser functionality"""
    
    def test_pydantic_output_parser(self):
        """Test PydanticOutputParser"""
        class Person(BaseModel):
            name: str = Field(description="Person's name")
            age: int = Field(description="Person's age")
            city: str = Field(description="Person's city")
        
        parser = PydanticOutputParser(model=Person)
        
        # Test successful parsing
        json_str = '{"name": "John Doe", "age": 30, "city": "New York"}'
        result = parser.parse(json_str)
        
        assert isinstance(result, Person)
        assert result.name == "John Doe"
        assert result.age == 30
        assert result.city == "New York"
        
        # Test format instructions
        instructions = parser.get_format_instructions()
        assert "name" in instructions
        assert "age" in instructions
        assert "city" in instructions
    
    def test_pydantic_output_parser_with_list(self):
        """Test PydanticOutputParser with list output"""
        class Item(BaseModel):
            name: str
            value: int
        
        parser = PydanticOutputParser(model=Item, as_list=True)
        
        # Test parsing list of items
        json_str = '[{"name": "item1", "value": 10}, {"name": "item2", "value": 20}]'
        result = parser.parse(json_str)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(item, Item) for item in result)
        assert result[0].name == "item1"
        assert result[1].value == 20
    
    def test_list_output_parser(self):
        """Test ListOutputParser"""
        parser = ListOutputParser()
        
        # Test numbered list
        numbered_text = "1. First item\n2. Second item\n3. Third item"
        result = parser.parse(numbered_text)
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert "First item" in result[0]
        assert "Second item" in result[1]
        assert "Third item" in result[2]
        
        # Test bulleted list
        bulleted_text = "- First item\n- Second item\n- Third item"
        result = parser.parse(bulleted_text)
        
        assert isinstance(result, list)
        assert len(result) == 3
    
    def test_json_output_parser(self):
        """Test JsonOutputParser"""
        parser = JsonOutputParser()
        
        # Test valid JSON
        json_str = '{"key1": "value1", "key2": 42, "key3": [1, 2, 3]}'
        result = parser.parse(json_str)
        
        assert isinstance(result, dict)
        assert result["key1"] == "value1"
        assert result["key2"] == 42
        assert result["key3"] == [1, 2, 3]
        
        # Test JSON with markdown code blocks
        markdown_json = """
        Here's the result:
        ```json
        {"name": "test", "value": 123}
        ```
        """
        result = parser.parse(markdown_json)
        
        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 123
    
    def test_boolean_output_parser(self):
        """Test BooleanOutputParser"""
        parser = BooleanOutputParser()
        
        # Test positive responses
        assert parser.parse("Yes") == True
        assert parser.parse("yes") == True
        assert parser.parse("YES") == True
        assert parser.parse("True") == True
        assert parser.parse("true") == True
        
        # Test negative responses
        assert parser.parse("No") == False
        assert parser.parse("no") == False
        assert parser.parse("NO") == False
        assert parser.parse("False") == False
        assert parser.parse("false") == False
        
        # Test with surrounding text
        assert parser.parse("Yes, I agree with that statement.") == True
        assert parser.parse("No, that's not correct.") == False
    
    def test_markdown_structure_parser(self):
        """Test MarkdownStructureParser"""
        class DocumentStructure(BaseModel):
            introduction: str = Field(description="Introduction section")
            main_content: str = Field(description="Main content section")
            conclusion: str = Field(description="Conclusion section")
        
        parser = MarkdownStructureParser(model=DocumentStructure)
        
        # Test markdown with sections
        markdown_text = dedent("""
        # Introduction
        This is the introduction section with important information.
        
        # Main Content
        This is the main content with detailed explanations.
        
        # Conclusion
        This is the conclusion that summarizes everything.
        """)
        
        result = parser.parse(markdown_text)
        
        assert isinstance(result, DocumentStructure)
        assert result.introduction
        assert result.main_content
        assert result.conclusion
    
    def test_openai_functions_pydantic_parser(self):
        """Test OpenAIFunctionsPydanticOutputParser"""
        class FunctionResult(BaseModel):
            action: str = Field(description="Action to take")
            parameters: Dict = Field(description="Action parameters")
        
        parser = OpenAIFunctionsPydanticOutputParser(model=FunctionResult)
        
        # Test function call parsing
        function_args = {
            "action": "send_email",
            "parameters": {"to": "user@example.com", "subject": "Test"}
        }
        # Test LLM function building
        llm_function = parser.build_llm_function()
        
        result = parser.parse(function_args)
        
        assert isinstance(result, FunctionResult)
        assert result.action == "send_email"
        assert result.parameters["to"] == "user@example.com"
        
        
        
    

    
    def test_output_parser_with_format_instructions(self):
        """Test output parsers with format instructions integration"""
        class Product(BaseModel):
            name: str = Field(description="Product name")
            price: float = Field(description="Product price")
            in_stock: bool = Field(description="Whether product is in stock")
        
        parser = PydanticOutputParser(model=Product)
        
        # Test that format instructions are generated
        instructions = parser.get_format_instructions()
        
        assert "name" in instructions
        assert "price" in instructions
        assert "in_stock" in instructions
        assert "json" in instructions.lower()
    
    def test_nested_pydantic_models(self):
        """Test parsing nested Pydantic models"""
        class Address(BaseModel):
            street: str
            city: str
            country: str
        
        class Person(BaseModel):
            name: str
            address: Address
            
        parser = PydanticOutputParser(model=Person)
        
        json_str = '''
        {
            "name": "John Doe",
            "address": {
                "street": "123 Main St",
                "city": "New York",
                "country": "USA"
            }
        }
        '''
        
        result = parser.parse(json_str)
        
        assert isinstance(result, Person)
        assert result.name == "John Doe"
        assert isinstance(result.address, Address)
        assert result.address.city == "New York"
    
   
    
    def test_json_output_parser_edge_cases(self):
        """Test JsonOutputParser edge cases"""
        parser = JsonOutputParser()
        
        # Test JSON in markdown with extra text
        complex_markdown = """
        Here's some explanation text.
        
        ```json
        {"result": "success", "data": {"count": 5}}
        ```
        
        And here's more text after.
        """
        
        result = parser.parse(complex_markdown)
        assert result["result"] == "success"
        assert result["data"]["count"] == 5
        
        # Test invalid JSON should raise exception
        with pytest.raises(OutputParserExceptionWithOriginal):
            parser.parse("definitely not json")
    
    def test_parser_type_properties(self):
        """Test that parsers have correct type properties"""
        assert PydanticOutputParser(model=BaseModel)._type == "pydantic"
        assert ListOutputParser()._type == "list"
        assert JsonOutputParser()._type == "json"
        assert BooleanOutputParser()._type == "boolean"


if __name__ == "__main__":
    TestOutputParsers().test_json_output_parser_edge_cases()

