from langchain_decorators.prompt_template import PromptDecoratorTemplate
from langchain_decorators.prompt_decorator import llm_prompt
import pytest
from langchain.schema.messages import AIMessage, HumanMessage, ToolMessage

from langchain_decorators.llm_chat_session import LlmChatSession
from langchain_decorators.function_decorator import get_function_schema, llm_function
from langchain.tools import tool
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)


@tool(parse_docstring=True)
def langchain_tool(instructions: str):
    """
    Use this if your are told to use langchain tool

    Args:
        instructions: instructions to pass to the function
    """
    return instructions + " executed"


@llm_function(dynamic_schema=True)
def express_emotion(emoji: str) -> str:
    """Use this tool to express your emotion as an emoji

    Args:
        emoji ({available_emojis}): the emoji to express

    """
    # In the example this function prints the emoji. For testing we return it,
    # so the ToolMessage content is deterministic and easy to assert.
    return emoji


def _find_last_tool_message(
    messages: list[ToolMessage], name: str
) -> ToolMessage | None:
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and getattr(m, "name", None) == name:
            return m
    return None


@llm_prompt
def prompt_example(user_input: str) -> str:
    """
    ```<prompt:system>
    You are a friendly but shy assistant. Try to reply with the least amount of words possible.

    Context:
    customer name is John
    current time is 2023-10-05T12:00:00
    ```
    ```<prompt:placeholder>
    {messages}
    ```
    ```<prompt:user>
    {user_input}
    ```
    """
    pass


def test_chat_session_tool_call_flow_matches_example():
    # Simulate the interactive session described in the example log without calling a real LLM.
    with LlmChatSession(tools=[express_emotion, langchain_tool]) as session:
        # 1) First user turn: "hi" => AI greets (no tool calls)
        session.add_message(HumanMessage(content="hi"))
        session.add_message(AIMessage(content="Hello, John! How can I help?"))
        assert (
            session.last_response_tool_calls == []
            or session.last_response_tool_calls is None
        )

        # 2) Second user turn: "how do you feel" => AI issues a tool call to express_emotion
        session.add_message(HumanMessage(content="how do you feel"))
        template = PromptDecoratorTemplate.from_func(prompt_example).get_final_template(
            user_input="how do you feel"
        )

        print("Prompt template:\n", template)
        with LlmChatSession(
            context={"available_emojis": ["😊", "😢", "😡"]}
        ) as prompt_session:
            func_schema = get_function_schema(express_emotion)

        assert "😊" in func_schema["parameters"]["properties"]["emoji"].get(
            "enum"
        ), "dynamic function schema"

        ai_tool_call_1 = AIMessage(
            content="",
            tool_calls=[
                {"name": "express_emotion", "args": {"emoji": "😊"}, "id": "call_1"}
            ],
        )
        session.add_message(ai_tool_call_1)

        # Verify tool call parsing
        assert session.is_calling_tool("express_emotion") is True
        assert (
            session.last_response_tool_calls
            and len(session.last_response_tool_calls) == 1
        )
        tc1 = session.last_response_tool_calls[0]
        assert tc1.name == "express_emotion"
        assert tc1.args == {"emoji": "😊"}

        # Execute tool calls and verify ToolMessage recorded in history
        session.execute_tool_calls()
        tm1 = _find_last_tool_message(session.message_history, name="express_emotion")
        assert tm1 is not None
        assert tm1.content == "😊"

        # 3) Third user turn: "call langchain" => AI asks for clarification (no tool calls)
        session.add_message(HumanMessage(content="call langchain"))
        session.add_message(AIMessage(content="What should I do with Langchain?"))
        assert (
            session.last_response_tool_calls == []
            or session.last_response_tool_calls is None
        )

        # 4) Fourth user turn: "use it to find the map" => AI calls langchain_tool
        session.add_message(HumanMessage(content="use it to find the map"))
        ai_tool_call_2 = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "langchain_tool",
                    "args": {"instructions": "find the map"},
                    "id": "call_2",
                }
            ],
        )
        session.add_message(ai_tool_call_2)

        assert session.is_calling_tool("langchain_tool") is True
        assert (
            session.last_response_tool_calls
            and len(session.last_response_tool_calls) == 1
        )
        tc2 = session.last_response_tool_calls[0]
        assert tc2.name == "langchain_tool"
        assert tc2.args == {"instructions": "find the map"}

        # Execute tool and verify result propagated as ToolMessage
        session.execute_tool_calls()
        tm2 = _find_last_tool_message(session.message_history, name="langchain_tool")
        assert tm2 is not None
        assert tm2.content == "find the map executed"


def test_example():
    from code_examples.tool_calling import Agent

    input_output = [
        ("hi, whats my name", lambda res, messages: "John" in res),
        (
            "my cat died...how do you feel?",
            lambda res, messages: messages[-1].type == "tool",
        ),
        (
            "now use langchain to find me new one",
            lambda res, messages: messages[-1].type == "tool",
        ),
    ]
    agent = Agent(customer_name="John")
    for user_input, assert_func in input_output:
        res = agent.invoke(user_input=user_input)
        assert assert_func(res, agent.memory)
