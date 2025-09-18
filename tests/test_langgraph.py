import asyncio
import pytest
from typing import Literal, TypedDict
from langchain_decorators.langgraph import (
    LlmNodeBase,
    node,
    conditional_transition,
    SequentialGraph,
)
from langchain_core.messages import AIMessage
from langchain_decorators.langgraph.graphs import StagedGraph, StagedGraphStateMin
from langgraph.types import Command
from langgraph.config import RunnableConfig


def get_sequence_graph():
    class MyNode(LlmNodeBase):
        status: str

        @node.after("__start__")
        def init(self):
            return {"status": "initialized"}

        @node.after("init")
        def prepare(self):
            return {"status": "prepared"}

        @node.after("init")
        def add_message(self):
            return {"messages": [AIMessage(content="Hello, world!")]}

        @conditional_transition("add_message")
        def finish(self) -> Literal["__end__"]:
            if self.status == "prepared":
                return "__end__"

    class MyNode2(LlmNodeBase):
        status: str
        status2: str

        @node.after("__start__")
        def init(self):
            return {"status2": "initialized"}

        @node.after("init")
        def prepare(self):
            return {"status2": "prepared"}

        @node.after("init")
        def add_message(self):
            return {"messages": [AIMessage(content="Hello, world!")]}

        @conditional_transition("add_message")
        def finish(self) -> Literal["__end__"]:
            if self.status2 == "prepared":
                return "finish"

    class State(TypedDict):
        status: str
        status2: str

    def print_state(state):
        print("\n\nState:")
        for key, val in state.items():
            print(f"{key}: {val}")

    graph = (
        SequentialGraph[State]
        .start()
        .then(MyNode)
        .then(MyNode2)
        .then(print_state)
        .compile()
    )
    return graph


def get_staged_graph():
    class MyNode(LlmNodeBase):
        current_page_url: str
        status: str

        @node.after("__start__")
        def init(self):
            return {"status": "initialized"}

        @node.after("init")
        def prepare(self):
            return {"status": "prepared"}

        @node.after("prepare")
        def add_message(self):
            return {"messages": [AIMessage(content="Hello, world!")]}

        @node.after("add_message")
        def finish(self) -> Command:
            if len(self.messages) > 2 and self.messages[-2].content == "Done":
                return StagedGraph.CMD_NEXT
            else:
                return Command(goto="__end__")

        @conditional_transition("add_message")
        def next_if_messages(self) -> Literal["finish"]:
            if len(self.messages) > 2 and self.messages[-2].content == "Done":
                return "finish"

    class MyNode2(LlmNodeBase):
        status: str
        type: str

        @node.after("__start__")
        def init(self):
            return {"status": "initialized"}

        @node.after("init")
        def prepare(self):
            return {"status": "prepared"}

        @node.after("init")
        def add_message(self):
            return {"messages": [AIMessage(content="Hello, world!")]}

        @node.after("add_message")
        def finish(self) -> Command:
            return Command(goto="__end__")

    # graph = MyNode.compile()

    class State(TypedDict):
        status: str
        status2: str

    graph = StagedGraph[State].start().then(MyNode).then(MyNode2).compile()
    return graph


async def stream_graph_updates(user_input: str):
    graph = get_sequence_graph()
    config = RunnableConfig(
        {
            "configurable": {
                "thread_id": "test_thread",
            },
        }
    )
    input = {
        "current_page_url": "https://www.imaginari.club",
    }
    if user_input:
        input["messages"] = [{"role": "user", "content": user_input}]

    results = []
    async for event in graph.astream(
        input, config=config, stream_mode="updates", subgraphs=True
    ):
        results.append(event[1])
    return results


async def stream_graph_updates_staging(user_input: str):
    graph = get_staged_graph()
    config = RunnableConfig(
        {
            "configurable": {
                "thread_id": "test_thread",
            },
        }
    )
    input = {
        "current_page_url": "https://www.imaginari.club",
    }
    if user_input:
        input["messages"] = [{"role": "user", "content": user_input}]
    else:
        input["messages"] = []
    messages = ["Hey", "OK", "Done", "OK", "Done"]
    results = []
    for message in messages:
        input["messages"].append({"role": "user", "content": message})
        async for event in graph.astream(
            input, config=config, stream_mode="messages", subgraphs=True
        ):
            results.append(event)
    return results


# ------------------- TESTS -------------------


@pytest.mark.asyncio
async def test_sequence_graph_runs():
    results = await stream_graph_updates("Hello, world!")
    # Just check that we get at least one event and it's a dict or similar
    assert results
    assert isinstance(results[0], dict) or hasattr(results[0], "__dict__")


@pytest.mark.asyncio
async def test_staged_graph_runs():
    results = await stream_graph_updates_staging("Hello!")
    assert results
    # Each event should be a tuple (event_type, event_data)
    assert all(isinstance(ev, tuple) and len(ev) == 2 for ev in results)


if __name__ == "__main__":
    # Allow direct run for quick manual check
    asyncio.run(stream_graph_updates_staging("Hello, world!"))
