import asyncio
from math import e
from typing import Literal
from langchain_decorators.langgraph import (
    LlmNodeBase,
    node,
    node_transition,
    SequentialGraph,
)
from langchain_core.messages import AIMessage
from langchain_decorators.langgraph.graphs import StagedGraph
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

        @node_transition("add_message")
        def finish(self) -> Literal["__end__"]:
            if self.status == "prepared":
                return "finish"

    graph = SequentialGraph.start().then(MyNode).then(("MyNode2", MyNode)).compile()
    return graph


def get_staged_graph():
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

        @node.after("add_message")
        def finish(self) -> Command:
            if len(self.messages) > 2 and self.messages[-2].content == "Done":
                return StagedGraph.CMD_NEXT
            else:
                return Command(goto="__end__")

    class MyNode2(LlmNodeBase):
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

        @node.after("add_message")
        def finish(self) -> Command:
            if len(self.messages) > 2 and self.messages[-2].content == "Done":
                return StagedGraph.CMD_NEXT
            else:
                return Command(goto="__end__")

    # graph = MyNode.compile()
    graph = StagedGraph.start().then(MyNode).then(MyNode2).compile()
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
    async for event in graph.astream(
        input, config=config, stream_mode="updates", subgraphs=True
    ):
        print(event[1])


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
    messages = ["Hey", "OK", "Done", "OK", "Done"]
    for message in messages:
        input["messages"].append({"role": "user", "content": message})
        async for event in graph.astream(
            input, config=config, stream_mode="messages", subgraphs=True
        ):
            print(event)


asyncio.run(stream_graph_updates_staging("Hello, world!"))
