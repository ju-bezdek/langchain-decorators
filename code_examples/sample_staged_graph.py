from typing import Annotated, Literal, TypedDict
import asyncio
from langgraph.types import Command
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_decorators.langgraph import (
    node_tool,
    node,
    LlmNodeBase,
    StagedGraph,
    conditional_transition,
)
from langchain_decorators import llm_prompt, GlobalSettings
from langchain_core.messages import AIMessage, ToolMessage, ToolCall, HumanMessage
from langchain_core.tools import InjectedToolCallId

GlobalSettings.define_settings(default_llm="openai:gpt-5")


class GraphState(TypedDict):
    """
    Generic state for a staged Trip Itinerary Planner example.
    """

    stage: Literal["init", "discovery", "refinement"]
    messages: Annotated[list[AnyMessage], add_messages]
    # Collected inputs and outputs during the flow
    preferences: dict | None
    selected_destination: str | None
    destination_info: str | None
    itinerary_outline: list[str] | None
    itinerary_suggestions: list[str] | None
    agent_stage: str | None


class TripPlannerAgent(LlmNodeBase):
    """
    Stage 1 agent:
    - Gather traveler preferences
    - Refine options (destination, dates, interests)
    - Finalize a high-level itinerary outline
    """

    preferences: dict | None = None
    selected_destination: str | None = None
    destination_info: str | None = None
    itinerary_outline: list[str] | None = None
    agent_stage: str | None = None

    @node.after("__start__")
    async def init(self, agent_stage: str):
        # Decide which sub-stage of this agent we are in based on the conversation.
        if agent_stage:
            res = await self.analyse_stage_prompt()
            return {"agent_stage": res["next_stage"]}
        else:
            return {"agent_stage": "discovery"}  # start here

    @conditional_transition(after="init")
    def route_prompt(
        self,
    ) -> Literal["prompt_1_discover", "prompt_2_refine", "prompt_3_finalize"]:
        if self.agent_stage in (None, "discovery"):
            return "prompt_1_discover"
        elif self.agent_stage == "refinement":
            return "prompt_2_refine"
        elif self.agent_stage == "finalize":
            return "prompt_3_finalize"
        else:
            return "prompt_2_refine"  # default

    @node
    @llm_prompt(model="openai:gpt-5-nano")
    async def prompt_1_discover(self):
        """
        ```<prompt:system>
        Collect info about where are we going
        Make your replies brief and concise.
        ```
        ```<prompt:placeholder>
        {messages}
        ```
        """

    @node
    @llm_prompt(model="openai:gpt-5-nano")
    async def prompt_2_refine(self):
        """
        ```<prompt:system>
        Find out from where, how long and when we are going
        Make your replies brief and concise.

        ```
        ```<prompt:placeholder>
        {messages}
        ```
        """
        return {}

    @node
    @llm_prompt(model="openai:gpt-5-nano")
    async def prompt_3_finalize(self):
        """
        ```<prompt:system>
        Summarize the conversation for the user. Do not offer anything else... that will happen in the next stage.
        Make your replies brief and concise.

        If user is OK with the plan, use finish tool to move to the next stage.
        ```
        ```<prompt:placeholder>
        {messages}
        ```
        """

    @llm_prompt(model="openai:gpt-5-nano", capture_stream=False)
    async def analyse_stage_prompt(self) -> dict:
        """
        ```<prompt:system>
        Determine which sub-stage the conversation is in:
        - "discovery": gather where are we going
        - "refinement": basic info like, when and for how long
        - "finalize": summarizing an outline before handing off to detailed planning (if user starts talking about booking, skip here)
        Always reply JSON:
        {{
          "current_stage": "discovery" | "refinement" | "finalize",
          "next_stage": "discovery" | "refinement" | "finalize"
        }}
        ```
        ```<prompt:placeholder>
        {messages}
        ```
        ```<prompt:user>
        In which sub-stage is this conversation now? Reply JSON only.
        ```
        """

    @node_tool()
    async def lookup_destination(
        self, destination: str, tool_call_id: Annotated[str, InjectedToolCallId]
    ):
        """
        Return a short overview of a destination (simulated).
        """
        OVERVIEW = {
            "Lisbon": "Lisbon offers coastal views, historic neighborhoods (Alfama, Bairro Alto), nearby day trips (Sintra, Cascais), and great food.",
            "Kyoto": "Kyoto features temples, gardens, tea culture, walkable districts (Gion, Arashiyama), and seasonal highlights (cherry blossoms, autumn foliage).",
            "Reykjavik": "Gateway to Iceland’s nature: Golden Circle, Blue Lagoon, waterfalls, and northern lights (seasonal). Compact, easy base for day trips.",
        }
        info = OVERVIEW.get(
            destination,
            f"Quick overview for {destination}: vibrant city, cultural highlights, and accessible day trips.",
        )
        return info

    @node_tool(bind_to_prompt_nodes=["prompt_3_finalize"])
    async def finish(self) -> Command:
        """
        Move to the next stage (detailed planning).
        """
        return StagedGraph.CMD_NEXT


class TravelAgent(LlmNodeBase):

    preferences: dict | None = None
    selected_destination: str | None = None
    destination_info: str | None = None
    itinerary_outline: list[str] | None = None
    itinerary_suggestions: list[str] | None = None

    @node.after("__start__")
    @llm_prompt
    async def prompt(self):
        """
        ```<prompt:system>
        Create a concise day-by-day plan from the outline.
        Keep it brief, practical, and paced. Offer small alternatives (A/B choices) when reasonable.
        Ask if the user wants to adjust timing or swap activities.
        Make your replies brief and concise.
        ```

        ```<prompt:placeholder>
        {messages}
        ```
        """

    @node_tool(require_confirmation=True)
    def book_flights(
        self, from_airport: str, to_airport: str, date: str, count: int = 1
    ) -> str:
        """
        Simulate booking flights (dummy implementation).
        """
        return "Flights booked successfully."

    @node_tool(require_confirmation=True)
    def book_car(
        self,
        pickup_point: str,
        from_date: str,
        to_date: str,
        car_type: str = "standard",
    ) -> str:
        """
        Simulate booking a car (dummy implementation).
        """
        return "Car booked successfully."

    @node_tool
    def book_hotel(
        self, city: str, hotel_name: str, check_in: str, check_out: str
    ) -> str:
        """
        Simulate booking a hotel (dummy implementation).
        """
        return "Hotel booked successfully."

    @node_tool
    def find_hotel(self, city: str, min_stars: int, max_price: int) -> str:
        """
        Simulate finding a hotel (dummy implementation).
        """

        @llm_prompt(model="openai:gpt-5-nano")
        async def fake_hotel_search(min_stars: int, max_price: int) -> str:
            """
            ```<prompt:system>
            Your task is to simulate a hotel search.

            Reply with structure:

            # HOTEL NAME
            - Location: Address
            - Stars: X
            - Price: $Y per night
            - Amenities: list of amenities
            - Description: brief description

            always offer 3 fake hotel names

            ```
            ```<prompt:user>
            Fake hotel names in {city} with at least {min_stars} stars and a maximum price of ${max_price} per night.
            ```
            """
            return "Hotel Example Name"

    @node_tool
    async def finish(self) -> Command:
        """
        End of the staged flow.
        """
        return StagedGraph.CMD_NEXT


async def init_state(state) -> Command:
    # Initialize empty state for the sample
    return Command(
        update={
            "preferences": None,
            "selected_destination": None,
            "destination_info": None,
            "itinerary_outline": None,
            "itinerary_suggestions": None,
            "agent_stage": None,
        },
        goto="__next__",
    )


def get_graph(**kwargs):
    graph = (
        StagedGraph[GraphState]
        .start()
        .then(init_state)
        .then(TripPlannerAgent)
        .then(TravelAgent)
        .compile(**kwargs)
    )
    return graph


# Graph for langgraph cli
graph = get_graph()

# --- Added for console interaction when running this file directly ---


def _extract_text(content) -> str:
    # Handle string or structured content [{"type":"text","text": "..."}]
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(p for p in parts if p).strip()
    return str(content)


async def _console_run():
    print("Trip Planner console. Type 'exit' or 'quit' to leave.")
    saver = InMemorySaver()
    graph = get_graph(checkpointer=saver)
    initial_state = {
        "messages": [],
    }
    default_answers = [
        "OK",
        "Ok, cool.. i want to rent a car .. i want to see the nature highlights",
        "Ok, the first option",
        "OK, looks good, lets go with this",
        "OK, I will arrive Sep. 5 at 10:00 AM",
        "OK",
        "Ok, book me hotels and flights from NYC ",
        "JFK, 2 people",
        "yes",
    ]

    def _input(prompt, default_answer):
        if default_answer:
            print(
                f"\033[90m{default_answer} (default answer, just hit Enter to confirm.. or write your own)\033[0m"
            )
        return input(prompt).strip() or default_answer or ""

    user = _input(
        "You: ",
        "I'm going to Iceland for 5 days in September, first give basic info about it",
    )
    invoke_data = Command(update={"messages": [HumanMessage(content=user)]})
    while True:

        default_answer = default_answers.pop(0) if default_answers else None
        prev_len = 0
        # Send only the delta message; add_messages will append it to the thread.
        config = {"configurable": {"thread_id": "12345"}}
        new_state = await graph.ainvoke(invoke_data, config)

        if new_state.get("__interrupt__"):
            print("Approval needed for tool calls:")
            for intr in new_state["__interrupt__"]:
                print(intr.value)
            approval = (
                _input("Approve tool calls? (y/n): ", default_answer).strip().lower()
            )
            if approval == "y":
                invoke_data = Command(resume=True)
            else:
                invoke_data = Command(resume=False)
            continue
        else:
            invoke_data = initial_state
            # Print only new AI messages since last turn
            msgs = new_state.get("messages", [])
            for m in msgs[prev_len:]:
                if getattr(m, "type", None) == "ai":
                    text = _extract_text(m.content)
                    if text:
                        print(f"Agent: {text}")
            invoke_data = new_state
            prev_len = len(msgs)
        print("Current state:", new_state.get("_current_stage"))
        try:
            user = _input("You: ", default_answer).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break

        initial_state["messages"].append(HumanMessage(content=user))
        invoke_data = initial_state


if __name__ == "__main__":
    asyncio.run(_console_run())
# --- End console runner additions ---
