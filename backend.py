import os
import certifi
from dotenv import load_dotenv
from langgraph.types import interrupt, Command

import operator
from typing import Any, TypedDict, Annotated
import uuid
import asyncio
import json

import nest_asyncio
nest_asyncio.apply()

from langgraph.graph import StateGraph, START, END
import psycopg
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)

from langchain_groq import ChatGroq

from mcp_client import (
    tavily_mcp_search,
    aviation_mcp_call,
    extract_destination,
    weather_mcp_search,
    forecast_mcp_search
)

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# =========================
# DATABASE
# =========================

def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL "
            "External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


# =========================
# API KEY
# =========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Please add it to your .env file."
    )


# =========================
# LLM
# =========================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY
)


# =========================
# STATE
# =========================

class TravelState(TypedDict, total=False):

    messages: Annotated[list[AnyMessage], operator.add]

    user_query: str

    # Supervisor + guardrail state
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    # Agent results
    flight_results: str
    hotel_results: str
    weather_results: str
    budget_results: str
    itinerary: str

    # HITL state
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    llm_calls: int


# =========================
# SHARED HELPERS
# =========================

KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}


AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


def _llm_text(system_prompt: str, user_prompt: str) -> str:

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    return str(response.content)


def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    return json.loads(text[start:end + 1])


def _empty_constraints() -> dict[str, Any]:

    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


# =========================
# SUPERVISOR AGENT
# =========================

def supervisor_agent(state: TravelState):

    query = state["user_query"]

    llm_calls = state.get("llm_calls", 0)

    # =========================
    # GUARDRAIL
    # =========================

    guardrail_prompt = f"""
Determine whether the following request belongs to travel planning or travel
information.

Valid requests can include destinations, flights, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing, or itineraries.

Block clearly unrelated requests and requests asking for harmful or illegal
instructions.

Do not block a valid travel request merely because some details are missing.

Return strict JSON only:

{{
  "allowed": true,
  "reason": ""
}}

User request:
{query}
"""

    try:

        guardrail_raw = _llm_text(
            "You are the input guardrail for a travel-planning application. "
            "Return strict JSON only.",
            guardrail_prompt,
        )

        guardrail_result = _json_from_llm(guardrail_raw)

        allowed = bool(
            guardrail_result.get("allowed", True)
        )

        guardrail_reason = str(
            guardrail_result.get("reason", "")
        ).strip()

        llm_calls += 1

    except Exception as exc:

        print(f"Guardrail fallback used: {exc}")

        allowed = True

        guardrail_reason = (
            "Guardrail validation fallback allowed the request."
        )

    # =========================
    # BLOCKED REQUEST
    # =========================

    if not allowed:

        reason = guardrail_reason or (
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, "
            "budget, or itinerary."
        )

        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [
                AIMessage(
                    content=f"Guardrail blocked request: {reason}"
                )
            ],
            "llm_calls": llm_calls,
        }

    # =========================
    # SUPERVISOR PROMPT
    # =========================

    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.

Choose only the specialist agents needed for the request.

Available agents:

- flight_agent: flights, airports, airlines, routes, airfare, or booking advice

- hotel_agent: hotels, accommodation, neighborhoods, or places to stay

- weather_agent: weather, climate, season, forecast, or packing advice

- budget_agent: cost, affordability, price limits, or budget feasibility

- itinerary_agent: creates the integrated travel plan when the user asks
  for a complete trip plan or itinerary

Important routing rules:

- Do NOT select agents that are unrelated to the user's request.
- A flight-only request should normally select only flight_agent.
- A hotel-only request should normally select only hotel_agent.
- A weather-only request should normally select only weather_agent.
- A budget-only request should normally select only budget_agent.
- A complete trip-planning request may select multiple specialist agents
  and itinerary_agent.
- itinerary_agent should be selected when an integrated itinerary or
  complete trip plan is requested.

Return strict JSON only:

{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query}
"""

    try:

        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. "
            "Return strict JSON only.",
            supervisor_prompt,
        )

        parsed = _json_from_llm(supervisor_raw)

        requested_agents = parsed.get(
            "selected_agents",
            []
        )

        selected_agents = [
            name
            for name in AGENT_ORDER
            if name in requested_agents
            and name in KNOWN_AGENTS
        ]

        constraints = _empty_constraints()

        parsed_constraints = parsed.get(
            "trip_constraints",
            {}
        )

        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(
            parsed.get("reasoning", "")
        ).strip()

        llm_calls += 1

    except Exception as exc:

        print(f"Supervisor fallback used: {exc}")

        # Original full workflow behavior is preserved
        # as the fallback.
        selected_agents = AGENT_ORDER.copy()

        constraints = _empty_constraints()

        reasoning = (
            "Supervisor parsing failed, so the original full travel "
            "workflow was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [
            AIMessage(
                content="Supervisor created the agent plan."
            )
        ],
        "llm_calls": llm_calls,
    }


# =========================
# GUARDRAIL BLOCKED AGENT
# =========================

def guardrail_blocked_agent(state: TravelState):

    reason = (
        state.get("final_response")
        or state.get("guardrail_reason")
        or "This request was blocked by the travel input guardrail."
    )

    return {
        "final_response": reason,
        "messages": [
            AIMessage(content=reason)
        ],
    }


# =========================
# FLIGHT AGENT
# =========================

FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:

1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""


def flight_agent(state: TravelState) -> dict:

    query = state["user_query"]

    try:

        airports = asyncio.run(
            aviation_mcp_call(
                "list_airports"
            )
        )

        airlines = asyncio.run(
            aviation_mcp_call(
                "list_airlines"
            )
        )

        print("\nAIRPORTS:", airports)
        print("\nAIRLINES:", airlines)

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:1500],
            airline_data=str(airlines)[:1500]
        )

        response = llm.invoke(
            [
                SystemMessage(
                    content="You are an expert travel flight planner."
                ),
                HumanMessage(content=prompt)
            ]
        )

        flight_data = response.content

    except Exception as e:

        flight_data = (
            f"Flight information unavailable: {str(e)}"
        )

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(
                content="Flight recommendations generated"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# HOTEL AGENT
# =========================

def hotel_agent(state: TravelState) -> dict:

    query = f"Best hotels for {state['user_query']}"

    hotel_results = asyncio.run(
        tavily_mcp_search(query)
    )

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(
                content="Hotel information fetched"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# WEATHER AGENT
# =========================

def weather_agent(state: TravelState):

    city = extract_destination(
        state["user_query"]
    )

    weather_data = asyncio.run(
        weather_mcp_search(city)
    )

    forecast_data = asyncio.run(
        forecast_mcp_search(city)
    )

    return {
        "weather_results": f"""
Current Weather:
{weather_data}

Forecast:
{forecast_data}
""",
        "messages": [
            AIMessage(
                content="Weather information fetched"
            )
        ]
    }


# =========================
# BUDGET AGENT
# =========================

def budget_agent(state: TravelState):

    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Return:

1. Estimated cost categories
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility

If exact live prices are unavailable, clearly label estimates as approximate.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content="You are a practical travel budget analyst."
            ),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "budget_results": response.content,
        "messages": [
            AIMessage(
                content="Budget assessment generated."
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# ITINERARY AGENT
# =========================

def itinerary_agent(state: TravelState):

    prompt = f"""
Create a complete travel itinerary.

User Query:
{state["user_query"]}

Flight Results:
{state.get("flight_results", "")}

Hotel Results:
{state.get("hotel_results", "")}

Weather Results:
{state.get("weather_results", "")}

Budget Results:
{state.get("budget_results", "")}

Make the itinerary practical, budget-aware, and easy to follow.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content="You are an expert travel planner"
            ),
            HumanMessage(content=prompt)
        ]
    )

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# HUMAN APPROVAL AGENT
# =========================

def human_approval_agent(state: TravelState):

    # Do not wrap interrupt() in try/except.
    # LangGraph uses it to pause execution.

    print(
        "\n========== HUMAN APPROVAL NODE REACHED =========="
    )

    print(
        "Itinerary generated:",
        bool(state.get("itinerary"))
    )

    review = interrupt(
        {
            "question": "Do you approve this itinerary?",

            "draft_itinerary": state.get(
                "itinerary",
                ""
            ),

            "approval_request": state.get(
                "approval_request",
                ""
            ),

            "selected_agents": state.get(
                "selected_agents",
                []
            ),

            "supervisor_reasoning": state.get(
                "supervisor_reasoning",
                ""
            ),

            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(
        review.get("approved", False)
    )

    human_feedback = str(
        review.get("feedback", "")
    ).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [
            AIMessage(
                content="Human approval step completed."
            )
        ],
    }


# =========================
# FINAL AGENT
# =========================

def final_agent(state: TravelState):

    if state.get("approved", False):

        review_instruction = (
            "The user approved the draft. "
            "Preserve its decisions while polishing it."
        )

    else:

        review_instruction = f"""
The user requested a revision. Apply this feedback carefully:

{state.get('human_feedback', '') or 'Improve the draft before finalizing it.'}
"""

    selected_agents = state.get("selected_agents", [])

    selected_sections = []

    if "flight_agent" in selected_agents:
        selected_sections.append("Flight Information")

    if "hotel_agent" in selected_agents:
        selected_sections.append("Hotel Suggestions")

    if "weather_agent" in selected_agents:
        selected_sections.append("Weather Information")

    if "budget_agent" in selected_agents:
        selected_sections.append("Estimated Budget")

    if "itinerary_agent" in selected_agents:
        selected_sections.append("Day-by-Day Itinerary")

    sections_text = "\n".join(
        f"{index}. {section}"
        for index, section in enumerate(selected_sections, start=1)
    )

    final_prompt = f"""
Generate the final travel response for the user.

Human Review:
{review_instruction}

User Request:
{state['user_query']}

Selected Agents:
{selected_agents}

Flights:
{state.get('flight_results', '')}

Hotels:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Budget Results:
{state.get('budget_results', '')}

Itinerary:
{state.get('itinerary', '')}

Only include information related to the agents selected by the supervisor.

The final response should contain these sections:

{sections_text}

Important:

- Do NOT create sections for agents that were not selected.
- Do NOT invent flight, hotel, weather, budget, or itinerary information.
- Use only the relevant results provided above.
- Be clear and practical.
- Mention that live flight APIs may not provide ticket prices when pricing
  is unavailable.
- Include weather-based travel advice when weather_agent was selected.
- Keep the response useful for real travel planning.
- Incorporate the human feedback when revision was requested.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content="You are a professional AI travel booking assistant."
            ),
            HumanMessage(content=final_prompt)
        ]
    )

    return {
        "final_response": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# =========================
# ROUTING
# =========================

ROUTE_MAP = {

    "guardrail_blocked":
        "guardrail_blocked",

    "flight_agent":
        "flight_agent",

    "hotel_agent":
        "hotel_agent",

    "weather_agent":
        "weather_agent",

    "budget_agent":
        "budget_agent",

    "itinerary_agent":
        "itinerary_agent",
        
    "final_agent": "final_agent",
}


def _selected_agents(state: TravelState) -> list[str]:

    selected = state.get(
        "selected_agents",
        []
    )

    return [
        agent
        for agent in AGENT_ORDER
        if agent in selected
    ]


def route_from_supervisor(state: TravelState) -> str:

    if not state.get(
        "guardrail_allowed",
        True
    ):
        return "guardrail_blocked"

    selected = _selected_agents(state)

    return (
        selected[0]
        if selected
        else "itinerary_agent"
    )


def route_after_agent(current_agent: str):

    def route(state: TravelState) -> str:

        selected = _selected_agents(state)

        current_index = AGENT_ORDER.index(
            current_agent
        )

        for next_agent in AGENT_ORDER[
            current_index + 1:
        ]:

            if next_agent in selected:
                return next_agent

        return "final_agent"

    return route


# =========================
# BUILD GRAPH
# =========================

graph = StateGraph(TravelState)


graph.add_node(
    "flight_agent",
    flight_agent
)

graph.add_node(
    "hotel_agent",
    hotel_agent
)

graph.add_node(
    "weather_agent",
    weather_agent
)

graph.add_node(
    "itinerary_agent",
    itinerary_agent
)

graph.add_node(
    "budget_agent",
    budget_agent
)

graph.add_node(
    "final_agent",
    final_agent
)

graph.add_node(
    "supervisor",
    supervisor_agent
)

graph.add_node(
    "guardrail_blocked",
    guardrail_blocked_agent
)

graph.add_node(
    "human_approval",
    human_approval_agent
)


# =========================
# START -> SUPERVISOR
# =========================

graph.add_edge(
    START,
    "supervisor"
)


# =========================
# SUPERVISOR ROUTING
# =========================

graph.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    ROUTE_MAP
)


# =========================
# AGENT ROUTING
# =========================

graph.add_conditional_edges(
    "flight_agent",
    route_after_agent("flight_agent"),
    ROUTE_MAP
)


graph.add_conditional_edges(
    "hotel_agent",
    route_after_agent("hotel_agent"),
    ROUTE_MAP
)


graph.add_conditional_edges(
    "weather_agent",
    route_after_agent("weather_agent"),
    ROUTE_MAP
)


graph.add_conditional_edges(
    "budget_agent",
    route_after_agent("budget_agent"),
    ROUTE_MAP
)


# =========================
# ITINERARY -> HUMAN APPROVAL
# =========================

graph.add_edge(
    "itinerary_agent",
    "human_approval"
)


# =========================
# HUMAN APPROVAL -> FINAL
# =========================

graph.add_edge(
    "human_approval",
    "final_agent"
)


# =========================
# FINAL -> END
# =========================

graph.add_edge(
    "final_agent",
    END
)


# =========================
# GUARDRAIL -> END
# =========================

graph.add_edge(
    "guardrail_blocked",
    END
)


# =========================
# POSTGRESQL CHECKPOINTER
# =========================

DATABASE_URL = get_database_url()


_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)


checkpointer = PostgresSaver(
    _conn
)


checkpointer.setup()


travel_graph = graph.compile(
    checkpointer=checkpointer
)


# =========================
# INTERRUPT HELPER
# =========================

def _interrupt_payload(
    result: dict[str, Any]
) -> dict[str, Any] | None:

    interrupts = result.get(
        "__interrupt__",
        []
    )

    if not interrupts:
        return None

    first_interrupt = interrupts[0]

    payload = getattr(
        first_interrupt,
        "value",
        first_interrupt
    )

    return (
        payload
        if isinstance(payload, dict)
        else {"value": payload}
    )


# =========================
# SERIALIZER
# =========================

def _serialize_result(
    result: dict[str, Any],
    thread_id: str
) -> dict[str, Any]:

    messages = result.get(
        "messages",
        []
    )

    last_message = (
        messages[-1].content
        if messages
        else ""
    )

    answer = (
        result.get("final_response")
        or last_message
    )

    interrupt_payload = _interrupt_payload(
        result
    )

    print(
        "\n========== SERIALIZER DEBUG =========="
    )

    print(
        "INTERRUPT PAYLOAD:",
        interrupt_payload
    )

    print(
        "REQUIRES APPROVAL:",
        interrupt_payload is not None
    )

    if interrupt_payload:

        answer = (
            interrupt_payload.get(
                "draft_itinerary"
            )
            or result.get(
                "itinerary",
                ""
            )
        )

    return {

        "thread_id":
            thread_id,

        "answer":
            answer,

        "requires_approval":
            interrupt_payload is not None,

        "approval_request": (
            interrupt_payload.get(
                "approval_request",
                ""
            )
            if interrupt_payload
            else result.get(
                "approval_request",
                ""
            )
        ),

        "flight_results":
            result.get(
                "flight_results",
                ""
            ),

        "hotel_results":
            result.get(
                "hotel_results",
                ""
            ),

        "weather_results":
            result.get(
                "weather_results",
                ""
            ),

        "budget_results":
            result.get(
                "budget_results",
                ""
            ),

        "itinerary":
            result.get(
                "itinerary",
                ""
            ),

        "approved":
            result.get(
                "approved",
                False
            ),

        "human_feedback":
            result.get(
                "human_feedback",
                ""
            ),

        "final_response":
            result.get(
                "final_response",
                ""
            ),

        "selected_agents":
            result.get(
                "selected_agents",
                []
            ),

        "trip_constraints":
            result.get(
                "trip_constraints",
                {}
            ),

        "supervisor_reasoning":
            result.get(
                "supervisor_reasoning",
                ""
            ),

        "llm_calls":
            result.get(
                "llm_calls",
                0
            ),
    }


# =========================
# FUNCTION FOR FASTAPI
# =========================

def run_travel_agent(
    user_input: str,
    thread_id: str | None = None
):

    if not thread_id:

        thread_id = (
            f"user_{uuid.uuid4().hex}"
        )

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=user_input
                )
            ],

            "user_query":
                user_input,

            # Supervisor + guardrail state

            "guardrail_allowed":
                True,

            "guardrail_reason":
                "",

            "selected_agents":
                [],

            "trip_constraints":
                _empty_constraints(),

            "supervisor_reasoning":
                "",

            # Agent results

            "flight_results":
                "",

            "hotel_results":
                "",

            "weather_results":
                "",

            "budget_results":
                "",

            "itinerary":
                "",

            # HITL state

            "approval_request":
                "",

            "approved":
                False,

            "human_feedback":
                "",

            "final_response":
                "",

            "llm_calls":
                0
        },

        config=config
    )

    return _serialize_result(
        result,
        thread_id
    )


# =========================
# RESUME TRAVEL AGENT
# =========================

def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = ""
):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        Command(
            resume={
                "approved":
                    approved,

                "feedback":
                    feedback,
            }
        ),
        config=config,
    )

    return _serialize_result(
        result,
        thread_id
    )