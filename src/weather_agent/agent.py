import uuid
from typing import Any, Final, NotRequired

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

_SYSTEM_PROMPT = (
    "You are a helpful weather assistant. You are only allowed to answer questions "
    "about the CURRENT temperature in cities or any operations on those current "
    "temperatures (e.g. comparing, ranking, or doing arithmetic on them). You are "
    "not allowed to answer questions about past or forecasted/future temperatures, "
    "or questions related to any other topic. When faced with an unrelated or "
    "non-current-temperature question, address that part and politely decline to "
    "answer it, without providing any information about it. "
    "ALWAYS answer using the same language the query is in."
)

_OUTPUT_CHECK_PROMPT = (
    "You are auditing a weather assistant that may only discuss the CURRENT "
    "temperature of cities (never past or forecasted/future temperatures, and "
    "nothing else). Below is the full conversation so far: the user's "
    "question, any tool calls and tool results, and the assistant's final "
    "reply as the last message. Judge that final reply.\n\n"
    "The reply is ALLOWED to state current city temperatures, compare or rank "
    "them (e.g. warmer/colder), and perform arithmetic derived purely from "
    "those current temperature values (sums, differences, averages, "
    "doubling, etc.). The reply is also ALLOWED to contain generic filler "
    "(e.g. offering further help) or a statement refusing/declining to "
    "answer a non-current-temperature part of the question, including a "
    "decline that mentions that topic by name. Merely mentioning a "
    "non-current-temperature topic while declining to address it is NOT a "
    "violation.\n\n"
    "Any part of the user's question that was routed to the "
    "'reject_non_temperature_query' tool must NEVER be answered, in whole "
    "or in part, anywhere in the final reply — treat every such part as "
    "strictly off-limits no matter how the reply words its refusal. "
    "Carefully re-check the final reply against the rest of the "
    "conversation to make sure no fact, opinion, translation, "
    "recommendation, past/forecasted temperature, or other "
    "non-current-temperature answer belonging to a rejected part of the "
    "question has slipped through.\n\n"
    "Flag the reply as containing non-weather information ONLY if it "
    "actually provides an answer, fact, opinion, translation, unit "
    "conversion unrelated to current temperature, or recommendation about "
    "something other than the current temperature of a city. "
    "Keep in mind that the user can ask and the weather assistant "
    "can answer in different languages!"
)

_INITIAL_PROMPT_TEMPLATE = (
    "Decompose the following user query into one or more sub-queries. Cities "
    "may be named directly or referred to indirectly (e.g. 'Hungary's top 3 "
    "most populated cities', 'western European capitals'). For any indirect "
    "reference, use your own knowledge to resolve it to the specific set of "
    "named cities it refers to before proceeding. Classify the sub-queries "
    "into two categories: (1) sub-queries that require the CURRENT (right "
    "now) temperature of one or more cities (named directly or resolved "
    "from an indirect reference), and (2) sub-queries that do not relate to "
    "the current temperature of any city — this includes questions about "
    "past or forecasted/future temperatures, and any other unrelated topic. "
    "For sub-queries in category (1), call the tool 'get_city_temperature' "
    "for each resolved city to get the current temperatures to answer the "
    "query. For sub-queries in category (2), call the tool "
    "'reject_non_temperature_query'.\n\n"
    "User query: {query}"
)

_REJECTION_MESSAGE: Final[str] = (
    "I can only assist with inquiries about the current temperature in a "
    "city. Please rephrase your question."
)

_TEMPERATURE_TOOL_NAME: Final[str] = "get_city_temperature"


class OutputCheck(BaseModel):
    contains_non_weather_info: bool = Field(
        description="True if the assistant answered non-weather questions."
    )


class AgentState(MessagesState):
    last_tool_messages: NotRequired[list[BaseMessage]]


class WeatherAgent:
    def __init__(self, llm: BaseChatModel, tools: list[BaseTool]) -> None:
        self._tools_by_name: dict[str, BaseTool] = {t.name: t for t in tools}
        self._llm_forced = llm.bind_tools(tools, tool_choice="required")
        self._llm_auto = llm.bind_tools(tools)
        self._llm_output_check: Runnable[LanguageModelInput, OutputCheck] = (
            llm.with_structured_output(OutputCheck)
        )
        self._checkpointer = InMemorySaver()
        self._config: RunnableConfig = {"configurable": {"thread_id": uuid.uuid4().hex}}
        self._graph: CompiledStateGraph[AgentState, None, AgentState, AgentState] = (
            self._build_graph()
        )

    def run(self, query: str, *, verbose: bool = False) -> str:
        initial_prompt = _INITIAL_PROMPT_TEMPLATE.format(query=query)
        initial_state: AgentState = {"messages": [HumanMessage(content=initial_prompt)]}
        final_state: dict[str, Any] = (
            self._run_verbose(initial_state)
            if verbose
            else self._graph.invoke(initial_state, self._config)
        )
        return str(final_state["messages"][-1].content)

    def _run_verbose(self, initial_state: AgentState) -> dict[str, Any]:
        for step in self._graph.stream(
            initial_state, self._config, stream_mode="values"
        ):
            final_state = step
            step["messages"][-1].pretty_print()
        return final_state

    def _classify_query(self, state: AgentState) -> dict[str, list[BaseMessage]]:
        messages = [SystemMessage(content=_SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [self._llm_forced.invoke(messages)]}

    def _compose_answer(self, state: AgentState) -> dict[str, list[BaseMessage]]:
        messages = [SystemMessage(content=_SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [self._llm_auto.invoke(messages)]}

    def _execute_tool_calls(self, state: AgentState) -> dict[str, list[BaseMessage]]:
        last_message = state["messages"][-1]
        tool_calls = (
            last_message.tool_calls if isinstance(last_message, AIMessage) else []
        )

        results: list[BaseMessage] = []
        for tool_call in tool_calls:
            tool_obj = self._tools_by_name[tool_call["name"]]
            observation = tool_obj.invoke(tool_call["args"])
            results.append(
                ToolMessage(
                    content=observation,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": results, "last_tool_messages": list(results)}

    def _was_query_fully_rejected(self, state: AgentState) -> str:
        tool_names = {message.name for message in state["last_tool_messages"]}
        if _TEMPERATURE_TOOL_NAME in tool_names:
            return "compose_answer"
        return "decline_query"

    def _decline_query_node(self, state: AgentState) -> dict[str, list[BaseMessage]]:
        return {"messages": [AIMessage(content=_REJECTION_MESSAGE)]}

    def _audit_answer_node(self, state: AgentState) -> dict[str, list[BaseMessage]]:
        messages = [
            *state["messages"],
            HumanMessage(content=_OUTPUT_CHECK_PROMPT),
        ]
        result = self._llm_output_check.invoke(messages)
        if result.contains_non_weather_info:
            return {"messages": [AIMessage(content=_REJECTION_MESSAGE)]}
        return {"messages": []}

    def _needs_more_tool_calls(self, state: AgentState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "execute_tool_calls"
        return "audit_answer"

    def _build_graph(
        self,
    ) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
        builder = StateGraph(AgentState)
        builder.add_node("classify_query", self._classify_query)
        builder.add_node("execute_tool_calls", self._execute_tool_calls)
        builder.add_node("decline_query", self._decline_query_node)
        builder.add_node("compose_answer", self._compose_answer)
        builder.add_node("audit_answer", self._audit_answer_node)

        builder.add_edge(START, "classify_query")
        builder.add_edge("classify_query", "execute_tool_calls")
        builder.add_conditional_edges(
            "execute_tool_calls",
            self._was_query_fully_rejected,
            ["compose_answer", "decline_query"],
        )
        builder.add_edge("decline_query", END)
        builder.add_conditional_edges(
            "compose_answer",
            self._needs_more_tool_calls,
            ["execute_tool_calls", "audit_answer"],
        )
        builder.add_edge("audit_answer", END)
        return builder.compile(checkpointer=self._checkpointer)
