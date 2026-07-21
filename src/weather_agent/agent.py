"""The LangGraph agentic loop for the weather agent.

Graph shape:

    START -> llm_call_router -> tool_node -> llm_call_synthesize -> should_continue -+
                                                   ^                                 |
                                                   +---------------------------------+
                                                                                      v
                                                                                     END

``llm_call_router`` is bound with ``tool_choice="required"`` so the model must
always call ``get_city_temperature`` and/or ``reject_non_temperature_query``
instead of answering directly. Every tool batch (whether it was a pure
rejection, a pure temperature lookup, or a composite question mixing both)
always flows through ``llm_call_synthesize``, which is bound without a forced
tool choice so it can compose a final natural-language answer (and may still
call a tool again for another city) — there is deliberately no short-circuit
around it, so the model is trusted to relay the reject tool's message
faithfully instead of the code enforcing verbatim wording structurally.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

_SYSTEM_PROMPT = (
    "You are a weather assistant. You can only answer questions about the "
    "current temperature of a named city. For every city mentioned in the "
    "user's question, call get_city_temperature once. For any other part "
    "of the question that is not about a city's current temperature, call "
    "reject_non_temperature_query with that part of the question as the "
    "query argument."
)


class WeatherAgent:
    """A minimal LangGraph agent answering city-temperature questions."""

    def __init__(self, llm: BaseChatModel, tools: list[BaseTool]) -> None:
        self._tools_by_name: dict[str, BaseTool] = {t.name: t for t in tools}
        self._llm_forced = llm.bind_tools(tools, tool_choice="required")
        self._llm_auto = llm.bind_tools(tools)
        self._graph: CompiledStateGraph[
            MessagesState, None, MessagesState, MessagesState
        ] = self._build_graph()

    def run(self, query: str, *, verbose: bool = False) -> str:
        """Answer a single question, returning the final answer text."""
        initial_state: MessagesState = {"messages": [HumanMessage(content=query)]}
        final_state: dict[str, Any] = (
            self._run_verbose(initial_state)
            if verbose
            else self._graph.invoke(initial_state)
        )
        return str(final_state["messages"][-1].content)

    def _run_verbose(self, initial_state: MessagesState) -> dict[str, Any]:
        final_state: dict[str, Any] = dict(initial_state)
        for step in self._graph.stream(initial_state, stream_mode="values"):
            final_state = step
            step["messages"][-1].pretty_print()
        return final_state

    def _llm_call_router(self, state: MessagesState) -> dict[str, list[BaseMessage]]:
        messages = [SystemMessage(content=_SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [self._llm_forced.invoke(messages)]}

    def _llm_call_synthesize(
        self, state: MessagesState
    ) -> dict[str, list[BaseMessage]]:
        messages = [SystemMessage(content=_SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [self._llm_auto.invoke(messages)]}

    def _tool_node(self, state: MessagesState) -> dict[str, list[BaseMessage]]:
        last_message = state["messages"][-1]
        tool_calls = (
            last_message.tool_calls if isinstance(last_message, AIMessage) else []
        )

        results: list[BaseMessage] = []
        for tool_call in tool_calls:
            tool_obj = self._tools_by_name[tool_call["name"]]
            observation = tool_obj.invoke(tool_call["args"])
            results.append(
                ToolMessage(content=observation, tool_call_id=tool_call["id"])
            )

        return {"messages": results}

    def _should_continue(self, state: MessagesState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tool_node"
        return END

    def _build_graph(
        self,
    ) -> CompiledStateGraph[MessagesState, None, MessagesState, MessagesState]:
        builder = StateGraph(MessagesState)
        builder.add_node("llm_call_router", self._llm_call_router)
        builder.add_node("tool_node", self._tool_node)
        builder.add_node("llm_call_synthesize", self._llm_call_synthesize)

        builder.add_edge(START, "llm_call_router")
        builder.add_edge("llm_call_router", "tool_node")
        builder.add_edge("tool_node", "llm_call_synthesize")
        builder.add_conditional_edges(
            "llm_call_synthesize", self._should_continue, ["tool_node", END]
        )
        return builder.compile()
