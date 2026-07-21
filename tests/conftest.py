"""Shared pytest fixtures for the weather agent's end-to-end test suite.

These tests exercise the full LangGraph agent flow against the real OpenAI
API and the real Open-Meteo API, so they require a valid ``OPENAI_API_KEY``
and network access. They are marked with the ``e2e`` pytest marker.
"""

import os
from collections.abc import Callable

import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from weather_agent.agent import WeatherAgent
from weather_agent.open_meteo_client import OpenMeteoClient
from weather_agent.tools import build_tools

_DEFAULT_MODEL = "gpt-4o"
_DEFAULT_JUDGE_MODEL = "gpt-5"


@pytest.fixture(scope="session", autouse=True)
def _load_env() -> None:
    load_dotenv()


@pytest.fixture(scope="session")
def openai_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is not set; skipping end-to-end agent tests.")
    return api_key


@pytest.fixture(scope="session")
def llm(openai_api_key: str) -> ChatOpenAI:
    """The real-API chat model powering the agent under test."""
    model = os.environ.get("OPENAI_MODEL", _DEFAULT_MODEL)
    base_url = os.environ.get("OPENAI_BASE_URL")
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(openai_api_key),
        base_url=base_url,
        temperature=0,
    )


@pytest.fixture(scope="session")
def judge_llm(openai_api_key: str) -> ChatOpenAI:
    """A stronger chat model used only to grade the agent's answers.

    Using a separate, more capable model than the agent itself reduces the
    risk of the judge sharing the same blind spots (or being fooled by
    subtly wrong answers) as the model being evaluated.
    """
    model = os.environ.get("OPENAI_JUDGE_MODEL", _DEFAULT_JUDGE_MODEL)
    base_url = os.environ.get("OPENAI_BASE_URL")
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(openai_api_key),
        base_url=base_url,
        temperature=0,
    )


@pytest.fixture
def make_agent(llm: ChatOpenAI) -> Callable[[], WeatherAgent]:
    """Factory building a fresh ``WeatherAgent`` (own thread/checkpointer) per call."""

    def _make() -> WeatherAgent:
        tools = build_tools(OpenMeteoClient())
        return WeatherAgent(llm, tools)

    return _make
