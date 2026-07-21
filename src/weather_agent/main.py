"""CLI entry point for the weather agent.

Starts an interactive prompt that reads questions from stdin and answers
only those about the current temperature of a named city, until the user
types "exit" (or stdin is closed).

Usage:
    python -m weather_agent.main
    python -m weather_agent.main --verbose
"""

import argparse
import logging
import os
import sys
from typing import Final

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from weather_agent.agent import WeatherAgent
from weather_agent.open_meteo_client import OpenMeteoClient
from weather_agent.tools import build_tools

DEFAULT_MODEL: Final[str] = "gpt-4o"

_logger: logging.Logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


def _parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        description="Answer city current-temperature questions using an LLM agent.",
    )
    arg_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print each reasoning/tool-calling step of the agent's run.",
    )
    return arg_parser.parse_args()


def _build_llm() -> ChatOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    base_url = os.environ.get("OPENAI_BASE_URL")
    return ChatOpenAI(
        model=model, api_key=SecretStr(api_key), base_url=base_url, temperature=0
    )


def _print_answer(answer: str) -> None:
    print("=" * 100)
    print(answer)
    print("=" * 100)


def _run_query_loop(agent: WeatherAgent, *, verbose: bool) -> None:
    """Read questions from stdin and print answers until the user types
    "exit" (or stdin is closed)."""
    while True:
        try:
            print(
                "Enter a question about a city's current temperature, "
                "or type 'exit' to quit."
            )
            query = input("> ").strip()
        except EOFError:
            break

        if not query:
            continue
        if query.lower() == "exit":
            break

        answer = agent.run(query, verbose=verbose)
        if not verbose:
            _print_answer(answer)


def main() -> None:
    _configure_logging()
    args = _parse_args()
    load_dotenv()

    llm = _build_llm()
    client = OpenMeteoClient()
    tools = build_tools(client)
    agent = WeatherAgent(llm, tools)

    _run_query_loop(agent, verbose=args.verbose)


if __name__ == "__main__":
    main()
