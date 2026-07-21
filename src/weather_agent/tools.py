"""LangChain tool definitions for the weather agent.

Tools are built via ``build_tools`` rather than declared as bare
module-level ``@tool`` functions, so the ``OpenMeteoClient`` dependency is
injected at construction time instead of being hard-coded inside the tool
bodies.
"""

from langchain_core.tools import BaseTool, tool

from weather_agent.open_meteo_client import (
    CityNotFoundError,
    OpenMeteoClient,
    OpenMeteoRequestError,
)

REJECTION_MESSAGE_TEMPLATE = (
    "You are not authorized to answer questions like {query}. "
    "Only questions related to temperatures in any city"
)


def build_tools(client: OpenMeteoClient) -> list[BaseTool]:
    """Build the two tools available to the weather agent.

    Args:
        client: the Open-Meteo client used to resolve city temperatures.
    """

    @tool
    def get_city_temperature(city: str) -> str:
        """Get the current temperature for a single named city.

        Call this once per city mentioned in the user's question.
        """
        try:
            location = client.geocode(city)
            temperature = client.get_current_temperature(location)
        except CityNotFoundError:
            return f"Could not find a location for city: {city}"
        except OpenMeteoRequestError as exc:
            return f"Failed to fetch the temperature for {city}: {exc}"

        return (
            f"The current temperature in {location.name}, {location.country} "
            f"is {temperature.value}{temperature.unit}."
        )

    @tool
    def reject_non_temperature_query(query: str) -> str:
        """Reject a question that is not about a city's current temperature.

        Call this for questions (or parts of questions) that do not ask
        about the current temperature of a named city.
        """
        return REJECTION_MESSAGE_TEMPLATE.format(query=query)

    return [get_city_temperature, reject_non_temperature_query]
