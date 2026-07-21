from langchain_core.tools import BaseTool, tool

from weather_agent.open_meteo_client import (
    CityNotFoundError,
    OpenMeteoClient,
    OpenMeteoRequestError,
)

REJECTION_MESSAGE_TEMPLATE = (
    "You are not authorized to answer '{query}'. "
    "Politely decline answering the questions, and do not attempt to answer them, "
    "clarifying that you can only help with the CURRENT temperature of a city."
)


def build_tools(client: OpenMeteoClient) -> list[BaseTool]:

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
        except OpenMeteoRequestError:
            return f"Failed to fetch the temperature for {city}"

        return (
            f"The current temperature in {location.name}, {location.country} "
            f"is {temperature.value}{temperature.unit}."
        )

    @tool
    def reject_non_temperature_query(query: str) -> str:
        """Reject a question that is not about a city's current temperature.

        Call this for queries (or sub-queries) that do not relate to
        the current temperature of a city.
        """
        return REJECTION_MESSAGE_TEMPLATE.format(query=query)

    return [get_city_temperature, reject_non_temperature_query]
