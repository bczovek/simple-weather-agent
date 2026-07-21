"""Client for the public Open-Meteo geocoding and forecast APIs.

See https://open-meteo.com/en/docs and
https://open-meteo.com/en/docs/geocoding-api for the underlying HTTP APIs.
"""

from dataclasses import dataclass
from typing import Any, Final, cast

import requests

_GEOCODING_URL: Final[str] = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL: Final[str] = "https://api.open-meteo.com/v1/forecast"
_REQUEST_TIMEOUT_SECONDS: Final[float] = 10.0


class CityNotFoundError(Exception):
    """Raised when the Open-Meteo geocoding API finds no match for a city."""

    def __init__(self, city: str) -> None:
        super().__init__(f"No location found for city: {city!r}")
        self.city: str = city


class OpenMeteoRequestError(Exception):
    """Raised when a request to an Open-Meteo API endpoint fails."""

    def __init__(self, url: str, cause: Exception) -> None:
        super().__init__(f"Request to {url} failed: {cause}")
        self.url: str = url


@dataclass(frozen=True)
class CityLocation:
    """A geocoded city location resolved from the Open-Meteo geocoding API."""

    name: str
    country: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class CurrentTemperature:
    """A current temperature reading resolved from the Open-Meteo forecast API."""

    value: float
    unit: str


class OpenMeteoClient:
    """Thin HTTP client wrapping the Open-Meteo geocoding and forecast APIs."""

    def geocode(self, city: str) -> CityLocation:
        """Resolve a city name to a single best-matching location.

        Raises:
            CityNotFoundError: if no location matches ``city``.
            OpenMeteoRequestError: if the geocoding request fails.
        """
        # Third-party JSON response: shape isn't statically known, so Any is
        # unavoidable for the raw payload values here.
        params: dict[str, Any] = {
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json",
        }
        payload = self._get(_GEOCODING_URL, params)

        results = payload.get("results")
        if not results:
            raise CityNotFoundError(city)

        first_result = results[0]
        return CityLocation(
            name=first_result["name"],
            country=first_result.get("country", ""),
            latitude=first_result["latitude"],
            longitude=first_result["longitude"],
        )

    def get_current_temperature(self, location: CityLocation) -> CurrentTemperature:
        """Fetch the current 2m air temperature for a geocoded location.

        Raises:
            OpenMeteoRequestError: if the forecast request fails.
        """
        # Third-party JSON response: shape isn't statically known, so Any is
        # unavoidable for the raw payload values here.
        params: dict[str, Any] = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "current": "temperature_2m",
        }
        payload = self._get(_FORECAST_URL, params)

        current = payload["current"]
        units = payload["current_units"]
        return CurrentTemperature(
            value=current["temperature_2m"], unit=units["temperature_2m"]
        )

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            response = requests.get(
                url, params=params, timeout=_REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            return cast(dict[str, Any], response.json())
        except requests.RequestException as exc:
            raise OpenMeteoRequestError(url, exc) from exc
