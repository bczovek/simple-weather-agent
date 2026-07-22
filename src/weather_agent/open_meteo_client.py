from dataclasses import dataclass
from typing import Any, Final, cast

import requests

_GEOCODING_URL: Final[str] = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL: Final[str] = "https://api.open-meteo.com/v1/forecast"
_REQUEST_TIMEOUT_SECONDS: Final[float] = 10.0


class CityNotFoundError(Exception):

    def __init__(self, city: str) -> None:
        super().__init__(f"No location found for city: {city!r}")
        self.city: str = city


class OpenMeteoRequestError(Exception):

    def __init__(self, url: str, cause: Exception) -> None:
        super().__init__(f"Request to {url} failed: {cause}")
        self.url: str = url


@dataclass(frozen=True)
class CityLocation:

    name: str
    country: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class CurrentTemperature:

    value: float
    unit: str


class OpenMeteoClient:

    def geocode(self, city: str) -> CityLocation:
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
