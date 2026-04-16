"""Sample tools to demonstrate the agent works end-to-end."""

from datetime import datetime


def get_weather(city: str) -> dict:
    """Retrieves the current weather for a given city.

    Args:
        city: Name of the city (e.g., "Houston", "New York", "London").

    Returns:
        dict with city, temperature, and conditions.
    """
    # Stubbed for demo — replace with real API call
    weather_data = {
        "houston": {"temp_f": 92, "conditions": "Sunny and humid"},
        "new york": {"temp_f": 75, "conditions": "Partly cloudy"},
        "london": {"temp_f": 62, "conditions": "Overcast with light rain"},
        "seattle": {"temp_f": 58, "conditions": "Cloudy"},
        "redmond": {"temp_f": 57, "conditions": "Cloudy with breaks"},
    }

    data = weather_data.get(city.lower(), {"temp_f": 70, "conditions": "Unknown"})
    return {
        "city": city,
        "temperature_f": data["temp_f"],
        "conditions": data["conditions"],
    }


def get_time(timezone: str) -> dict:
    """Gets the current time in a given timezone.

    Args:
        timezone: Timezone name (e.g., "US/Central", "US/Eastern", "UTC").

    Returns:
        dict with timezone and current time.
    """
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    return {
        "timezone": timezone,
        "current_time": now,
        "note": "Stubbed — always returns UTC for demo purposes.",
    }
