import os

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
MODEL_DIR = os.getenv(
    "MODEL_DIR", os.path.join(os.path.dirname(__file__), "../../models")
)
OPEN_METEO_URL = os.getenv("OPEN_METEO_URL", "https://api.open-meteo.com/v1/forecast")

# variables we support (Open-Meteo names to stay wire-compatible)
SUPPORTED_HOURLY = [
    "temperature_2m",
    "precipitation",
    "wind_speed_100m",
    "wind_direction_100m",
    "dew_point_2m",
    "surface_pressure",
]
