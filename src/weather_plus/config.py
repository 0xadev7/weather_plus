import os

MODEL_DIR = os.getenv("MODEL_DIR", "models")
OPEN_METEO_URL = os.getenv("OPEN_METEO_URL", "https://api.open-meteo.com/v1/forecast")
USE_BASELINE = int(os.getenv("USE_BASELINE", "1"))
OM_TIMEOUT = int(os.getenv("OM_TIMEOUT", "120"))
BREAKDOWN_DEG = int(os.getenv("BREAKDOWN_DEG", "30"))

SUPPORTED = {
    "temperature_2m": "temperature_2m",
    "dew_point_2m": "dew_point_2m",
    "surface_pressure": "surface_pressure",
    "precipitation": "precipitation",
    "wind_speed_100m": "wind_speed_100m",
    "wind_direction_100m": "wind_direction_100m",
}
BASELINE_NEEDED = ",".join(SUPPORTED.keys())
