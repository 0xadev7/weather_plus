from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    latitude: List[float]
    longitude: List[float]
    hourly: List[str] | str
    start_hour: str
    end_hour: str
    timezone: Optional[str] = None


class HourlyPayload(BaseModel):
    time: List[str]
    # variable arrays will be injected dynamically


class ForecastResponse(BaseModel):
    latitude: List[float]
    longitude: List[float]
    generationtime_ms: float
    utc_offset_seconds: int
    timezone: str
    timezone_abbreviation: str
    hourly_units: Dict[str, str]
    hourly: Dict[str, Any]
