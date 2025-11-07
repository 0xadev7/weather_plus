"""
Tiny client that mimics the interface your miner expects from openmeteo_requests.Client.
"""

from __future__ import annotations
import requests
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class _HourlyVar:
    arr: np.ndarray

    def ValuesAsNumpy(self):
        return self.arr


@dataclass
class _Hourly:
    vars: List[_HourlyVar]

    def Variables(self, i: int):
        return self.vars[i]

    def VariablesLength(self):
        return len(self.vars)


@dataclass
class _Response:
    _hourly: _Hourly

    def Hourly(self):
        return self._hourly


class Client:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def weather_api(
        self, url: str | None, params: Dict[str, Any], method: str = "POST"
    ) -> List[_Response]:
        # ignore provided url; use our base_url for compatibility
        endpoint = f"{self.base_url}/v1/forecast"
        r = requests.post(endpoint, json=params, timeout=60)
        r.raise_for_status()
        data = r.json()

        time_list = data["hourly"]["time"]
        T = len(time_list)

        # OM compat: pack requested hourly variables in original order
        hourly_req = params["hourly"]
        if isinstance(hourly_req, str):
            hourly_req = [hourly_req]

        # build per-grid response (Open-Meteo returns per-location)
        # our API returns grid-major arrays; we'll map to per-location objects
        lat = params["latitude"]
        lon = params["longitude"]
        grid = [(a, b) for a in lat for b in lon]
        G = len(grid)

        responses = []
        for g in range(G):
            vars_for_g = []
            for name in hourly_req:
                arr = np.array(data["hourly"][name])[:, g]
                vars_for_g.append(_HourlyVar(arr=arr))
            responses.append(_Response(_Hourly(vars=vars_for_g)))
        return responses
