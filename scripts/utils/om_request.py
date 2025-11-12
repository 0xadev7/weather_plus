import time
import os
import requests, random

from rate_limiter import MultiWindowRateLimiter

# Create a single global limiter instance (module-level)
OM_LIMITER = MultiWindowRateLimiter(
    per_minute=int(os.getenv("OM_RATE_PER_MIN", "600")),
    per_hour=int(os.getenv("OM_RATE_PER_HOUR", "5000")),
    per_day=int(os.getenv("OM_RATE_PER_DAY", "10000")),
    per_month=int(os.getenv("OM_RATE_PER_MONTH", "300000")),
)


def om_request(url, params, timeout=60):
    """
    One safe Open-Meteo API call with multi-window limiting and polite retries.
    """
    backoff = 1.0
    resp = None
    for attempt in range(8):
        OM_LIMITER.acquire()  # <-- enforce 600/min, 5k/hr, 10k/day, 300k/month

        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code < 500 and resp.status_code != 429:
            # Success or client error (not retryable)
            return resp

        # 429 or 5xx → exponential backoff with small jitter, still protected by limiter
        sleep_s = backoff + random.uniform(0, 0.25)
        time.sleep(sleep_s)
        backoff = min(backoff * 2.0, 60.0)

    # Final attempt without raising; return last response if present, else raise
    if resp is not None:
        resp.raise_for_status()

    return resp
