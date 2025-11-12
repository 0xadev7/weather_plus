# ---- Multi-window rate limiter (minute/hour/day/month) ----
import time, threading
from collections import deque
from dataclasses import dataclass


@dataclass
class Window:
    secs: int
    limit: int
    q: deque


class MultiWindowRateLimiter:
    """
    Sliding-window limiter across multiple windows.
    Keeps timestamps of recent calls and sleeps just enough so the next call is safe.
    Thread-safe for single-process concurrency.
    """

    def __init__(
        self,
        per_minute: int = 600,
        per_hour: int = 5000,
        per_day: int = 10000,
        per_month: int = 300000,
    ):
        self.windows = [
            Window(60, per_minute, deque()),
            Window(3600, per_hour, deque()),
            Window(86400, per_day, deque()),
        ]
        # Month handled as calendar month (resets on month change)
        self.month_limit = per_month
        self.month_key = time.strftime("%Y-%m")  # e.g., "2025-11"
        self.month_count = 0
        self._lock = threading.Lock()

    def _roll_month_if_needed(self):
        mk = time.strftime("%Y-%m")
        if mk != self.month_key:
            self.month_key = mk
            self.month_count = 0

    def _prune(self, now):
        for w in self.windows:
            cutoff = now - w.secs
            while w.q and w.q[0] <= cutoff:
                w.q.popleft()

    def acquire(self):
        """
        Block until making one more call will not exceed any window.
        """
        while True:
            with self._lock:
                now = time.time()
                self._roll_month_if_needed()
                self._prune(now)

                # compute deficits
                sleeps = []
                for w in self.windows:
                    if len(w.q) >= w.limit:
                        # Next allowable time is when the oldest timestamp exits the window
                        oldest = w.q[0]
                        sleeps.append((oldest + w.secs) - now)

                if self.month_count >= self.month_limit:
                    # Sleep until next month boundary
                    # rough: sleep to first day of next month UTC
                    t = time.gmtime(now)
                    # compute first day next month 00:00:00
                    year, mon = t.tm_year, t.tm_mon
                    if mon == 12:
                        year, mon = year + 1, 1
                    else:
                        mon += 1
                    # seconds to next month
                    import calendar, datetime as dt

                    next_month = dt.datetime(
                        year, mon, 1, 0, 0, 0, tzinfo=dt.timezone.utc
                    )
                    sleeps.append(
                        (
                            next_month - dt.datetime.fromtimestamp(now, dt.timezone.utc)
                        ).total_seconds()
                    )

                if not sleeps:
                    # record the call
                    for w in self.windows:
                        w.q.append(now)
                    self.month_count += 1
                    return  # safe to proceed

                # sleep for the minimum required time + tiny jitter
                to_sleep = max(0.0, min(sleeps)) + 0.01
            time.sleep(to_sleep)
