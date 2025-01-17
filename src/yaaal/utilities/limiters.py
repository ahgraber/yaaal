# ruff: NOQA: E731
import asyncio
from collections import deque
from functools import wraps
import logging
from threading import Lock
import time

import requests
from tenacity import RetryCallState, retry, stop_after_attempt

logger = logging.getLogger(__name__)


def wait_429(retry_state: RetryCallState):
    """Tenacity callback to detect anticipated wait time from a 429 too many requests http response.

    ```py
    @retry(
        wait=wait_openai,
        stop=stop_after_attempt(6),
    )
    def fn_with_openai_call(openai_client, prompt: str):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {}"role": "system", "content": "..."},
                    {"role": "user", "content": "..."},
                ],
                logit_bias={"67907": 100, "15921": 100},
                max_tokens=1,
                temperature=0,
            )
            if response.choices:
                return response.choices[0].message.content
            else:
                return "An error occurred: No valid response received"
        except Exception as e:
            return f"An error occurred: {str(e)}
    ```
    """
    ex = retry_state.outcome.exception()
    if isinstance(ex, requests.exceptions.HTTPError) and ex.response.status_code == 429:
        retry_after = ex.response.headers.get("Retry-After") or 4
        try:
            return int(retry_after) + 1
        except (TypeError, ValueError):
            pass
    return 0


class TimeWindowRateLimiter:
    """Rate limiter that allows a maximum number of actions over sliding time window (seconds).

    If the maximum number of actions occurs in the time window, the rate limiter will block until a slot becomes available.
    This is not an async limiter; it assumes that actions are synchronous and blocking.
    Therefore, if an action takes a long time to complete, it will block subsequent actions from starting until it completes _even if it exceeds the window period_.
    """

    def __init__(self, max_actions: int, window_seconds: int, min_interval: float = 0.1):
        if max_actions <= 0:
            raise ValueError("max_actions must be > 0")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if min_interval < 0:
            raise ValueError("min_interval must be >= 0")

        self.max_actions = max_actions
        self.window_seconds = window_seconds
        self.min_interval = min_interval
        self.start_times = deque()
        self.n_active = 0
        # self.last = time.monotonic()

        self.lock = Lock()

    def _update(self):
        """Remove actions should no longer be blocking the queue.

        An action may take longer to execute than the time period we track.
        This action counts against the limit until its start time exits the window.
        Once its start time exits the window, its slot becomes available for new operations, while the action itself continues running to completion.
        """
        window_start = time.monotonic() - self.window_seconds

        while self.start_times and self.start_times[0] < window_start:
            self.start_times.popleft()
            if self.n_active > 0:
                logging.debug("Sliding window freed a slot")
                self.n_active -= 1

    def _wait(self):
        """Wait until a slot is available."""
        while True:
            with self.lock:
                self._update()
                if len(self.start_times) < self.max_actions:
                    self.start_times.append(time.monotonic())
                    self.n_active += 1
                    return

                if self.start_times:
                    wait_time = max(
                        self.min_interval,
                        self.start_times[0] + self.window_seconds - time.monotonic(),
                    )
                else:
                    wait_time = self.min_interval

            # Release lock during sleep
            if wait_time > 0:
                logging.debug(f"Waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)

    def _complete(self):
        """Remove completed action from active count."""
        logging.debug("Completing action")
        with self.lock:
            if self.n_active > 0:
                self.n_active -= 1

    def __call__(self, func):
        """Apply rate limiting to a function as a decorator."""

        # TODO: add support for async functions?
        @wraps(func)
        def wrapped(*args, **kwargs):
            self._wait()
            try:
                return func(*args, **kwargs)
            finally:
                self._complete()

        return wrapped


class AsyncTimeWindowRateLimiter:
    """Async rate limiter that allows a maximum number of actions over sliding time window (seconds).

    If the maximum number of actions occurs in the time window, the rate limiter will block until a slot becomes available.
    """

    def __init__(self, max_actions: int, window_seconds: int, min_interval: float = 0.1):
        if max_actions <= 0:
            raise ValueError("max_actions must be > 0")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if min_interval < 0:
            raise ValueError("min_interval must be >= 0")

        self.max_actions = max_actions
        self.window_seconds = window_seconds
        self.min_interval = min_interval
        self.start_times = deque()
        self.n_active = 0
        # self.last = time.monotonic()

        self.lock = asyncio.Lock()

    def _update(self):
        """Remove actions should no longer be blocking the queue.

        An action may take longer to execute than the time period we track.
        This action counts against the limit until its start time exits the window.
        Once its start time exits the window, its slot becomes available for new operations, while the action itself continues running to completion.
        """
        window_start = time.monotonic() - self.window_seconds

        while self.start_times and self.start_times[0] < window_start:
            self.start_times.popleft()
            if self.n_active > 0:
                logging.debug("Sliding window freed a slot")
                self.n_active -= 1

    async def _wait(self):
        """Wait until a slot is available."""
        while True:
            async with self.lock:
                self._update()
                if len(self.start_times) < self.max_actions:
                    self.start_times.append(time.monotonic())
                    self.n_active += 1
                    return

                if self.start_times:
                    wait_time = max(
                        self.min_interval,
                        self.start_times[0] + self.window_seconds - time.monotonic(),
                    )
                else:
                    wait_time = self.min_interval

            # Release lock during sleep
            if wait_time > 0:
                logging.debug(f"Waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)

    async def _complete(self):
        """Remove completed action from active count."""
        logging.debug("Completing action")
        async with self.lock:
            if self.n_active > 0:
                self.n_active -= 1

    def __call__(self, func):
        """Apply rate limiting to a function as a decorator."""
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapped(*args, **kwargs):
                await self._wait()
                try:
                    return await func(*args, **kwargs)
                finally:
                    await self._complete()

            return async_wrapped
        else:
            raise TypeError("Only async functions can be decorated with async rate limiter")
