import os
import time

import pytest

from yaaal.utilities.limiters import AsyncTimeWindowRateLimiter, TimeWindowRateLimiter


class TestTimeLimitRateLimiter:
    def test_init_invalid_parameters(self):
        with pytest.raises(ValueError, match="max_actions must be > 0"):
            TimeWindowRateLimiter(max_actions=0, window_seconds=10)

        with pytest.raises(ValueError, match="max_actions must be > 0"):
            TimeWindowRateLimiter(max_actions=-1, window_seconds=10)

        with pytest.raises(ValueError, match="window_seconds must be > 0"):
            TimeWindowRateLimiter(max_actions=5, window_seconds=0)

        with pytest.raises(ValueError, match="min_interval must be >= 0"):
            TimeWindowRateLimiter(max_actions=5, window_seconds=10, min_interval=-0.1)

    def test_instant_function(self):
        """'Instant' function will end up rate limited due to max actions."""
        limiter = TimeWindowRateLimiter(max_actions=5, window_seconds=5)

        @limiter
        def testfunc():
            return 1

        start = time.monotonic()
        result = [testfunc() for _ in range(10)]
        end = time.monotonic()

        assert sum(result) == 10
        # Limiter allows 5 actions every 5 seconds.
        # 5 actions in the first 5-second-window, then the remaining 5 almost immediately after.
        assert end - start > 5

    def test_function_with_nonblocking_runtime(self):
        """If function runtime * window_actions > window_seconds, then no blocking occurs."""
        limiter = TimeWindowRateLimiter(max_actions=5, window_seconds=5)

        @limiter
        def testfunc():
            time.sleep(2)
            return 1

        start = time.monotonic()
        result = [testfunc() for _ in range(10)]
        end = time.monotonic()

        assert sum(result) == 10
        # Limiter allows 5 actions every 5 seconds.
        # Each action takes 2 seconds, so each window will only hold 2.5 actions
        # The limiter is not the blocker, the function runtime is.
        assert end - start >= 20


@pytest.mark.asyncio(loop_scope="function")
class TestAsyncTimeLimitRateLimiter:
    async def test_init_invalid_parameters(self):
        with pytest.raises(ValueError, match="max_actions must be > 0"):
            AsyncTimeWindowRateLimiter(max_actions=0, window_seconds=10)

        with pytest.raises(ValueError, match="max_actions must be > 0"):
            AsyncTimeWindowRateLimiter(max_actions=-1, window_seconds=10)

        with pytest.raises(ValueError, match="window_seconds must be > 0"):
            AsyncTimeWindowRateLimiter(max_actions=5, window_seconds=0)

        with pytest.raises(ValueError, match="min_interval must be >= 0"):
            AsyncTimeWindowRateLimiter(max_actions=5, window_seconds=10, min_interval=-0.1)

        with pytest.raises(TypeError, match="Only async functions can be decorated"):
            limiter = AsyncTimeWindowRateLimiter(max_actions=5, window_seconds=5)

            @limiter
            def testfunc():
                return 1

    async def test_instant_function(self):
        """'Instant' function will end up rate limited due to max actions."""
        limiter = AsyncTimeWindowRateLimiter(max_actions=5, window_seconds=5)

        @limiter
        async def testfunc():
            return 1

        start = time.monotonic()
        result = [await testfunc() for _ in range(10)]
        end = time.monotonic()

        assert sum(result) == 10
        # Limiter allows 5 actions every 5 seconds.
        # 5 actions in the first 5-second-window, then the remaining 5 almost immediately after.
        assert end - start > 5

    async def test_function_with_nonblocking_runtime(self):
        """If function runtime * window_actions > window_seconds, then no blocking occurs."""
        limiter = AsyncTimeWindowRateLimiter(max_actions=5, window_seconds=5)

        @limiter
        async def testfunc():
            time.sleep(2)
            return 1

        start = time.monotonic()
        result = [await testfunc() for _ in range(10)]
        end = time.monotonic()

        assert sum(result) == 10
        # Limiter allows 5 actions every 5 seconds.
        # Each action takes 2 seconds, so each window will only hold 2.5 actions
        # The limiter is not the blocker, the function runtime is.
        assert end - start >= 20
