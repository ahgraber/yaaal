import asyncio
from typing import Callable


def get_create_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create a running event loop."""
    try:
        from IPython.core.getipython import get_ipython

        if get_ipython() is not None:
            import nest_asyncio

            nest_asyncio.apply()

    except ImportError:
        pass

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()

    return loop


def synchronize(afunc: Callable, *args, **kwargs):
    """Run async function in synchronous context."""
    loop = get_create_event_loop()
    return loop.run_until_complete(afunc(*args, **kwargs))
