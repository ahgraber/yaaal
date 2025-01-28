import contextlib
import logging

import tqdm

LOG_FMT = "%(asctime)s - %(levelname)-8s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"


def basic_log_config(level=logging.WARNING, **kwargs) -> None:
    """Configure logging defaults for all loggers."""
    logging.basicConfig(level=level, format=LOG_FMT, **kwargs)


@contextlib.contextmanager
def suppress_logs(logger: logging.Logger):
    """Context manager to temporarily disable logs."""
    try:
        logger.disabled = True
        yield
    finally:
        logger.disabled = False


@contextlib.contextmanager
def logging_redirect_tqdm(
    loggers: list[logging.Logger] | None = None,
    tqdm_class: type[tqdm.std.tqdm] = tqdm.std.tqdm,
    *,
    level: int | None = None,
):
    """Replace StreamHandlers with a TQDM handler of the same level.

    Ref: https://github.com/tqdm/tqdm/issues/1272#issuecomment-2065265443
    """
    loggers = loggers or [logging.getLogger()]
    if level is None:
        for logger in loggers:
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    level = handler.level
                    break

    with tqdm.contrib.logging.logging_redirect_tqdm(loggers, tqdm_class):
        if level is not None:
            for logger in loggers:
                for handler in logger.handlers:
                    if isinstance(handler, tqdm.contrib.logging._TqdmLoggingHandler):
                        handler.setLevel(level)
        yield
