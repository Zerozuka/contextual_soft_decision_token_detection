"""Loguru file logging."""

from loguru import logger


def setup_logger(level: str = "INFO") -> None:
    """Also write the log to ``main.log`` in the (Hydra) run directory."""
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | "
        "<level>{level: <8}</level> | <yellow>Line {line: >4}"
        " ({file}):</yellow> <b>{message}</b>"
    )
    logger.add(
        "main.log",
        level=level,
        format=log_format,
        colorize=False,
        backtrace=True,
        diagnose=True,
    )
