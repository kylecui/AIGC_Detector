import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_dir: str | Path = "logs", level: int = logging.INFO) -> logging.Logger:
    """Configure structured logging with rotation."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        log_path / "aigc_detector.log",
        maxBytes=10_000_000,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger("aigc_detector")
    logger.setLevel(level)
    logger.addHandler(handler)

    # Also log to console via rich
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    logger.addHandler(console_handler)

    return logger
