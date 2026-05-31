"""
Custom logger for the MapleStory auto-detection system.
Creates timestamped log files and outputs to console.
"""
import logging
import os
import datetime


class MSLogger:
    """Custom logger wrapping Python's logging module."""

    def __init__(self, name: str = "MSBot"):
        self.logger = logging.Logger(name)
        self.logger.setLevel(logging.INFO)

        # File handler: log/<name>_<timestamp>.log
        os.makedirs("log", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = os.path.join("log", f"{name}_{timestamp}.log")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def set_level(self, level: int):
        self.logger.setLevel(level)

    def add_handler(self, handler: logging.Handler):
        self.logger.addHandler(handler)


# Module-level singleton for global use
logger = MSLogger()
