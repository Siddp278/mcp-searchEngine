import logging
import os
from datetime import datetime

# Custom filter: Only allow DEBUG messages
class DebugOnlyFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.DEBUG


def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all logs

    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)  # Needed to allow filtering
    logger.addHandler(fh)

    # streamhandler streams to console, so limited information is processed to it.
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.DEBUG)
    sh.addFilter(DebugOnlyFilter())
    logger.addHandler(sh)

    return logger

