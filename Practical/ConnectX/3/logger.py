import os
import logging

FILE = "game.log"
LEVEL = logging.DEBUG
LOGGING_ENABLED = False


class _FileHandler(logging.FileHandler):
    def emit(self, record):
        if not LOGGING_ENABLED:
            return
        super().emit(record)

def init_logger(_logger):
    if os.path.exists(FILE):
        os.remove(FILE)

    while _logger.hasHandlers():
        _logger.removeHandler(_logger.handlers[0])

    _logger.setLevel(LEVEL)
    ch = _FileHandler(FILE)
    ch.setLevel(LEVEL)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H-%M-%S"
    )
    ch.setFormatter(formatter)
    _logger.addHandler(ch)


logger = logging.getLogger()
