import logging
import os
import time
__name__ = "histo"
__version__ = "0.1.0"


LOG_FILENAME = os.path.join('logs', f"{str(int(time.time()))}.lg")
LOG_FORMAT = "%(message)s"
logging.basicConfig(format=LOG_FORMAT)
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.FileHandler(filename=LOG_FILENAME, encoding="utf-8"))
LOGGER.setLevel(logging.INFO)
