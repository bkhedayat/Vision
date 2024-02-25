import logging 

LOGGING_NAME = "Yolo"
class Logger:
    "Implements log messages"
    def __init__(this):
        # set basic config
        logging.basicConfig(format="%(asctime)s: %(name)s: %(message)s", level=logging.INFO)

    def create(this, name):
        """ Creates the logger"""
        # set the logging level to warning
        logging.getLogger(name).setLevel(logging.WARNING)
        return logging.getLogger(__name__)


# create Logger instance
LOGGER = Logger().create(LOGGING_NAME)