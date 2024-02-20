import logging 

LOGGING_NAME = "Yolo"
class Logger:
    "Implements log messages"
    def init():
        # set basic config
        logging.basicConfig(format="%(asctime)s: %(name)s: %(message)s", level=logging.INFO)

    def create(this) -> logging.Logger:
        """ Creates the logger"""
        # set the logging level to warning
        logging.getLogger(LOGGING_NAME).setLevel(logging.WARNING)
        return logging.getLogger(__name__)
