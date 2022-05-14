"""Utilities for logging"""
import logging
from termcolor import colored


def color(string: str, color_name: str = 'yellow') -> str:
    """Returns colored string for output to terminal"""
    return colored(string, color_name)


def print_update(message: str, width: int = 140, fillchar: str = ":", color="yellow") -> str:
    """Prints an update message

    Args:
        message (str): message
        width (int): width of new update message
        fillchar (str): character to be filled to L and R of message

    Returns:
        str: print-ready update message
    """
    message = message.center(len(message) + 2, " ")
    print(colored(message.center(width, fillchar), color))


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    Args:
        log_path (str): path to the log file
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
