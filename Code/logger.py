import logging
from paths import LOG_PATH

def get_logger(name: str = "project_logger") -> logging.Logger:
    """
    Creates and returns a logger with file and console output.

    Args:
        name (str): Logger name (default: "project_logger")

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # File handler: logs detailed messages to a file
    file_handler = logging.FileHandler(LOG_PATH, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Stream handler: logs concise messages to console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_formatter = logging.Formatter('%(levelname)s - %(message)s')
    stream_handler.setFormatter(stream_formatter)

    # Attach handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
