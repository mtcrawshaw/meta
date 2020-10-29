"""
Logger utility object to write to log files during training.
"""


class Logger:
    """
    Logger object.
    """

    def __init__(self, log_path: str = None) -> None:
        """ Init function for Logger object. """

        self.log_path = log_path

    def log(self, msg: str) -> None:
        """ Write to log file. """

        if self.log_path is None:
            return

        with open(self.log_path, "a+") as f:
            f.write(msg)


logger = Logger()
