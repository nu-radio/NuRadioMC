import logging

LOGGING_STATUS = 25


class NuRadioLogger(logging.Logger):
    def __init__(self, name="NuRadioReco"):
        super().__init__(name)

        # Add STATUS as the level name, to be used in message formatting
        logging.addLevelName(LOGGING_STATUS, "STATUS")

    def addHandler(self, hdlr: logging.Handler):
        # Ensure that the logger level is always lower than the lowest handler
        if hdlr.level < self.level:
            self.setLevel(hdlr.level)

        super().addHandler(hdlr)

    def status(self, message, *args, **kwargs):
        print(message)
        if self.isEnabledFor(LOGGING_STATUS):
            self._log(LOGGING_STATUS, message, args, **kwargs)


def addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    Notes
    -----
    This function was taken from
    `this answer  <https://stackoverflow.com/questions/2183233/how-to-add-a-custom-loglevel-to-pythons-logging-facility/35804945#35804945>`_

    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        raise AttributeError('{} already defined in logging module'.format(levelName))
    if hasattr(logging, methodName):
        raise AttributeError('{} already defined in logging module'.format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError('{} already defined in logger class'.format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


def setup_logger(name="NuRadioReco", level=None):
    """
    Set up the parent logger which all module loggers should pass their logs on to. Any handler which was
    previously added to the logger is cleared, and a single new `logging.StreamHandler()` with a custom
    formatter is added. Next to this, an extra logging level STATUS is added with level=`LOGGING_STATUS`,
    which is defined in `module.py` (as of February 2024, its value is 25). Then STATUS is also set as
    the default logging level.

    Parameters
    ----------
    name : str, default="NuRadioReco"
        The name of the base logger
    level : int, default=25
        The logging level to use for the base logger
    """
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        # Don't change the logger if it already exists
        logger.warning(f"Logger {name} already has handlers. Not changing anything, returning the existing logger...")
        return logger
    logger.propagate = False

    # First clear all the handlers
    logger.handlers = []

    # Create a StreamHandler with correct level and format
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '\033[33;20m%(levelname)s - \033[93m%(asctime)s - \033[32m%(name)s - \033[0m%(message)s',
        datefmt="%Y %b %d @ %H:%M:%S UTC%z"
    )
    handler.setFormatter(formatter)

    if level is not None:
        handler.setLevel(level=level)
    else:
        handler.setLevel(LOGGING_STATUS)

    # Then add our custom handler to the logger
    logger.addHandler(handler)

    return logger
