import logging

LOGGING_STATUS = 25


class NuRadioLogger(logging.Logger):
    """
    Custom logging class for NuRadio modules and applications. It adds a custom log level STATUS,
    which has level=`LOGGING_STATUS` as defined in `logging.py` (as of February 2024, its value is 25).
    The associated `status()` call is also implemented.
    """
    def __init__(self, name):
        super().__init__(name)

        # Add STATUS as the level name, to be used in message formatting
        logging.addLevelName(LOGGING_STATUS, "STATUS")

    def status(self, message, *args, **kwargs):
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

    Examples
    --------
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
    Set up the parent logger which all module loggers should pass their logs on to. If this one already
    exists, nothing is done and the logger is returned as is. Otherwise, a single new `logging.StreamHandler()`
    with a custom formatter is added.

    Parameters
    ----------
    name : str, default="NuRadioReco"
        The name of the base logger
    level : int, default=25
        The logging level to use for the base logger

    Notes
    -----
    This function is only meant to be called once, on import. It is part of the `init` scripts, so you
    should not need to call this function explicitly.
    """
    logger = logging.getLogger(name)

    if len(logger.handlers) > 0:  # method hasHandlers() also checks parents -> ends up at root logger
        # Don't change the logger if it already exists
        logger.warning(f"Logger {name} already has handlers. Not changing anything, returning the existing logger...")
        return logger
    logger.propagate = False

    # Create a StreamHandler with fancy formatter
    handler = logging.StreamHandler()
    handler.setFormatter(get_fancy_formatter())
    handler.setLevel(1)  # we want the handler to be accepting all records from child loggers

    # Then add our custom handler to the logger
    logger.addHandler(handler)

    # Finally, set the logging level
    if level is not None:
        logger.setLevel(level=level)
    else:
        logger.setLevel(LOGGING_STATUS)

    return logger


def get_fancy_formatter():
    """
    Returns the formatter used in the NuRadio logger.

    Returns
    -------
    formatter : logging.Formatter
    """
    formatter = logging.Formatter(
        '\033[33;20m%(levelname)s - \033[93m%(asctime)s - \033[32m%(name)s - \033[0m%(message)s',
        datefmt="%Y %b %d @ %H:%M:%S UTC%z"
    )
    return formatter


def set_general_log_level(level):
    """
    Set the logging level of the NuRadioMC and NuRadioReco loggers to `level`.

    Parameters
    ----------
    level : int
        The desired logging level
    """
    nrr_logger = logging.getLogger("NuRadioReco")
    nrr_logger.setLevel(level)

    nrmc_logger = logging.getLogger("NuRadioMC")
    nrmc_logger.setLevel(level)
