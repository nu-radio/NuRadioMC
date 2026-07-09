import time
from collections import defaultdict

def pretty_time_delta(seconds):
    """
    Convert a time duration in seconds to a human-readable format.

    Parameters
    ----------
    seconds : int
        The time duration in seconds.

    Returns
    -------
    str
        The human-readable time duration in the format 'XdXhXmXs', where X represents the number of days, hours, minutes, and seconds.

    Examples
    --------
    >>> logger = TimeLogger()
    >>> logger.__pretty_time_delta(3665)
    '0d1h1m5s'
    >>> logger.__pretty_time_delta(7200)
    '0d2h0m0s'
    >>> logger.__pretty_time_delta(120)
    '0d0h2m0s'
    >>> logger.__pretty_time_delta(30)
    '0d0h0m30s'
    """

    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%dd%dh%dm%ds' % (days, hours, minutes, seconds)
    elif hours > 0:
        return '%dh%dm%ds' % (hours, minutes, seconds)
    elif minutes > 0:
        return '%dm%ds' % (minutes, seconds)
    else:
        return '%ds' % (seconds,)


class timeLogger:
    """
    A class for logging and tracking time durations for different categories.

    Parameters
    ----------
    logger : object
        The logger object used for logging.
    update_interval : int, optional
        The time interval (in seconds) at which the logger should be updated.
        Default is 20 seconds.

    Methods
    -------
    reset_times(categories=None)
        Reset the time values for the specified categories or all categories.
    start_time(category)
        Start counting times for a specific category.
    stop_time(category)
        Stop the timer for the specified category.
    pretty_time_delta(seconds)
        Convert a time duration in seconds to a human-readable format.
    show_time(n_event_groups, i_event_group, other_threshold=2, num_triggers=None)
        Display the progress information of a simulation run.
    """

    def __init__(self, logger, update_interval=20):
        """
        Initialize the TimeLogger object.

        Parameters
        ----------
        logger : object
            The logger object used for logging.
        update_interval : int, optional
            The time interval (in seconds) at which the logger should be updated.
            Default is 20 seconds.
        """
        self.__times = defaultdict(float)
        self.__total_start_time = None
        self.__start_times = {}
        self.__last_update = None
        self.__logger = logger
        self.__update_interval = update_interval  # sec

    def reset_times(self, categories=None):
        """
        Reset the time values for the specified categories or all categories.

        Parameters
        ----------
        categories : list, optional
            A list of categories for which the time values should be reset.
            If not provided, all categories will be reset.

        Returns
        -------
        None

        Notes
        -----
        This method resets the time values for the specified categories or all categories
        to zero. It also updates the total start time and the last update time.

        Examples
        --------
        >>> logger = TimeLogger()
        >>> logger.reset_times()  # Reset all categories
        >>> logger.reset_times(['category1', 'category2'])  # Reset specific categories
        """
        self.__total_start_time = time.monotonic()
        self.__last_update = time.monotonic()

        if categories is None:
            self.__times = defaultdict(float)
        else:
            for category in categories:
                self.__times[category] = 0

    def start_time(self, category):
        """
        Start counting times for a specific category.

        Parameters
        ----------
        category : str
            The category for which to start the timer.

        Notes
        -----
        This method starts the timer for the specified category by recording the current time using `time.monotonic()`.

        If the category does not exist in the time log, it will be added and the timer will be reset to zero.

        Examples
        --------
        >>> logger = TimeLogger()
        >>> logger.start_time('simulation')
        """
        self.__start_times[category] = time.monotonic()

    def stop_time(self, category):
        """
        Stop the timer for the specified category.

        Parameters
        ----------
        category : str
            The category for which to stop the timer.

        Raises
        ------
        RuntimeError
            If the timer for the specified category was not started before stopping it.
        """
        if category not in self.__start_times.keys() or self.__start_times[category] is None:
            raise RuntimeError('It looks like you stopped taking time for {} before starting it.'.format(category))

        self.__times[category] += time.monotonic() - self.__start_times[category]
        self.__start_times[category] = None

    def show_time(self, n_event_groups, i_event_group, other_threshold=2, num_triggers=None, force=False):
        """
        Display the progress information of a simulation run.

        Parameters
        ----------
        n_event_groups : int
            Total number of event groups.
        i_event_group : int
            Index of the current event group.
        other_threshold : float, optional
            Categories that account for less than this percentage of the
            elapsed time are grouped together into a single 'others' column
            instead of being shown individually. Default is 2%.
        num_triggers : int, optional
            The number of triggered events so far. If provided, it is
            included in the status message.
        force : bool, optional
            Print regardless of interval. (Default: False)

        Returns
        -------
        None
        """
        if time.monotonic() - self.__last_update > self.__update_interval or force:
            self.__last_update = time.monotonic()
            elapsed_time = time.monotonic() - self.__total_start_time
            projected_time = elapsed_time * (n_event_groups - i_event_group - 1) / (i_event_group + 1)
            total_accounted_time = sum(self.__times.values())

            names = []
            percentages = []
            other_time = 0
            for category, category_time in self.__times.items():
                percentage = category_time / elapsed_time * 100
                if percentage < other_threshold:
                    other_time += category_time
                else:
                    names.append(category)
                    percentages.append('{:.0f}%'.format(percentage))

            if other_time > 0:
                names.append('others')
                percentages.append('{:.0f}%'.format(other_time / elapsed_time * 100))

            names.append('unaccounted')
            percentages.append('{:.0f}%'.format((elapsed_time - total_accounted_time) / elapsed_time * 100))

            widths = [max(len(name), len(pct)) for name, pct in zip(names, percentages)]
            header_row = ' | '.join(name.center(width) for name, width in zip(names, widths))
            value_row = ' | '.join(pct.center(width) for pct, width in zip(percentages, widths))
            time_account_string = '\t{}\n\t{}'.format(header_row, value_row)

            num_trigger_str = ''
            if num_triggers is not None:
                num_trigger_str = f" ({num_triggers} triggered events)"

            self.__logger.status(
                'Processing event group {} / {}{}. Duration: {}, ETA: {}. \n{}'.format(
                    i_event_group,
                    n_event_groups,
                    num_trigger_str,
                    pretty_time_delta(elapsed_time),
                    pretty_time_delta(projected_time),
                    time_account_string
                )
            )