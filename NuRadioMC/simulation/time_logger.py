import astropy.time

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
        Default is 5 seconds.

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
    show_time(n_event_groups, i_event_group)
        Display the progress information of a simulation run.
    """

    def __init__(self, logger, update_interval=5):
        """
        Initialize the TimeLogger object.

        Parameters
        ----------
        logger : object
            The logger object used for logging.
        update_interval : int, optional
            The time interval (in seconds) at which the logger should be updated.
            Default is 5 seconds.
        """
        self.__times = {}
        self.__total_start_time = None
        self.__start_times = {}
        self.__last_update = None
        self.__logger = logger
        self.__update_interval = astropy.time.TimeDelta(update_interval, format='sec')

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
        self.__times = {}
        self.__total_start_time = astropy.time.Time.now()
        self.__last_update = astropy.time.Time.now()
        if categories is None:
            for key in self.__times:
                self.__times[key] = 0
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
        This method starts the timer for the specified category by recording the current time using `astropy.time.Time.now()`.

        If the category does not exist in the time log, it will be added and the timer will be reset to zero.

        Examples
        --------
        >>> logger = TimeLogger()
        >>> logger.start_time('simulation')
        """
        if category not in self.__times:
            self.__times[category] = 0
            self.__logger.info(f"Time category {category} not found. Adding category and resetting time to zero.")
        self.__start_times[category] = astropy.time.Time.now()

    def stop_time(self, category):
        """
        Stop the timer for the specified category.

        Parameters
        ----------
        category : str
            The category for which to stop the timer.

        Raises
        ------
        KeyError
            If the specified category is not found in the time logger.
        RuntimeError
            If the timer for the specified category was not started before stopping it.
        """
        if category not in self.__times:
            raise KeyError('Time category {} not found. Did you reset times?'.format(category))
        if category not in self.__start_times.keys() or self.__start_times[category] is None:
            raise RuntimeError('It looks like you stopped taking time for {} before starting it.'.format(category))
        self.__times[category] += (astropy.time.Time.now() - self.__start_times[category]).sec
        self.__start_times[category] = None

    def show_time(self, n_event_groups, i_event_group):
        """
        Display the progress information of a simulation run.

        Parameters:
            n_event_groups (int): Total number of event groups.
            i_event_group (int): Index of the current event group.

        Returns:
            None

        Raises:
            None
        """
        if astropy.time.Time.now() - self.__last_update > self.__update_interval:
            self.__last_update = astropy.time.Time.now()
            elapsed_time = astropy.time.Time.now() - self.__total_start_time
            projected_time = elapsed_time * (n_event_groups - i_event_group - 1) / (i_event_group + 1)
            total_accounted_time = 0
            time_account_string = ''
            for category in self.__times:
                total_accounted_time += self.__times[category]
                time_account_string = time_account_string + '{} = {:.0f}%, '.format(category, self.__times[category] / elapsed_time.sec * 100)
            time_account_string = time_account_string + 'unaccounted: {:.0f}%'.format((elapsed_time.sec - total_accounted_time) / elapsed_time.sec * 100)
            self.__logger.status(
                'Processing event group {}/{}. ETA: {}, time consumption: {}'.format(
                    i_event_group,
                    n_event_groups,
                    pretty_time_delta(projected_time.sec),
                    time_account_string
                )
            )