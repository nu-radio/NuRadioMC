import astropy.time

class timeLogger:
    def __init__(
        self,
        logger,
        update_interval=5
        ):
        self.__times = {}
        self.__total_start_time = None
        self.__start_times = {}
        self.__last_update = None
        self.__logger = logger
        self.__update_interval = astropy.time.TimeDelta(update_interval, format='sec')
    
    def reset_times(self, categories):
        self.__times = {}
        self.__total_start_time = astropy.time.Time.now()
        self.__last_update = astropy.time.Time.now()
        for category in categories:
            self.__times[category] = 0
    
    def start_time(self, category):
        if category not in self.__times.keys():
            raise KeyError('Time category {} not found. Did you reset times?'.format(category))
        self.__start_times[category] = astropy.time.Time.now()
    
    def stop_time(self, category):
        if category not in self.__times.keys():
            raise KeyError('Time category {} not found. Did you reset times?'.format(category))
        if category not in self.__start_times.keys() or self.__start_times[category] is None:
            raise RuntimeError('It looks like you stopped taking time for {} before starting it.'.format(category))
        self.__times[category] += (astropy.time.Time.now() - self.__start_times[category]).sec
        self.__start_times[category] = None
    
    def __pretty_time_delta(self, seconds):
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

    def show_time(
        self,
        n_event_groups,
        i_event_group
        ):
        if astropy.time.Time.now() - self.__last_update > self.__update_interval:
            self.__last_update = astropy.time.Time.now()
            elapsed_time = astropy.time.Time.now() - self.__total_start_time
            projected_time = elapsed_time * (n_event_groups - i_event_group - 1) / (i_event_group + 1)
            total_accounted_time = 0
            time_account_string = ''
            for category in self.__times.keys():
                total_accounted_time += self.__times[category]
                time_account_string = time_account_string + '{} = {:.0f}%, '.format(category, self.__times[category] / elapsed_time.sec * 100)
            time_account_string = time_account_string + 'unaccounted: {:.0f}%'.format((elapsed_time.sec - total_accounted_time) / elapsed_time.sec * 100)
            self.__logger.status(
                'processing event group {}/{}. ETA: {}, time consumption: {}'.format(
                    i_event_group,
                    n_event_groups,
                    self.__pretty_time_delta(projected_time.sec),
                    time_account_string
                )
            )