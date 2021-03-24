import six
import NuRadioReco.detector.detector
import NuRadioReco.detector.generic_detector
import NuRadioReco.modules.io.NuRadioRecoio
import NuRadioReco.utilities.metaclasses


@six.add_metaclass(NuRadioReco.utilities.metaclasses.Singleton)
class DetectorProvider(object):
    """
    Class to provide the detector object to the other functions.
    By declaring this class a Singleton, all functions have access
    to the same DetectorProvider object and always use the same
    detector object.
    """
    def __init__(self):
        self.__detector = None
        self.__io = None
        self.__current_event_i = None
        self.__unix_time_periods = None
        self.__astropy_time_periods = None

    def set_detector(self, filename, assume_inf, antenna_by_depth):
        """
        Creates a detector object that can be provided to the functions

        Parameters:
        -------------
        filename: string
            Path to the .json file containing the detector description
        """
        self.__detector = NuRadioReco.detector.detector.Detector.__new__(NuRadioReco.detector.detector.Detector)
        self.__detector.__init__(source='json', json_filename=filename, assume_inf=assume_inf, antenna_by_depth=antenna_by_depth)

    def set_generic_detector(self, filename, default_station, default_channel, assume_inf, antenna_by_depth):
        """
        Creates a GenericDetector object that can be provided to the functions

        Parameters:
        -----------------
        filename: string
            Path to the .json file containing the detector description
        default_station: int
            ID of the station to be set as default station
        default_channel: int or None
            ID of the channel to be set as default channel
        """
        import NuRadioReco.detector.generic_detector
        self.__detector = NuRadioReco.detector.generic_detector.GenericDetector.__new__(NuRadioReco.detector.generic_detector.GenericDetector)
        self.__detector.__init__(filename, default_station, default_channel, assume_inf=assume_inf, antenna_by_depth=antenna_by_depth)

    def set_event_file(self, filename):
        """
        Reads the detector description stored in a .nur file

        Parameters:
        ---------------------
        filename: string
            Path to the .nur file containing the detector description
        """
        self.__io = NuRadioReco.modules.io.NuRadioRecoio.NuRadioRecoio([filename])
        event = self.__io.get_event_i(0)
        self.__detector = self.__io.get_detector()
        self.__current_event_i = 0
        for station in event.get_stations():
            self.__detector.update(station.get_station_time())
            break

    def get_detector(self):
        """
        Returns the current detector description object
        """
        return self.__detector

    def set_time_periods(self, unix, astropy_time):
        """
        Store the time periods contained in a detector file

        Parameters:
        ----------------
        unix: array of numbers
            Unix timestamps of the time periods
        astropy_time: array of astropy times
            Astropy time objects of the time periods
        """
        self.__unix_time_periods = unix
        self.__astropy_time_periods = astropy_time

    def get_time_periods(self):
        """
        Get the time periods stored in this class.
        Returns the same periods as unix time stamps
        and astropy time objects
        """
        return self.__unix_time_periods, self.__astropy_time_periods

    def get_n_events(self):
        """
        Get the number of events in the event file
        from which the detector is read
        """
        if self.__io is None:
            return 0
        return self.__io.get_n_events()

    def get_event_ids(self):
        """
        Get the IDs of the events in the event file
        from which the detector is read
        """
        if self.__io is None:
            return None
        return self.__io.get_event_ids()

    def set_event(self, i_event):
        """
        Set the event for which the detector description
        should be given.
        Only works if detector description is read from
        a .nur file.

        Parameters:
        ------------------
        i_event: int
            Number of the event for which the detector
            description should be read. Note that this is
            not the event ID, but the event index!
        """
        if self.__io is None:
            return
        event = self.__io.get_event_i(i_event)
        self.__detector = self.__io.get_detector()
        self.__current_event_i = i_event
        for station in event.get_stations():
            self.__detector.update(station.get_station_time())
            break

    def get_current_event_i(self):
        """
        Return the index (not the ID) of the event
        for which the detector description is currently
        read. Only works if detector is read from a
        .nur file.
        """
        return self.__current_event_i
