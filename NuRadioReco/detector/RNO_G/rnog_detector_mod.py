import copy
import logging

from NuRadioReco.detector.RNO_G.rnog_detector import \
    Detector, _keys_not_in_dict, _check_detector_time


def replace_value_in_dict(d, keys, value):
    """
    Replaces the value of a nested dict entry.
    Example:
        d = {1: {2: {3: 1, 4: 2}}}
        replace_value_in_dict(d, [1, 2, 4], 14)
        print(d)
        # {1: {2: {3: 1, 4: 14}}}
    """
    if isinstance(keys, str):
        keys = [keys]

    d_tmp = d
    while True:
        key = keys.pop(0)  # get the first key
        if not len(keys):
            d_tmp[key] = value
            break
        else:
            d_tmp = d_tmp[key]


class ModDetector(Detector):
    def __init__(self, *args, **kwargs):
        super(ModDetector, self).__init__(*args, **kwargs)
        self.logger = logging.getLogger("NuRadioReco.RNOGDetectorMod")
        self.logger.setLevel(kwargs["log_level"])


    def modify_channel_description(self, station_id, channel_id, keys, value):
        """
        This function allows you to replace/modifty the description of a channel.
        Parameters
        ----------
        station_id: int
            The station id
        channel_id: int
            The channel id
        keys: list of str
            The list of keys of the corresponding part of the description to be changed
        value: various types
            The value of the description to be changed
        """

        if not self.has_station(station_id):
            err = f"Station id {station_id} not commission at {self.get_detector_time()}"
            self.logger.error(err)
            raise ValueError(err)

        channel_dict = self.get_channel(station_id, channel_id)
        if _keys_not_in_dict(channel_dict, keys):  # to simplify the code here all keys have to exist already
            raise KeyError(
                f"Could not find {keys} for station.channel {station_id}.{channel_id}.")

        replace_value_in_dict(channel_dict, keys, value)


    @_check_detector_time
    def get_channel(self, station_id, channel_id):
        """
        Returns a dictionary of all channel parameters

        Parameters
        ----------
        station_id: int
            The station id
        channel_id: int
            The channel id

        Returns
        -------

        channel_info: dict
            Dictionary of channel parameters
        """
        self.get_signal_chain_response(station_id, channel_id)  # this adds `total_response` to dict
        channel_data = copy.deepcopy(self._Detector__get_channel(station_id, channel_id, with_position=True, with_signal_chain=True))

        for key in self._Detector__default_values:

            # In this class overwritting is valid
            if isinstance(self._Detector__default_values[key], dict):
                channel_data[key] = self._Detector__default_values[key][channel_id]
            else:
                channel_data[key] = self._Detector__default_values[key]

        return channel_data


    def modify_station_description(self, station_id, keys, value):
            """
            This function allows you to replace/modifty the description of a channel.
            Parameters
            ----------
            station_id: int
                The station id
            keys: list of str
                The list of keys of the corresponding part of the description to be changed
            value: various types
                The value of the description to be changed
            """
            station_data = self.get_station(station_id)

            if _keys_not_in_dict(station_data, keys):  # to simplify the code here all keys have to exist already
                raise KeyError(
                    f"Could not find {keys} for station {station_id}.")

            replace_value_in_dict(station_data, keys, value)


    def export(self, filename, json_kwargs=None):
        """
        Export the buffered detector description.

        Parameters
        ----------

        filename: str
            Filename of the exported detector description

        json_kwargs: dict
            Arguments passed to json.dumps(..). (Default: None -> dict(indent=0, default=_json_serial))
        """
        raise NotImplementedError("Exporting the detector description is not implemented for this class.")
