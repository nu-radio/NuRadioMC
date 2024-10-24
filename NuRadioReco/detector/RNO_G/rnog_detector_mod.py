import copy
import logging

import numpy as np
from NuRadioReco.detector.response import Response

from NuRadioReco.detector.RNO_G.rnog_detector import \
    Detector, _keys_not_in_dict, _check_detector_time


def replace_value_in_dict(d, keys, value):
    """
    Replaces the value of a nested dict entry.

    Examples
    --------

    .. code-block::

        d = {1: {2: {3: 1, 4: 2}}}
        replace_value_in_dict(d, [1, 2, 4], 14)
        print(d) # {1: {2: {3: 1, 4: 14}}}

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

    def add_response(self, station_id, channel_id, response):
        """ Add an additional response to the `total_response`

        Parameters
        ----------

        station_id: int
            The station id

        channel_id: int
            The channel id

        response: response.Response
            A response object to be added to the `total_response`

        """

        orig_response = self.get_signal_chain_response(station_id, channel_id)
        signal_chain_dict = self.get_channel_signal_chain(station_id, channel_id)

        # modify the total response
        signal_chain_dict["total_response"] = orig_response * response

        # write the modified signal chain back to the buffered station
        self._Detector__buffered_stations[station_id]["channels"][channel_id]['signal_chain'] = signal_chain_dict

    def add_component(self, station_id, channel_id, component):
        """ Add an additional component to the `response_chain` and the corresponding response to the `total_response`

        Parameters
        ----------

        station_id: int
            The station id

        channel_id: int
            The channel id

        componennt: dict
            A dictionary with the properties of the component to be added
          
        """

        # generate a response object from the component dict
        component_response = Response(component['frequencies'],
                                        np.array([component['mag'], component['phase']]),
                                        component['y-axis_units'],
                                        time_delay=component['time_delay'],
                                        name=component['name'],
                                        station_id=station_id,
                                        channel_id=channel_id)

        orig_response = self.get_signal_chain_response(station_id, channel_id)
        signal_chain_dict = self.get_channel_signal_chain(station_id, channel_id)

        # modify the signal chain
        signal_chain_dict["total_response"] = orig_response * component_response
        signal_chain_dict["response_chain"][component['name']] = component

        # write the modified signal chain back to the buffered station
        self._Detector__buffered_stations[station_id]["channels"][channel_id]['signal_chain'] = signal_chain_dict

    def add_manual_time_delay(self, station_id, channel_id, time_delay):
        """ Add an additional time delay to the signal chain and total response

        Parameters
        ----------

        station_id: int
            The station id

        channel_id: int
            The channel id

        time_delay: float
            The manual time delay to be added

        """

        # dummy dict, that would just add a unity to the response
        null_component = {'weight': 1,
                          'y-axis_units': ['mag', 'rad'],
                          'mag': [1,1],
                          'phase': [0,0],
                          'frequencies': [1e-3,1e1],
                          'time_delay': 0}

        # specify component starting from defaults
        component = copy.copy(null_component)
        component['name'] = "MOD_manual_time_delay"
        component['time_delay'] = time_delay

        # add the component to the response chain
        self.add_component(station_id, channel_id, component)

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
