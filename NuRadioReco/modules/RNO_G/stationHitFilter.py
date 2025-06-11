"""
Hit Filter Module (RNO-G specific, can be upgraded to more general use in the future)
The main purpose is to reject thermal events in data.
"""

from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import trace_utilities, units

from collections import defaultdict
import numpy as np
import logging
import math
import copy
import time


class stationHitFilter:

    def __init__(self, complete_time_check=False, complete_hit_check=False, time_window=10.0*units.ns, threshold_multiplier=6.5,
                 select_trigger=None):
        """
        Passes event through "hit filter". Looks for temporal coincidence in multiple channel pairs.

        Currently this module is designed specifically for the deep components of an RNO-G station and uses only
        the deep in-ice channels. The module checks for temporal coincidences in multiple channel pairs (uses only
        antennas at ~100 depth; i.e., not the shallow v-pols). Adjacent channels are put into groups:
        G0: (0,1,2,3); G1: (9,10); G2: (23,22); G3: (8,4). To determine the timing and amplitude of a "hit" in every channel,
        the Hilbert Transform is applied to the waveform and the maximum is found. The time of the maximum is then used to
        check for coincidences in each group. This "time checker" requires at least 1 coincident pair in G0 (PA),
        and another coincident pair in any other group (including G0 - with an additional requirement that channel
        pairs need to be connected). When the time check fails, by default, a "hit checker" will see if there's
        any in-ice channel that has a high hit (maximum > threshold_multiplier * noise_RMS) and whenever there's
        a high hit the event passes the module.

        Parameters
        ----------
        complete_time_check: bool (default: False)
            If False, time checker will stop early whenever criteria are satisfied.
            If True, time checker will run through all channel groups no matter what.
        complete_hit_check: bool (default: False)
            If False, only run the hit checker when the time checker fails.
            If True, always run the hit checker.
        time_window: float (default: 10.0*units.ns)
            Coincidence window for two adjacent channels.
        threshold_multiplier: float (default: 6.5)
            High hit threshold multiplier, where a hit threshold is the multiplier times the noise RMS.
        select_trigger: str (default: None)
            Select a specific trigger type to run this filter on. If None, all triggers will be evaluated.
            If you select a specific trigger, events with other triggers will be treated as if they have passed
            the module (but not counted).
        """
        self._complete_time_check = complete_time_check
        self._complete_hit_check = complete_hit_check
        self._dT = time_window
        self._threshold_multiplier = threshold_multiplier
        self._select_trigger = select_trigger

        self._in_ice_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]
        self._in_ice_channel_groups = ([0, 1, 2, 3], [9, 10], [23, 22], [8, 4])
        self._channel_pairs_in_PA = ([0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3])

        self._n_channels_in_ice = len(self._in_ice_channels)

        self._envelope_max_time_index = None
        self._envelope_max_time = None
        self._noise_RMS = None
        self._times = None
        self._traces = None
        self._envelope_traces = None
        self._hit_thresholds = None
        self._is_over_hit_threshold = None
        self._passed_hit_filter = None
        self._passed_time_checker = None
        self._passed_hit_checker = None

        ### 2-D list ###
        # This list contains sub-lists for all groups.
        # Each sub-list contains bool(s) indicating the COINCIDENCE of channel pair(s) in each group
        # self._is_in_time_window[0] is the PA, it has 6 pairs (6 bools)
        # Initiating each element with None, it will later be replaced with bool when running the time checker
        self._is_in_time_window_template = [[None for _ in range(math.factorial(len(group)-1))] for group in self._in_ice_channel_groups]


    def _channel_mapping(self, channel_in):
        """
        Channel Mappings for specifically channels on Helper String 2

        The Hit Filter works only on all 15 in-ice channels.
        This function maps channels 21, 22, 23 to indices 12, 13, 14
        (Skip surface channels 12 ~ 20).

        Parameters
        ----------
        channel_in: int
            Input channel number

        Returns
        -------
        channel_out: int
            Output channel number
        """
        channel_out = channel_in - 9 if channel_in >= 21 else channel_in
        return channel_out


    def _time_checker(self):
        """
        Check the hit times between channels in groups to select coincident pairs.

        See if there are at least 2 coincident pairs in time sequence in Group 0 (PA),
        and if there's only 1 pair in PA, then find the other pair in other groups;
        otherwise, the input event fails the time checker.

        Returns
        -------
        self._passed_time_checker: bool
            Event passed the time checker or not
        """
        self._passed_time_checker = False
        envelope_max_time = self.get_envelope_max_time()
        coincidences_in_time_sequence_PA = [False, False, False]

        self._is_in_time_window = copy.deepcopy(self._is_in_time_window_template)
        for i_group, group in enumerate(self._in_ice_channel_groups):
            # Group 0 is special because we require at least one coincident pair in this group
            if i_group == 0:
                # For e.g, channel 3 and 0 take 3 times the dT.
                dT_multipliers = np.diff(self._channel_pairs_in_PA).flatten()
                for i_channel_pair, channel_pair in enumerate(self._channel_pairs_in_PA):
                    hit_time_difference = abs(envelope_max_time[channel_pair[1]] - envelope_max_time[channel_pair[0]])
                    self._is_in_time_window[i_group][i_channel_pair] = hit_time_difference <= dT_multipliers[i_channel_pair] * self._dT

                    # This condition assures that the conicident pairs are in time sequence. For example:
                    # Having a conicdence only in pairs (0, 1) and (0, 3) is not valid. But having (0, 1) and (1, 3) is valid.
                    if self._is_in_time_window[i_group][i_channel_pair]:
                        coincidences_in_time_sequence_PA[channel_pair[0]] = self._is_in_time_window[i_group][i_channel_pair]

                if np.sum(coincidences_in_time_sequence_PA) >= 2:
                    self._passed_time_checker = True

            else:
                # If the time checker already passed, we can skip the rest of the groups when not completing the time checking
                if self._passed_time_checker and not self._complete_time_check:
                    break

                # If there are no coincidences in PA, we can skip the rest of the groups
                if not np.any(coincidences_in_time_sequence_PA) and not self._complete_time_check:
                    break

                if len(group) != 2:
                    raise NotImplementedError("For any channel group other than Group 0 (PA), only 2 channels are supported for now.")

                hit_time_difference = abs(envelope_max_time[self._channel_mapping(group[0])] - envelope_max_time[self._channel_mapping(group[1])])
                self._is_in_time_window[i_group][0] = hit_time_difference <= self._dT

                if self._is_in_time_window[i_group][0] and np.sum(coincidences_in_time_sequence_PA) >= 1:
                    self._passed_time_checker = True
                    if not self._complete_time_check:
                        break

        return self._passed_time_checker


    def _hit_checker(self):
        """
        Find a high hit within all 15 in-ice channels.

        See if there's at least 1 high hit within channels,
        if yes then the event passes the hit checker.

        Returns
        -------
        self._passed_hit_checker: bool
            Event passed the hit checker or not
        """
        self._is_over_hit_threshold = np.amax(self.get_envelope_traces(), axis=-1) > self.get_hit_thresholds()
        self._passed_hit_checker = np.any(self._is_over_hit_threshold)

        return self._passed_hit_checker


    def begin(self, log_level=logging.INFO):
        """
        Set up logging info.

        Parameters
        ----------
        log_level: enum
            Set verbosity level of logger (default: logging.INFO)
        """
        self.logger = logging.getLogger('NuRadioReco.RNO_G.stationHitFitter')
        self.logger.setLevel(log_level)
        self.__counting_dict = defaultdict(int)
        self.__is_wanted_trigger_type = None
        self.__total_run_time = 0


    def set_up(self, set_of_traces, set_of_times, noise_RMS):
        """
        Set things up before passing to the Hit Filter.

        This setup function calculates the noise RMS,
        sets the high hit threshold for each channel, gets the Hilbert envelope,
        and finds the time when the maximum happens (hit).
        IMPORTANT: If you use this function by yourself,
        make sure your inputs come from all 15 in-ice channels in order only,
        rather than all 24 channels.

        Parameters
        ----------
        set_of_traces: 2-D array of floats
            A set of input trace arrays of all 15 in-ice channels
        set_of_times: 2-D array of floats
            A set of input times arrays of all 15 in-ice channels
        noise_RMS: 1-D array of floats
            A set of input noise RMS values of all 15 in-ice channels
        """
        self._passed_hit_checker = None

        if len(set_of_traces) != self._n_channels_in_ice or len(set_of_times) != self._n_channels_in_ice:
            raise NotImplementedError("Make sure your inputs come from all in-ice channels in order for Hit Filter to perform.")

        self._traces = set_of_traces
        self._times = set_of_times
        self._envelope_traces = trace_utilities.get_hilbert_envelope(self._traces)
        self._envelope_max_time_index = np.argmax(self._envelope_traces, axis=-1)
        self._envelope_max_time = self._times[range(len(self._times)), self._envelope_max_time_index]

        if noise_RMS is not None:
            if len(noise_RMS) != self._n_channels_in_ice:
                raise NotImplementedError("Make sure your input RMS values come from all in-ice channels in order for Hit Filter to perform.")
            self._noise_RMS = noise_RMS
        else:
            self._noise_RMS = np.array([trace_utilities.get_split_trace_noise_RMS(trace) for trace in self._traces])

        self._hit_thresholds = self._noise_RMS * self.get_threshold_multiplier()


    def apply_hit_filter(self):
        """
        See if the input event will survive the Hit Filter or not.

        After the setup, it first checks with the time checker,
        if event passed the time checker then it passes the Hit Filter;
        if event failed the time checker, then checks with the
        hit checker to find a high hit.

        Returns
        -------
        self._passed_hit_filter: bool
            Event passed the Hit Filter or not
        """
        self._passed_hit_filter = self._time_checker()

        # If time checker failed or if you want to complete the hit checking regardless
        if not self._passed_hit_filter or self._complete_hit_check:
            self._hit_checker()

        # Only update _passed_hit_filter with the hit checker when the time checker did not pass
        if not self.is_passed_time_checker():
            self._passed_hit_filter = self.is_passed_hit_checker()

        return self._passed_hit_filter


    @register_run()
    def run(self, evt, station, det=None, noise_RMS_all=None):
        """
        Runs the Hit Filter.

        Parameters
        ----------
        evt: `NuRadioReco.framework.event.Event`
            Using the event object to get the trigger type
        station: `NuRadioReco.framework.station.Station`
            The station to use the Hit Filter on
        det: Detector object | None
            Detector object (not used in this method,
            included to have the same signature as other NuRadio classes)
        noise_RMS_all: 1-D numpy array (default: None)
            Noise RMS values of all 24 channels, if not given the Hit Filter will calculate them for the 15 in-ice channels

        Returns
        -------
        self.is_passed_hit_filter(): bool
            Event passed the Hit Filter or not
        """
        t0 = time.time()
        trigger_type = evt.get_station().get_first_trigger().get_name()

        # Only run the module on selected trigger type
        if self._select_trigger is not None and trigger_type != self._select_trigger:
            self._passed_hit_filter = True
            return True

        # This implicitly obeys the channel mapping
        traces = np.array([np.array(channel.get_trace()) for channel in station.iter_channels() if channel.get_id() in self._in_ice_channels])
        times = np.array([np.array(channel.get_times()) for channel in station.iter_channels() if channel.get_id() in self._in_ice_channels])

        if noise_RMS_all is not None:
            noise_RMS = noise_RMS_all[self._in_ice_channels]  # HACK: use channel IDs to index noise_RMS
        else:
            noise_RMS = noise_RMS_all

        self.set_up(traces, times, noise_RMS)
        self.apply_hit_filter()

        self.__is_wanted_trigger_type = trigger_type == "LT"
        self.__counting_dict[trigger_type] += 1
        self.__counting_dict[f"{trigger_type}_passed"] += int(self.is_passed_hit_filter())
        self.__counting_dict["total"] += 1

        self.__total_run_time += time.time() - t0
        return self.is_passed_hit_filter()


    def end(self):
        event_count = self.__counting_dict.pop("total")
        counts = (f"Processed Total: {event_count} events. Total run time: {self.__total_run_time:.2f} s, "
            f"Time per event: {self.__total_run_time / event_count * 1000:.2f} ms")

        trigger_types = np.unique([key.strip("_passed") for key in self.__counting_dict.keys()])
        for trigger_type in trigger_types:
            counts += (f"\n{trigger_type:>10} triggers: {self.__counting_dict[f'{trigger_type}_passed']:4d} / {self.__counting_dict[trigger_type]:4d} events "
                f"({self.__counting_dict[f'{trigger_type}_passed'] / self.__counting_dict[trigger_type] * 100:.2f} %)")

        self.logger.info(counts)


    #####################
    ###### Getters ######
    #####################

    def get_threshold_multiplier(self):
        """
        Returns
        -------
        self._threshold_multiplier: float
            High hit threshold multiplier (default: 6.5)
        """
        return self._threshold_multiplier

    def get_in_ice_channels(self):
        """
        Returns
        -------
        self._in_ice_channels: 1-D list of ints
            In-ice channel IDs in a list: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]
        """
        return self._in_ice_channels

    def get_in_ice_channel_groups(self):
        """
        Returns
        -------
        self._in_ice_channel_groups: Several 1-D lists of ints in a tuple
            In-ice channel groups in a tuple: ([0, 1, 2, 3], [9, 10], [23, 22], [8, 4])
        """
        return self._in_ice_channel_groups

    def get_channel_pairs_in_PA(self):
        """
        Returns
        -------
        self._channel_pairs_in_PA: Several 1-D lists of ints in a tuple
            Channel pairs of the PA in a tuple: ([0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3])
        """
        return self._channel_pairs_in_PA

    def get_envelope_max_time_index(self):
        """
        Returns
        -------
        self._envelope_max_time_index: 1-D numpy array of ints
            Hit time index of each channel
        """
        return self._envelope_max_time_index

    def get_envelope_max_time(self):
        """
        Returns
        -------
        self._envelope_max_time: 1-D numpy array of floats
            Hit time of each channel
        """
        return self._envelope_max_time

    def get_in_ice_channels_noise_RMS(self):
        """
        Returns
        -------
        self._noise_RMS: 1-D numpy array of floats
            Noise RMS values for channels
        """
        return self._noise_RMS

    def get_times(self):
        """
        Returns
        -------
        self._times: 2-D numpy array of floats
            Arrays of times for channels
        """
        return self._times

    def get_traces(self):
        """
        Returns
        -------
        self._traces: 2-D numpy array of floats
            Arrays of traces for channels
        """
        return self._traces

    def get_envelope_traces(self):
        """
        Returns
        -------
        self._envelope_traces: 2-D numpy array of floats
            Arrays of envelope traces for channels
        """
        return self._envelope_traces

    def get_hit_thresholds(self):
        """
        Returns
        -------
        self._hit_thresholds: 1-D numpy array of floats
            Hit threshold of each channel
        """
        return self._hit_thresholds

    def is_passed_hit_filter(self):
        """
        Returns
        -------
        self._passed_hit_filter: bool
            See if event passed the Hit Filter
        """
        return self._passed_hit_filter

    def is_passed_time_checker(self):
        """
        Returns
        -------
        self._passed_time_checker: bool
            See if event passed the time checker
        """
        return self._passed_time_checker

    def is_passed_hit_checker(self):
        """
        Returns
        -------
        self._passed_hit_checker: bool
            See if event passed the hit checker
        """
        if self._passed_hit_checker is not None:
            return self._passed_hit_checker
        else:
            raise NotImplementedError("Cannot call is_passed_hit_checker() when hit checking wasn't performed.")

    def is_over_hit_threshold(self):
        """
        Returns
        -------
        self._is_over_hit_threshold: 1-D numpy array of bools
            See if there's a high hit in each channel
        """
        if self._complete_hit_check:
            return self._is_over_hit_threshold
        else:
            raise NotImplementedError("Cannot call is_over_hit_threshold() when complete_hit_check is False.")

    def is_in_time_window(self):
        """
        Returns
        -------
        self._is_in_time_window: 2-D list of bools
            See if a channel pair is coincident in the group
        """
        if self._complete_time_check:
            return self._is_in_time_window
        else:
            raise NotImplementedError("Cannot call is_in_time_window() when complete_time_check is False.")

    def is_wanted_trigger_type(self):
        """
        Returns
        -------
        self.__is_wanted_trigger_type: bool
            When we want to select only events with low threshold triggers.
        """
        if self.__is_wanted_trigger_type is not None:
            return self.__is_wanted_trigger_type
        else:
            raise NotImplementedError("Cannot call is_wanted_trigger_type() when self.__is_wanted_trigger_type is None.")

    def is_in_time_window_PA(self):
        """
        Returns
        -------
        dict: dictionary of bools
            See if channel pairs are coincident or not in Group 0 (PA)
            In the dictionary, there are 6 pairs:
            (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
            To see if a pair is coincident one can do, for example:
            is_in_time_window_PA[(0,1)], then it will be True or False
        """
        dict = {}
        for i_pair, pair in enumerate(self._channel_pairs_in_PA):
            i = pair[0]
            j = pair[1]
            dict[(i,j)] = self._is_in_time_window[0][i_pair]
        return dict
