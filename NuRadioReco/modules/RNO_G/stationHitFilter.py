"""
Hit Filter Module (RNO-G specific, can be upgraded to more general use in the future)
The main purpose is to reject thermal events in data.
"""

from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import trace_utilities, units

import numpy as np
import logging

logger = logging.getLogger('NuRadioReco.RNO_G.stationHitFitter')

class stationHitFilter:

    def __init__(self, complete_time_check=False, complete_hit_check=False, time_window=10.0*units.ns, is_multi_thresholds=False, threshold_multipliers=[6.5, 6.0, 5.5, 5.0, 4.5]):
        """
        See if an event is thermal noise based on the coincidence window.

        It deals with all the in-ice channels (15 waveforms) for an RNO-G station,
        adjacent channels are put into groups: G1:(0,1,2,3); G2:(9,10); G3:(23,22); G4:(8,4).
        It applies the Hilbert Transform to each waveform and finds their maximum (hit)
        and get the time when the hit happens, then it checks to see if hits in each group
        are in the coincidence window. This time checker requires at least 1 coincident pair in G1
        and another coincident pair in any group. When the time check fails, the Hit Filter will
        see if there's any in-ice channel that has a high hit (maximum > 6.5*noise_RMS), which is the
        hit checker, and whenever there's a high hit it passed the Hit Filter.
        Options for the COMPLETENESS of checks:
        CASE I (default): complete_hit_check=False and complete_time_check=False
        It's faster, the time checker and the hit checker don't have to go through all channels
        unless nothing has been found. If an event passed the time checker already,
        then the hit checker would be skipped.
        CASE II: complete_hit_check=True and complete_time_check=True
        All groups and all channels will be checked no matter what,
        so it will take a little bit more time.

        Parameters
        ----------
        complete_time_check: bool (default: False)
            If users want the time checker to run through all channel groups
        complete_hit_check: bool (default: False)
            If users want the high hit checker to run through all channels
        time_window: float (default: 10.0*units.ns)
            Coincidence window for two adjacent channels
        is_multi_thresholds: bool (default: False)
            If users want to check with multiple hit thresholds
        threshold_multipliers: 1-D list (default: [6.5, 6.0, 5.5, 5.0, 4.5])
            Different hit threshold multipliers where a hit threshold is a multiplier times the noise RMS,
            if not using multiple thresholds only the first element (largest multiplier) is used
        """
        self._complete_time_check = complete_time_check
        self._complete_hit_check = complete_hit_check
        self._dT = time_window
        self._is_multi_thresholds = is_multi_thresholds

        if self._is_multi_thresholds:
            self._threshold_multipliers = threshold_multipliers
        else:
            self._threshold_multipliers = [threshold_multipliers[0]]

        self._n_thresholds = len(self._threshold_multipliers)

        self._in_ice_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]
        self._in_ice_channel_groups = ([0, 1, 2, 3], [9, 10], [23, 22], [8, 4])
        self._channel_pairs_in_PA = ([0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3])

        self._n_channels_in_ice = len(self._in_ice_channels)
        self._n_channel_pairs_in_PA = len(self._channel_pairs_in_PA)

        self._envelope_max_time_index = np.array([None] * self._n_channels_in_ice)
        self._envelope_max_time = np.array([None] * self._n_channels_in_ice)
        self._noise_RMS = np.array([None] * self._n_channels_in_ice)
        self._times = [[] for i in range(self._n_channels_in_ice)]
        self._traces = [[] for i in range(self._n_channels_in_ice)]
        self._envelope_traces = [[] for i in range(self._n_channels_in_ice)]
        self._hit_thresholds = [[None for _ in range(self._n_thresholds)] for i in range(self._n_channels_in_ice)]
        self._is_over_hit_threshold = [[None for _ in range(self._n_thresholds)] for i in range(self._n_channels_in_ice)]

        ### 2-D list ###
        # This list contains sub-lists for all groups.
        # Each sub-list contains bool(s) indicating the COINCIDENCE of channel pair(s) in each group
        # self._is_in_time_window[0] is the PA, it has 6 pairs (6 bools)
        # Initiating each element with None, it will later be replaced with bool when running the time checker
        self._is_in_time_window = []
        for i_group, group in enumerate(self._in_ice_channel_groups):
            if i_group == 0:
                self._is_in_time_window.append([None for _ in range(self._n_channel_pairs_in_PA)])
            else:
                self._is_in_time_window.append([None for _ in range(len(group)-1)])


    def _channel_mapping(self, channel_in):
        """
        Channel Mappings for specifically channels on Helper String 2

        The Hit Filter works only on all 15 in-ice channels,
        but it takes all 24 channels as the input to avoid any possible confusion on channel order.
        This function maps channels 21, 22, 23 to indices 12, 13, 14 in the output array
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

        See if there are at least 2 coincident pairs in time sequence in Group 1 (PA),
        and if there's only 1 pair in PA, then find the other pair in other groups;
        otherwise, the input event fails the time checker.

        Returns
        -------
        self._passed_time_checker: bool
            Event passes the time checker or not
        """
        self._passed_time_checker = False
        envelope_max_time = self.get_envelope_max_time()
        coincidences_in_time_sequence_PA = [False, False, False]

        for i_group, group in enumerate(self._in_ice_channel_groups):
            if i_group == 0:
                for i_channel_pair, channel_pair in enumerate(self._channel_pairs_in_PA):
                    hit_time_difference = abs(envelope_max_time[channel_pair[1]] - envelope_max_time[channel_pair[0]])
                    dT_multiplier = abs(channel_pair[1] - channel_pair[0])
                    self._is_in_time_window[i_group][i_channel_pair] = hit_time_difference <= dT_multiplier * self._dT

                    if channel_pair[0] == 0 and self._is_in_time_window[i_group][i_channel_pair]:
                        coincidences_in_time_sequence_PA[0] = True
                    elif channel_pair[0] == 1 and self._is_in_time_window[i_group][i_channel_pair]:
                        coincidences_in_time_sequence_PA[1] = True
                    elif channel_pair[0] == 2 and self._is_in_time_window[i_group][i_channel_pair]:
                        coincidences_in_time_sequence_PA[2] = True
                if sum(coincidences_in_time_sequence_PA) >= 2:
                    self._passed_time_checker = True
                    if not self._complete_time_check:
                        break
                elif not sum(coincidences_in_time_sequence_PA) and not self._complete_time_check:
                    break
            else:
                hit_time_difference = abs(envelope_max_time[self._channel_mapping(group[0])] - envelope_max_time[self._channel_mapping(group[1])])
                self._is_in_time_window[i_group][0] = hit_time_difference <= self._dT

                if self._is_in_time_window[i_group][0] and sum(coincidences_in_time_sequence_PA) >= 1:
                    self._passed_time_checker = True
                    if not self._complete_time_check:
                        break

        return self._passed_time_checker


    def _hit_checker(self):
        """
        Find a high hit (maximum > 6.5*noise_RMS) in all 15 in-ice channels.

        See if there's at least 1 high hit in all input channels,
        if yes then the event passes the hit checker.
        When multiple thresholds are used, the hit checker then becomes more lenient;
        for example, if there are two hits above the second-highest threshold but below the highest threshold,
        the event will pass the hit checker too (when is_multi_thresholds = True).

        Returns
        -------
        self._passed_hit_checker: bool
            Event passes the hit checker or not
        """
        self._passed_hit_checker = False
        envelopes = self.get_envelope_traces()
        hit_thresholds = self.get_hit_thresholds()
        n_counts_above_threshold = np.zeros(self._n_thresholds)

        for i in range(self._n_thresholds):
            for i_channel in range(self._n_channels_in_ice):
                self._is_over_hit_threshold[i_channel][i] = np.amax(envelopes[i_channel]) > hit_thresholds[i_channel][i]
                n_counts_above_threshold[i] += int(self._is_over_hit_threshold[i_channel][i])

            if n_counts_above_threshold[i] > i:
                self._passed_hit_checker = True

        return self._passed_hit_checker


    def begin(self):
        """(Unused)"""
        pass


    def set_up(self, set_of_traces, set_of_times, noise_RMS):
        """
        Set things up before passing to the Hit Filter.

        This setup function calculates the noise RMS,
        sets the hit threshold(s),
        gets the Hilbert envelope,
        and finds the time when the maximum happens (hit).

        Parameters
        ----------
        set_of_traces: 2-D array of floats
            A set of input trace arrays of all 24 channels
        set_of_times: 2-D array of floats
            A set of input times arrays of all 24 channels
        noise_RMS: 1-D array of floats
            A set of input noise RMS values of all 24 channels
        """
        self._passed_time_checker = None
        self._passed_hit_checker = None
        self._passed_hit_filter = None

        for i, channel in enumerate(self._in_ice_channels):
            if noise_RMS is not None:
                self._noise_RMS[i] = noise_RMS[channel]
            else:
                self._noise_RMS[i] = trace_utilities.get_split_trace_noise_RMS(set_of_traces[channel])

            self._hit_thresholds[i] = self._noise_RMS[i] * np.array(self.get_threshold_multipliers())

            self._traces[i] = set_of_traces[channel]
            self._envelope_traces[i] = trace_utilities.get_hilbert_envelope(set_of_traces[channel])
            self._times[i] = set_of_times[channel]
            self._envelope_max_time_index[i] = np.array(self._envelope_traces[i]).argmax()
            self._envelope_max_time[i] = self._times[i][self._envelope_max_time_index[i]]


    def apply_hit_filter(self):
        """
        See if the input event will survive the Hit Filter or not.

        After the setup, it first checks with the time checker,
        if event passed the time checker then it passes the Hit Filter;
        if event failed the time checker, then checks with the hit checker.
        """
        self._passed_hit_filter = False

        self._time_checker()

        if not self._complete_hit_check:
            if self.is_passed_time_checker():
                self._passed_hit_filter = True
                return self._passed_hit_filter
            else:
                self._hit_checker()
                if self.is_passed_hit_checker():
                    self._passed_hit_filter = True
        else:
            self._hit_checker()
            if self.is_passed_time_checker() or self.is_passed_hit_checker():
                self._passed_hit_filter = True

        return self._passed_hit_filter


    @register_run()
    def run(self, evt, station, det=None, noise_RMS=None):
        """
        Runs the Hit Filter.

        Parameters
        ----------
        evt: `NuRadioReco.framework.event.Event` | None
        station: `NuRadioReco.framework.station.Station`
            The station to use the Hit Filter
        det: Detector object | None
            Detector object (not used in this method,
            included to have the same signature as other NuRadio classes)
        noise_RMS: 1-D numpy array (default: None)
            Noise RMS values of all channels, if not given the Hit Filter will calculate them

        Returns
        -------
        self.is_passed_hit_filter(): bool
            Event passed the Hit Filter or not
        """
        set_of_traces = np.array([np.array(channel.get_trace()) for channel in station.iter_channels()])
        set_of_times = np.array([np.array(channel.get_times()) for channel in station.iter_channels()])
        self.set_up(set_of_traces, set_of_times, noise_RMS)
        self.apply_hit_filter()

        return self.is_passed_hit_filter()


    def end(self):
        """(Unused)"""
        pass


    #####################
    ###### Getters ######
    #####################

    def get_threshold_multipliers(self):
        """
        Returns
        -------
        np.array(self._threshold_multipliers): 1-D numpy array of floats
            Threshold multipliers (default: np.array([6.5, 6.0, 5.5, 5.0, 4.5]))
        """
        return np.array(self._threshold_multipliers)

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
            Time indices of hits for channels
        """
        return self._envelope_max_time_index

    def get_envelope_max_time(self):
        """
        Returns
        -------
        self._envelope_max_time: 1-D numpy array of floats
            Times of hits for channels
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
        np.array(self._times): 2-D numpy array of floats
            Arrays of times for channels
        """
        return np.array(self._times)

    def get_traces(self):
        """
        Returns
        -------
        np.array(self._traces): 2-D numpy array of floats
            Arrays of traces for channels
        """
        return np.array(self._traces)

    def get_envelope_traces(self):
        """
        Returns
        -------
        np.array(self._envelope_traces): 2-D numpy array of floats
            Arrays of envelope traces for channels
        """
        return np.array(self._envelope_traces)

    def get_hit_thresholds(self):
        """
        Returns
        -------
        np.array(self._hit_thresholds): 2-D numpy array of floats
            Arrays of hit thresholds for channels
        """
        return np.array(self._hit_thresholds)

    def is_in_time_window(self):
        """
        Returns
        -------
        self._is_in_time_window: 2-D list of bools
            See if channel pairs passed the time checker in all groups
        """
        if self._complete_time_check:
            return self._is_in_time_window

    def is_in_time_window_PA(self):
        """
        Returns
        -------
        dict: dictionary of bools
            See if channel pairs passed the time checker in group 1 (PA)
            In the dictionary, there are 6 pairs:
            (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
            To see if a pair is coincident one can do, for example:
            is_in_time_window_PA[(0,1)], then it will be True or False
        """
        dict = {}
        for pair in self._channel_pairs_in_PA:
            i = pair[0]
            j = pair[1]
            for k in self._is_in_time_window[0]:
                dict[(i,j)] = self._is_in_time_window[0][k]
        return dict

    def is_over_hit_threshold(self):
        """
        Returns
        -------
        self._is_over_hit_threshold: 2-D list of bools
            See if there are high hits in channels
        """
        if self._complete_hit_check:
            return self._is_over_hit_threshold

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
        return self._passed_hit_checker

    def is_passed_hit_filter(self):
        """
        Returns
        -------
        self._passed_hit_filter: bool
            See if event passed the Hit Filter
        """
        return self._passed_hit_filter
