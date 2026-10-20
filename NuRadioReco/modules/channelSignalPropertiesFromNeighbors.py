import numpy as np
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.modules.base.module import register_run


class channelSignalPropertiesFromNeighbors:
    """
    Sets signal properties, like timing, arrival direction and whether a signal is likely present in the first place.
    It does this by extrapolating these values from the neighboring channels.
    This module requires either the channelTimeOffsetCalculator to be run beforehand or the channelParameters
    signal_time_offsets, signal_receiving_zeniths, signal_receiving_azimuths and signal_regions to be set to reasonable
    values some other way.
    """
    def __init__(self):
        self.__fix_region_bookkeeping = True
        self.__require_region_in_trace = True
        self.__fix_snr_bookkeeping = True
        self.__max_region_outside = 20.
        self.n_dropped_outside_trace = 0

    def begin(self, fix_region_bookkeeping=True, require_region_in_trace=True, fix_snr_bookkeeping=True, max_region_outside=20.):
        """
        Parameters
        ----------
        fix_region_bookkeeping: bool
            CHECK 1. The original writes chp.signal_regions with ONLY the regions
            it interpolated, discarding the measured ones, while signal_ray_type,
            signal_time_offset and the two angles keep all of them. The IFT pairs
            regions with ray types BY INDEX, so once the lengths differ it windows
            the trace onto the wrong pulse. Measured over 1278 channel entries the
            four other arrays agree in length 100% of the time and signal_regions
            only 85.7%, and in 180 of 180 mismatched channels the region treated
            as the direct ray sits on the secondary pulse.
        require_region_in_trace: bool
            CHECK 2. Do not invent a region that falls outside the recorded trace.
            The interpolation is a straight-line fit against channel depth with no
            test that anything is there - in run 5195 it placed 'direct' regions
            at ~8118 ns for every channel while the traces start at 8381 ns, so
            the IFT windowed those channels to a range containing no samples.
        fix_snr_bookkeeping: bool
            CHECK 3. signal_region_snrs is written by the pulse finder, one entry
            per MEASURED region in the order the pulses were found. When this
            module adds a region it rewrites signal_ray_type sorted by ray type,
            but leaves signal_region_snrs alone, so entry i of one no longer
            belongs to entry i of the other - and the energy reconstructor picks
            the ray type to use by pairing exactly those two by index. With this
            flag the SNRs are rewritten in the same order as signal_ray_type,
            with 0 for a region that was interpolated rather than measured.
        """
        self.__fix_snr_bookkeeping = fix_snr_bookkeeping
        # ns (the NuRadioReco base unit); how far a region may stick out of the trace
        self.__max_region_outside = max_region_outside
        self.__fix_region_bookkeeping = fix_region_bookkeeping
        self.__require_region_in_trace = require_region_in_trace
        self.n_dropped_outside_trace = 0

    @register_run()
    def run(self, event, station, detector, channel_groups):
        """
        Run the module

        Parameters
        ----------
        event: NuRadioReco.framework.event.Event object
            The event the module should be run on
        station: NuRadioReco.framework.station.Station object
            The station the module should be run on
        detector: NuRadioReco.detector.detector.Detector object of child object
            The detector description
        channel_groups: 2D array of integers
            A list of groups of channels. In each group, a linear fit will be done to the signal arrival time and
            direction and the position of the signal search windows of those channels where they are specified.
            These fits will then be used to fill in these values for channels where they are missing.

        """
        for i_group, channel_group in enumerate(channel_groups):
            channel_depths = np.zeros(len(channel_group))
            channel_signal_time_offsets = np.zeros((len(channel_group), 3))
            channel_signal_zeniths = np.zeros((len(channel_group), 3))
            channel_signal_azimuths = np.zeros((len(channel_group), 3))
            channel_signal_regions = np.zeros((len(channel_group), 3, 2))
            channel_has_region = np.zeros((len(channel_group), 3), dtype=bool)
            channel_region_snrs = np.zeros((len(channel_group), 3))
            channel_has_snrs = np.zeros(len(channel_group), dtype=bool)
            for i_channel, channel_id in enumerate(channel_group):
                channel_depths[i_channel] = detector.get_relative_position(station.get_id(), channel_id)[2]
                channel = station.get_channel(channel_id)
                if channel.has_parameter(chp.signal_regions):
                    ch_sig_regs = channel.get_parameter(chp.signal_regions)
                    if len(ch_sig_regs) > 0:
                        ch_ray_types = channel.get_parameter(chp.signal_ray_type).astype(int)
                        ch_snrs = None
                        if channel.has_parameter(chp.signal_region_snrs):
                            ch_snrs = np.atleast_1d(channel.get_parameter(chp.signal_region_snrs))
                            # only usable while it still pairs with the regions
                            if len(ch_snrs) != len(ch_sig_regs):
                                ch_snrs = None
                        channel_has_snrs[i_channel] = ch_snrs is not None
                        for i_region, signal_region in enumerate(ch_sig_regs):
                            channel_signal_time_offsets[i_channel, ch_ray_types[i_region] - 1] = channel.get_parameter(chp.signal_time_offset)[i_region]
                            channel_signal_zeniths[i_channel, ch_ray_types[i_region] - 1] = channel.get_parameter(chp.signal_receiving_zenith)[i_region]
                            channel_signal_azimuths[i_channel, ch_ray_types[i_region] - 1] = channel.get_parameter(chp.signal_receiving_azimuth)[i_region]
                            channel_signal_regions[i_channel, ch_ray_types[i_region] - 1] = channel.get_parameter(chp.signal_regions)[i_region]
                            channel_has_region[i_channel, ch_ray_types[i_region] - 1] = True
                            if ch_snrs is not None:
                                channel_region_snrs[i_channel, ch_ray_types[i_region] - 1] = ch_snrs[i_region]
            for i_channel, channel_id in enumerate(channel_group):
                channel = station.get_channel(channel_id)
                signal_regions = []
                added_region = False
                for ray_type in range(3):
                    has_pulse_mask = channel_has_region[:, ray_type]
                    if np.sum(np.ones(len(channel_group))[has_pulse_mask]) >= 2:
                        time_offset_fit = np.polyfit(channel_depths[has_pulse_mask], channel_signal_time_offsets[:, ray_type][has_pulse_mask], deg=1)
                        zenith_fit = np.polyfit(channel_depths[has_pulse_mask], channel_signal_zeniths[:, ray_type][has_pulse_mask], deg=1)
                        azimuth_fit = np.polyfit(channel_depths[has_pulse_mask], channel_signal_azimuths[:, ray_type][has_pulse_mask], deg=1)
                        lower_signal_region_fit = np.polyfit(channel_depths[has_pulse_mask], channel_signal_regions[:, ray_type, 0][has_pulse_mask], deg=1)
                        upper_signal_region_fit = np.polyfit(channel_depths[has_pulse_mask], channel_signal_regions[:, ray_type, 1][has_pulse_mask], deg=1)
                        x_coords = np.arange(np.min(channel_depths), np.max(channel_depths))
                        if not channel_has_region[i_channel, ray_type]:
                            lower = channel_depths[i_channel] * lower_signal_region_fit[0] + lower_signal_region_fit[1]
                            upper = channel_depths[i_channel] * upper_signal_region_fit[0] + upper_signal_region_fit[1]

                            # CHECK 2: refuse a region the recorded trace cannot contain
                            if self.__require_region_in_trace:
                                channel_times = channel.get_times()
                                # the region may stick out of the trace by at most
                                # max_region_outside; an overlap at the edge is not a pulse
                                if (lower < channel_times[0] - self.__max_region_outside
                                        or upper > channel_times[-1] + self.__max_region_outside):
                                    self.n_dropped_outside_trace += 1
                                    continue

                            channel_signal_time_offsets[i_channel, ray_type] = channel_depths[i_channel] * time_offset_fit[0] + time_offset_fit[1]
                            channel_signal_zeniths[i_channel, ray_type] = channel_depths[i_channel] * zenith_fit[0] + zenith_fit[1]
                            channel_signal_azimuths[i_channel, ray_type] = channel_depths[i_channel] * azimuth_fit[0] + azimuth_fit[1]
                            channel_signal_regions[i_channel, ray_type, 0] = lower
                            channel_signal_regions[i_channel, ray_type, 1] = upper
                            channel_has_region[i_channel, ray_type] = True
                            signal_regions.append(channel_signal_regions[i_channel, ray_type])
                            added_region = True
                if added_region:
                    # CHECK 1: write ALL regions this channel has, in the same
                    # order as the four arrays below (all indexed by
                    # channel_has_region), so signal_regions and signal_ray_type
                    # stay index-consistent for the IFT's pairing loop.
                    if self.__fix_region_bookkeeping:
                        channel.set_parameter(
                            chp.signal_regions,
                            channel_signal_regions[i_channel][channel_has_region[i_channel]]
                        )
                    else:
                        channel.set_parameter(chp.signal_regions, np.array(signal_regions))
                    channel.set_parameter(chp.signal_time_offset, channel_signal_time_offsets[i_channel][channel_has_region[i_channel]])
                    channel.set_parameter(chp.signal_receiving_zenith, channel_signal_zeniths[i_channel][channel_has_region[i_channel]])
                    channel.set_parameter(chp.signal_receiving_azimuth, channel_signal_azimuths[i_channel][channel_has_region[i_channel]])
                    channel.set_parameter(chp.signal_ray_type, np.arange(3, dtype=int)[channel_has_region[i_channel]] + 1)
                    # CHECK 3: keep the SNRs in the same order as signal_ray_type
                    if self.__fix_snr_bookkeeping and channel_has_snrs[i_channel]:
                        channel.set_parameter(
                            chp.signal_region_snrs,
                            channel_region_snrs[i_channel][channel_has_region[i_channel]]
                        )
