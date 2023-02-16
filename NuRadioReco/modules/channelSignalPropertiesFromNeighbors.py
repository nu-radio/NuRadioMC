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
        pass

    def begin(self):
        pass

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
            for i_channel, channel_id in enumerate(channel_group):
                channel_depths[i_channel] = detector.get_relative_position(station.get_id(), channel_id)[2]
                channel = station.get_channel(channel_id)
                if channel.has_parameter(chp.signal_regions):
                    ch_sig_regs = channel.get_parameter(chp.signal_regions)
                    if len(ch_sig_regs) > 0:
                        ch_ray_types = channel.get_parameter(chp.signal_ray_types).astype(int)
                        for i_region, signal_region in enumerate(ch_sig_regs):
                            channel_signal_time_offsets[i_channel, ch_ray_types[i_region] - 1] = channel.get_parameter(chp.signal_time_offsets)[i_region]
                            channel_signal_zeniths[i_channel, ch_ray_types[i_region] - 1] = channel.get_parameter(chp.signal_receiving_zeniths)[i_region]
                            channel_signal_azimuths[i_channel, ch_ray_types[i_region] - 1] = channel.get_parameter(chp.signal_receiving_azimuths)[i_region]
                            channel_signal_regions[i_channel, ch_ray_types[i_region] - 1] = channel.get_parameter(chp.signal_regions)[i_region]
                            channel_has_region[i_channel, ch_ray_types[i_region] - 1] = True
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
                            channel_signal_time_offsets[i_channel, ray_type] = channel_depths[i_channel] * time_offset_fit[0] + time_offset_fit[1]
                            channel_signal_zeniths[i_channel, ray_type] = channel_depths[i_channel] * zenith_fit[0] + zenith_fit[1]
                            channel_signal_azimuths[i_channel, ray_type] = channel_depths[i_channel] * azimuth_fit[0] + azimuth_fit[1]
                            channel_signal_regions[i_channel, ray_type, 0] = channel_depths[i_channel] * lower_signal_region_fit[0] + lower_signal_region_fit[1]
                            channel_signal_regions[i_channel, ray_type, 1] = channel_depths[i_channel] * upper_signal_region_fit[0] + upper_signal_region_fit[1]
                            channel_has_region[i_channel, ray_type] = True
                            signal_regions.append(channel_signal_regions[i_channel, ray_type])
                            added_region = True
                if added_region:
                    channel.set_parameter(chp.signal_regions, np.array(signal_regions))
                    channel.set_parameter(chp.signal_time_offsets, channel_signal_time_offsets[i_channel][channel_has_region[i_channel]])
                    channel.set_parameter(chp.signal_receiving_zeniths, channel_signal_zeniths[i_channel][channel_has_region[i_channel]])
                    channel.set_parameter(chp.signal_receiving_azimuths, channel_signal_azimuths[i_channel][channel_has_region[i_channel]])
                    channel.set_parameter(chp.signal_ray_types, np.arange(3, dtype=int)[channel_has_region[i_channel]] + 1)
