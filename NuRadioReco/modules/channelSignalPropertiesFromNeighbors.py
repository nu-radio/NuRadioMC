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
        self.__require_region_in_trace = True  #  default of the new option
        self.__max_region_outside = 20.  #  default of the new option (ns)

    def begin(self, require_region_in_trace=True, max_region_outside=20.):  #  new options
        """
        Parameters
        ----------
        require_region_in_trace: bool
            Do not interpolate a region that lies outside the recorded trace.
        max_region_outside: float
            How far an interpolated region may extend beyond the recorded trace (ns).
        """
        self.__require_region_in_trace = require_region_in_trace  #  store the new option
        self.__max_region_outside = max_region_outside  #  store the new option

    @register_run()
    def run(self, event, station, detector, channel_groups):
        """
        Run the module

        Every signal region of a channel keeps its own entry (CHANGED 2026-10-20): the
        parameters signal_regions, signal_ray_type, signal_time_offset,
        signal_receiving_zenith, signal_receiving_azimuth, signal_region_snrs and
        pulse_times always have one entry per region, in the same order. Measured
        regions keep their order; interpolated regions are appended after them, with
        signal_region_snrs and pulse_times set to NaN because nothing was measured.

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
            channel_regions = [[] for _ in channel_group]  #  one list of regions per channel, one entry per REGION (was: one slot per ray type)
            for i_channel, channel_id in enumerate(channel_group):
                channel_depths[i_channel] = detector.get_relative_position(station.get_id(), channel_id)[2]
                channel = station.get_channel(channel_id)
                if not channel.has_parameter(chp.signal_regions):
                    continue
                ch_sig_regs = channel.get_parameter(chp.signal_regions)
                if len(ch_sig_regs) == 0:
                    continue
                n_regions = len(ch_sig_regs)
                ch_ray_types = np.atleast_1d(channel.get_parameter(chp.signal_ray_type)).astype(int)
                ch_time_offsets = np.atleast_1d(channel.get_parameter(chp.signal_time_offset))
                ch_zeniths = np.atleast_1d(channel.get_parameter(chp.signal_receiving_zenith))
                ch_azimuths = self.__per_region(channel, chp.signal_receiving_azimuth, n_regions, 0.)  #  azimuth is not always set (2D vertex)
                ch_snrs = self.__per_region(channel, chp.signal_region_snrs, n_regions, np.nan)  #  SNR of every region, NaN if not one per region
                ch_pulse_times = self.__per_region(channel, chp.pulse_times, n_regions, np.nan)  #  pulse time of every region, NaN if not one per region
                for i_region in range(n_regions):  #  every region is kept, also two of the same ray type
                    channel_regions[i_channel].append({
                        'region': np.array(ch_sig_regs[i_region], dtype=float),
                        'ray_type': int(ch_ray_types[i_region]),
                        'time_offset': ch_time_offsets[i_region],
                        'zenith': ch_zeniths[i_region],
                        'azimuth': ch_azimuths[i_region],
                        'snr': ch_snrs[i_region],
                        'pulse_time': ch_pulse_times[i_region],
                    })

            added_region = np.zeros(len(channel_group), dtype=bool)
            for ray_type in (1, 2, 3):
                # one point per channel for the fit: the measured region of this ray type; if a
                # channel has two of them, the one closest in time to the other channels' regions
                fit_points = self.__fit_points(channel_regions, ray_type)  #  one fit point per channel
                has_pulse_mask = np.array([point is not None for point in fit_points])
                if np.sum(has_pulse_mask) < 2:
                    continue
                depths = channel_depths[has_pulse_mask]
                points = [point for point in fit_points if point is not None]
                time_offset_fit = np.polyfit(depths, [p['time_offset'] for p in points], deg=1)
                zenith_fit = np.polyfit(depths, [p['zenith'] for p in points], deg=1)
                azimuth_fit = np.polyfit(depths, [p['azimuth'] for p in points], deg=1)
                lower_signal_region_fit = np.polyfit(depths, [p['region'][0] for p in points], deg=1)
                upper_signal_region_fit = np.polyfit(depths, [p['region'][1] for p in points], deg=1)
                for i_channel, channel_id in enumerate(channel_group):
                    if has_pulse_mask[i_channel]:
                        continue
                    channel = station.get_channel(channel_id)
                    depth = channel_depths[i_channel]
                    lower = depth * lower_signal_region_fit[0] + lower_signal_region_fit[1]
                    upper = depth * upper_signal_region_fit[0] + upper_signal_region_fit[1]
                    channel_times = channel.get_times()  #  time range of the recorded trace
                    if self.__require_region_in_trace and (lower < channel_times[0] - self.__max_region_outside or upper > channel_times[-1] + self.__max_region_outside):  #  region not in the trace
                        continue  #  do not invent it
                    channel_regions[i_channel].append({  #  interpolated region appended after the measured ones
                        'region': np.array([lower, upper]),
                        'ray_type': ray_type,
                        'time_offset': depth * time_offset_fit[0] + time_offset_fit[1],
                        'zenith': depth * zenith_fit[0] + zenith_fit[1],
                        'azimuth': depth * azimuth_fit[0] + azimuth_fit[1],
                        'snr': np.nan,  #  interpolated, nothing measured
                        'pulse_time': np.nan,  #  interpolated, nothing measured
                    })
                    added_region[i_channel] = True

            for i_channel, channel_id in enumerate(channel_group):
                if not added_region[i_channel]:
                    continue  # channels without an interpolated region are left exactly as they were
                channel = station.get_channel(channel_id)
                regions = channel_regions[i_channel]
                channel.set_parameter(chp.signal_regions, np.array([r['region'] for r in regions]))  #  all regions, one entry each
                channel.set_parameter(chp.signal_ray_type, np.array([r['ray_type'] for r in regions], dtype=int))  #  same order
                channel.set_parameter(chp.signal_time_offset, np.array([r['time_offset'] for r in regions]))  #  same order
                channel.set_parameter(chp.signal_receiving_zenith, np.array([r['zenith'] for r in regions]))  #  same order
                channel.set_parameter(chp.signal_receiving_azimuth, np.array([r['azimuth'] for r in regions]))  #  same order
                channel.set_parameter(chp.signal_region_snrs, np.array([r['snr'] for r in regions]))  #  same order, NaN for interpolated
                channel.set_parameter(chp.pulse_times, np.array([r['pulse_time'] for r in regions]))  #  same order, NaN for interpolated

    @staticmethod
    def __per_region(channel, parameter, n_regions, fill):  #  new helper
        """The parameter as one value per region, or `fill` for every region if that is not what is stored."""
        if channel.has_parameter(parameter):
            values = np.atleast_1d(channel.get_parameter(parameter)).astype(float)
            if len(values) == n_regions:
                return values
        return np.full(n_regions, fill)

    @staticmethod
    def __fit_points(channel_regions, ray_type):  #  new helper
        """
        For every channel the region of this ray type that goes into the straight-line fit,
        or None. A channel with one such region uses it. A channel with two or more (the time
        offset calculator gave two pulses the same ray type) uses the one whose centre is
        closest to the median centre of the channels that have exactly one; if there are
        none, the one with the largest SNR (NaN counts as lowest).
        """
        candidates = [[region for region in regions if region['ray_type'] == ray_type] for regions in channel_regions]
        unique_centres = [np.mean(c[0]['region']) for c in candidates if len(c) == 1]
        reference = np.median(unique_centres) if len(unique_centres) > 0 else None
        points = []
        for c in candidates:
            if len(c) == 0:
                points.append(None)
            elif len(c) == 1:
                points.append(c[0])
            elif reference is not None:
                points.append(min(c, key=lambda region: abs(np.mean(region['region']) - reference)))
            else:
                points.append(max(c, key=lambda region: -np.inf if np.isnan(region['snr']) else region['snr']))
        return points
