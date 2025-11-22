"""
Tools to reconstruct 'generic' impulsive signals

This module contains several helper functions and one class to reconstruct the arrival times
/ arrival direction of 'generic' impulsive signals (impulsive signals where the signal shape is not
known in advance, e.g. wind-induced events).

Overview
--------
`find_threshold_crossing`
    Finds the times where the hilbert envelope of a trace first crosses
    a given threshold.

`find_threshold_crossing_from_stft`
    Similar to `find_threshold_crossing`, but uses a short-time Fourier transform (STFT)
    to allow for power dispersion in the system. This function also supports the use
    of a coincidence window.

`get_dt_correlation`
    Determines the time lag between `channels <NuRadioReco.framework.channel.Channel>`
    using cross-correlation. Alternatively, the channels can be cross-correlated with
    a given template, instead.

`ImpulsiveSignalReconstructor`
    A class that implements a full plane-wave direction reconstruction
    using the above methods, with the usual ``run`` method.

"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.constants
import scipy.signal
import scipy.ndimage
import scipy.stats
from math import ceil
from NuRadioReco.utilities import units, ice, geometryUtilities
from NuRadioReco.framework.parameters import stationParameters
from NuRadioReco.modules.base.module import register_run
import radiotools.helper as hp
import logging
import warnings

logger = logging.getLogger('NuRadioReco.impulsiveSignalReconstructor')

SPEED_OF_LIGHT = scipy.constants.c * units.m / units.s # convert to NuRadio units

def find_threshold_crossing(channels, threshold=None, offset=5*units.ns, min_amp=0, debug=False):
    """
    Find the time where the hilbert envelope of a trace first crosses some threshold

    Parameters
    ----------
    channels : list of `NuRadioReco.framework.channel.Channel` objects
        the channels to use in the reconstruction
    threshold : float | str | None, optional

        * If a float, the threshold value.
        * If the string "max", the times of the hilbert envelope maxima are returned.
        * Otherwise, if not specified (default), attempts to estimate the noise
          distribution and sets the threshold to the 90th percentile.

    offset : float or list of floats, optional
        Time at the start and end of each trace to exclude.
        If a list, the entries are the offsets for the start and end of the trace,
        respectively. By default, the first and last 5 ns of each trace are excluded,
        to avoid spurious peaks in the hilbert envelope resulting from the assumed
        periodicity of the trace.
    min_amp : float, optional
        Minimum amplitude for a trace to be considered valid.
        If the maximum amplitude of a channel does not exceed ``min_amp``,
        no threshold crossing will be returned for this channel.
    debug : bool, default False
        If ``True``, create some debug plots

    Returns
    -------
    threshold_times : np.ndarray of floats
        An array with the threshold crossing times for each channel.
        For channels that do not exceed ``threshold`` (or ``min_amp``),
        this will be ``np.nan``.
    """
    offset = np.atleast_1d(offset)
    if len(offset) == 1:
        offset = (offset[0], offset[0])
    if debug:
        fig, axs = plt.subplots(3, 1, figsize=(4, 6))

    threshold_times = np.nan * np.zeros(len(channels))

    for i, channel in enumerate(channels):
        trace = channel.get_trace()
        sampling_rate = channel.get_sampling_rate()
        trace -= np.mean(trace)
        offset_samples = ceil(offset[0] * sampling_rate), ceil(offset[1] * sampling_rate)
        if offset_samples[1] == 0:
            offset_samples[1] = None # needed for correct numpy slicing

        hilbert_envelope = channel.get_hilbert_envelope_mag()
        if threshold == 'max':
            threshold_xing = np.argmax(hilbert_envelope[offset_samples[0]:-offset_samples[1]], keepdims=True)
        elif isinstance(threshold, str):
            raise ValueError(f'Argument `threshold` has value"{threshold}" but only a float or the string `max` are accepted.')
        else:
            current_threshold = threshold
            if threshold is None: # approximate the noise with a normal distribution
                # we take the 33% quantile to (hopefully) ignore the signal part of the trace
                # and then rescale this to obtain the corresponding 1-sigma of a normal distribution
                scale = np.sqrt(np.quantile(trace**2, 1/3)) / scipy.stats.norm.ppf(2/3, scale=1)
                # We take the 90th percentile of the distribution as follows:
                # we use the inverse CDF to determine the largest (positive) expected value in a trace with
                # 20x the actual trace length, meaning the expected number of threshold crossings within the
                # trace length is 1/20 (positive) + 1/20 (negative) = 1/10
                current_threshold = scipy.stats.norm.isf(0.05 / channel.get_number_of_samples(), scale=scale)
                logger.debug(f'Auto-threshold for channel {channel.get_id()} is {current_threshold/units.mV:.2f} mV')

            threshold_xing = np.where(
                hilbert_envelope[offset_samples[0]:-offset_samples[1]] > current_threshold
            )[0]

        if np.max(hilbert_envelope) < min_amp:
            logger.debug("Reject - max amp {:.2f} < {:.2f}".format(np.max(hilbert_envelope), min_amp))
            continue
        if len(threshold_xing) == 0:
            logger.debug(
                "Reject - no threshold crossing (max amp {:.2f} < {:.2f})".format(
                    np.max(hilbert_envelope[offset_samples[0]:-offset_samples[1]]), current_threshold))
            continue

        threshold_t_sample = threshold_xing[0] + offset_samples[0]

        threshold_times[i] = channel.get_times()[threshold_t_sample]

        if debug:
            axs[i].plot(channel.get_times(), hilbert_envelope)
            axs[i].plot(channel.get_times(), trace)
            axs[i].axvline(threshold_times[i], ls=":", color='red')
            axs[i].set_xlim(threshold_times[i] - 50, threshold_times[i] + 150)
            axs[i].set_ylim(-.1*threshold, 5*threshold)
            axs[i].set_ylabel('amplitude')

    if debug:
        if len(threshold_times)==len(channels):
            axs[-1].set_xlabel('Time [ns]')
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

    return np.asarray(threshold_times)

def find_threshold_crossing_from_stft(
        channels, max_delta_t=50*units.ns, passband=None, mode='psd',
        window=('tukey', 0.1), use_maximum=False, debug=False):
    """
    Find the start time of a pulse in multiple channels.

    Uses a short-time fourier transform (STFT) to effectively integrate the power across multiple frequency
    bins, and subsequently look for coincident threshold crossings across multiple channels.

    Parameters
    ----------
    channels : list of `NuRadioReco.framework.channel.Channel` objects
        the channels to use in the reconstruction
    max_delta_t : float, default: 50 * units.ns
        The coincidence window between all channels.
        Only threshold crossings that fall within a maximum time span of ``max_delta_t``
        will be selected.
    passband : tuple of 2 floats, optional
        The low- and high-pass frequencies of the passband
        to apply before correlating. If not provided,
        defaults to [60 MHz, 750 MHz]
    mode : str {'psd', 'amplitude'}, default: 'psd'
        Which metric to use in the STFT:

        * 'psd' : power (default)
        * 'amplitude' : amplitude

    window : str or tuple, optional
        Arguments to pass to `scipy.signal.get_window` to determine
        which window to use in the STFT. Defaults to ``('tukey', 0.1)``
    use_maximum : bool, default: False
        If True, considers only local maxima (instead of threshold crossing)
        to determine pulse times and coincidences.
    debug : bool or str, default: False
        If True, produce some debug plots and show them.
        If a string, should specify a path to save the debug plots to.

    Returns
    -------
    threshold_times: np.ndarray of floats
        An array with the threshold crossing times for each channel

    Notes
    -----
    This function will return the first threshold crossing but maximizes the number
    of channels included. E.g. if there is a 2-fold coincidence and a 3-fold coincidence
    at different times, the times of the 3-fold coincidence will be returned; if there are
    multiple possible 3-fold coincidences, only the first 3-fold coincidence will be returned.
    """
    crossings =  np.zeros(len(channels))
    n_samples_window = 64

    if passband is None:
        passband = [60*units.MHz, 750*units.MHz]
    if debug:
        fig, axs = plt.subplots(
            3, len(channels), figsize=(2*len(channels), 3.5),
            layout='constrained', sharex=True, sharey='row')
        fig.get_layout_engine().set(hspace=0, h_pad=0)

    crossings = []
    stft_plots = []
    trace_start_times = []

    for i, channel in enumerate(channels):
        f, t, stft_abs = scipy.signal.spectrogram(
            channel.get_trace(), channel.get_sampling_rate(),
            nperseg=n_samples_window, noverlap=n_samples_window-1,
            mode=mode, window=window)

        t_mid = len(t) // 2
        trace_start_times.append(channel.get_trace_start_time())
        stft_median = np.median(stft_abs[:,:t_mid], axis=1)
        stft_max = np.max(stft_abs[:,t_mid:], axis=1)
        fmask = (f > passband[0]) & (f < passband[1])
        max_over_med = stft_max / stft_median
        # we select only frequencies with a significant excess over the median,
        # defined here as SNR of 4 for magnitude or SNR of 10 for power
        # TODO: this is probably not optimal for all detectors / trace lengths
        if mode == 'magnitude':
            inclusion_threshold = 4
        else:
            inclusion_threshold = 10

        if np.sum(max_over_med[fmask] > inclusion_threshold) > 1: # at least 2 bins
            fmask &= max_over_med > inclusion_threshold
        else:
            fmask &= max_over_med >= np.sort(max_over_med[fmask])[-2]

        n_bins = np.sum(fmask)
        stftsum = np.sum(stft_abs[fmask], axis=0) / np.sum(stft_median[fmask])
        if mode == 'magnitude':
            threshold = 1 + 4/np.sqrt(n_bins-1) # empirical parameterization for the 99th percentile noise fluctuations
        elif mode == 'psd':
            threshold = 1 + 12/(n_bins-1)**(2/3)

        if use_maximum:
            # use a maximum filter with half the width of the window
            max_filter = scipy.ndimage.maximum_filter(stftsum[t_mid:], size=n_samples_window//2)
            mask_is_max = max_filter == stftsum[t_mid:] # this selects all the peaks
            # if np.any(max_filter > threshold):
            mask_is_max &= max_filter > threshold # only consider peaks above the threshold
            crossings.append(mask_is_max)
            if debug:
                axs[2,i].plot(
                    t[t_mid:][mask_is_max] + channel.get_trace_start_time(), max_filter[mask_is_max],
                    ls='', marker='x', color='k')
        else:
            # a threshold crossing implies sample i is below the threshold, but sample i+1 is above it
            above_threshold = stftsum[t_mid-1:] > threshold
            crossing = above_threshold[1:] * ~above_threshold[:-1]
            crossings.append(crossing)

        if debug:
            axs[0,i].plot(channel.get_times(), channel.get_trace(), color='k', lw=.5)
            stft_plots.append(stft_abs)
            axs[2,i].plot(t+channel.get_trace_start_time(), stftsum, 'r', lw=1)
            axs[2,i].axhline(threshold, color='k', ls=':')
            axs[2,i].set_yscale('log')
            axs[0,i].set_title(f'Channel {channel.get_id()}')
            axs[2,i].set_xlabel('Time [ns]')

    ## We need to account for potentially different trace start times
    ## we will do this by prepending/appending an appropriate number of zeros
    trace_start_times = np.asarray(trace_start_times)
    prepend_zeros = ((trace_start_times - min(trace_start_times)) * channel.get_sampling_rate()).astype(int)
    append_zeros = max(prepend_zeros) - prepend_zeros # also append zeros to ensure equal-length arrays

    crossings_sync = np.vstack([
        np.concatenate([np.zeros(prepend_zeros[i]), crossings[i], np.zeros(append_zeros[i])])
        for i in range(len(crossings))])

    # we check for coincidences within max_delta_t by applying a sliding window
    window_length = int(max_delta_t * channel.get_sampling_rate() + 1)
    sliding_view = np.lib.stride_tricks.sliding_window_view(crossings_sync, window_shape=(len(crossings_sync), window_length))
    window_index = np.argmax(np.sum(np.any(sliding_view, axis=-1), axis=-1)) # first window with maximum number of threshold crossings
    crossing_samples = window_index + t_mid + np.arange(window_length)
    crossings_indices = np.where(sliding_view[0, window_index], crossing_samples[None], np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN slice encountered')
        first_crossings = np.nanmin(crossings_indices, axis=-1) # will raise a non-blocking RuntimeWarning if one or more channels did not cross the threshold
    logger.debug(f'Threshold crossing (samples): {first_crossings}')
    crossing_times = (first_crossings) / channel.get_sampling_rate() + min(trace_start_times) + t[0] # convert to time
    logger.debug(f'Threshold crossing (time): {crossing_times}')

    if debug:
        vmin = np.quantile(stft_plots, .75)
        vmax = np.max(stft_plots)
        norm = matplotlib.colors.LogNorm(vmin=vmin/units.mV**2, vmax=vmax/units.mV**2)

        for i, stft_abs in enumerate(stft_plots):
            cax = axs[1,i].imshow(
                stft_abs/units.mV**2, aspect='auto', origin='lower', norm=norm,
                extent=(t[0]+channel.get_trace_start_time(), channel.get_trace_start_time()+t[-1], 0, f[-1]/units.MHz))
            axs[1,i].set_ylim(passband[0]/units.MHz, passband[1]/units.MHz)

        axs[1,0].set_ylabel('Frequency [MHz]')
        axs[0,0].set_ylabel('Voltage [V]')
        axs[2,0].set_ylabel('Spectral excess')
        for i in range(len(crossing_times)):
            for j in range(3):
                axs[j,i].axvline(crossing_times[i], color=['k','w','k'][j], ls=':')

        plt.colorbar(mappable=cax, ax=axs[:], label='Energy Spectral Density [$\mathrm{mV}^2/\mathrm{GHz}$]')

        if isinstance(debug, str):
            plt.savefig(debug)
            plt.close(fig)
        else:
            plt.show()

    return crossing_times


def get_dt_correlation(channels, pos, passband=None, n_index=1., templates=None, full_output=False):
    """
    Determines the time delay between channels using correlation

    Parameters
    ----------
    channels : list of `NuRadioReco.framework.channel.Channel` objects
    pos : list
        The positions of the ``channels``
    passband : tuple of 2 floats
        The low- and high-pass frequencies of the passband
        to apply before correlating. If not provided,
        defaults to [60 MHz, 750 MHz]
    n_index : float, optional
        Used to compute the maximum allowed time delay between two channels;
        time delays larger than the travel time of light are excluded. Default: 1.
    templates : None | 1dim or 2dim array of floats, optional
        If no template is provided, the channel traces are cross-correlated directly with each other.
        Otherwise, if a template is provided, the channel traces are correlated with the template.
        If multiple templates are provided, each channel is correlated with all templates, and
        the maximum value at each time step is used.
    full_output : bool, default: False
        If True, additionally return the correlation values for each channel.

    Returns
    -------
    dt : array of floats
        The time delays that maximize the correlation; has
        the same shape as ``channels``
    corr : array of floats, optional
        Only if ``full_output==True``. The correlation at ``dt``
        for each channel

    """
    if passband is None:
        passband = [0.06, 0.75]

    channel = channels[0]
    sampling_rate = channel.get_sampling_rate()

    ds_max = np.max([np.linalg.norm(pos[i]-pos[j]) for i in range(3) for j in range(3)])
    max_sample_delay = n_index * ds_max / SPEED_OF_LIGHT * sampling_rate
    corrs = []

    if templates is not None: # use templates to determine the pulse positions. This is slightly more involved...
        # the number of correlation samples to consider for each channel
        # To consider all possible combinations, this would have to be long enough to include all correlation samples,
        # but the memory requirement scales as n_correlation_samples**n_channels, so we pick something smaller.
        n_correlation_samples = 100

        if not hasattr(templates[0], '__len__'): # only one template
            templates = [templates]
        for channel in channels:
            corr = 0
            for template in templates:
                corr = np.maximum(
                    corr, np.abs(hp.get_normalized_xcorr(
                        channel.get_filtered_trace(passband, filter_type='butterabs', order=8),
                        template
                )))
            corrs.append(corr)

        corrs= np.array(corrs)
        channels_sorted = np.argsort(-np.max(corrs,axis=1)) # sort channels in decreasing order of maximum correlation
        corr_index = np.argsort(-corrs, axis=-1) # sort correlation indices in decreasing order of correlation
        # pos_xy = pos[channels_sorted[1:3]][:, :2] - pos[channels_sorted[0:1]][:, :2]
        dpos = pos[channels_sorted] - pos[channels_sorted[0:1]]

        i0 = corr_index[channels_sorted[0], :n_correlation_samples] # only look at n_correlation_samples highest correlation values for first channel

        for k1 in range(len(corrs[1]) // n_correlation_samples):
            i1 = corr_index[channels_sorted[1], k1*n_correlation_samples:(k1+1)*n_correlation_samples]
            shift1 = i1[None] - i0[:, None] + int(np.round((channels[channels_sorted[1]].get_trace_start_time() - channels[channels_sorted[0]].get_trace_start_time()) * sampling_rate))
            if np.any(np.abs(shift1) < max_sample_delay):
                break # if this takes more than one iteration, this method probably isn't working very well...

        for k2 in range(len(corrs[2]) // n_correlation_samples):
            i2 = corr_index[channels_sorted[2], k2*n_correlation_samples:(k2+1)*n_correlation_samples]


            i = np.meshgrid(i0, i1, i2, indexing='ij')
            t1 = (i[1] - i[0]) / sampling_rate + channels[channels_sorted[1]].get_trace_start_time() - channels[channels_sorted[0]].get_trace_start_time()
            t2 = (i[2] - i[0]) / sampling_rate + channels[channels_sorted[2]].get_trace_start_time() - channels[channels_sorted[0]].get_trace_start_time()

            ds = SPEED_OF_LIGHT / n_index * np.array([t1.flatten(), t2.flatten()])

            # We need to ensure a plane wave solution exists.
            # We use the SVD to determine the 2D solution vector (in some unitarily transformed coordinate system)
            # and check this has norm < 1, which guarantees a 3D plane wave direction with norm 1 exists.
            u, s, _ = np.linalg.svd(dpos[1:])
            valid = np.linalg.norm(np.diag(1/s) @ u.T @ ds, axis=0) < 1

            if np.any(valid):
                total_corr = corrs[channels_sorted[0], i0][:, None, None] + corrs[channels_sorted[1], i1][None, :, None] + corrs[channels_sorted[2], i2][None, None, :]
                total_corr = total_corr.flatten()
                total_corr[~valid] = 0
                max_index = np.argmax(total_corr.flatten())

                mesh_index = np.unravel_index(max_index, i[0].shape)
                t0 = i[0][mesh_index] / sampling_rate + channels[channels_sorted[0]].get_trace_start_time()
                t1_max = t1[mesh_index] + t0
                t2_max = t2[mesh_index] + t0

                dt = np.zeros(3)
                dt[channels_sorted] = np.array([t0, t1_max, t2_max])
                if full_output:
                    return dt, np.array([corrs[channels_sorted[i], mesh_index[i]] for i in range(3)])
                return dt

    else: # no templates provided - we use the direct cross-correlation between the channels instead
        for i in range(len(channels)):
            for j in range(i+1, len(channels)):
                corr = np.abs(hp.get_normalized_xcorr(
                    channels[i].get_filtered_trace(passband, filter_type='butterabs', order=8),
                    channels[j].get_filtered_trace(passband, filter_type='butterabs', order=8)
                ))
                corrs.append(corr)

        corrs = np.array(corrs)
        lags = scipy.signal.correlation_lags(len(channel.get_trace()), len(channel.get_trace()))

        lags_valid = lags[np.abs(lags) < max_sample_delay]
        sample_delays = np.meshgrid(lags_valid, lags_valid)
        sample_delays.append(sample_delays[1]-sample_delays[0])
        sample_delays = np.array(sample_delays).reshape(3,-1)

        # we require that a plane wave solution exists
        # if the non-singular part of the equation has a solution with norm < 1,
        # this guarantees that there exists a full 3D plane wave solution with norm 1
        dpos = pos - pos[0:1]
        ds = SPEED_OF_LIGHT / n_index * sample_delays[:len(channels)-1] / sampling_rate
        u, s, _ = np.linalg.svd(dpos[1:])
        valid = np.linalg.norm(np.diag(1/s) @ u.T @ ds, axis=0) < 1

        sample_delays = sample_delays[:,valid] - lags[0]
        total_corr = np.sum([corrs[i,j] for i, j in enumerate(sample_delays)], axis=0)
        max_index = np.argmax(total_corr)
        time_shifts = (sample_delays[:, max_index] + lags[0]) / sampling_rate
        dt = np.array([0, *-time_shifts[:len(channels)-1]])

        if full_output:
            return dt, np.array([corrs[i, j] for i,j in enumerate(sample_delays[:, max_index])])

        return dt

class ImpulsiveSignalReconstructor():

    def __init__(self):
        """
        Class to reconstruct the direction of impulsive signals
        """
        pass

    def begin(self):
        """Unused"""

    @register_run()
    def run(self, evt, station, det, use_channels, method='stft', **kwargs):
        """
        Run the direction reconstruction

        Parameters
        ----------
        event : Event
            Event to reconstruct (unused)
        station : Station
            Station that contains the channels to use in the reconstruction
        det : Detector
            Detector description
        use_channels : list
            List of channel ids to use for the reconstruction
        method : str, default='stft'
            Which method to use to estimate the pulse arrival times:

            * ``'simple_threshold'``: use simple threshold crossings,
              using the `find_threshold_crossing` function;
            * ``'xcorr'``: use pairwise cross-correlation or
              correlation with a template, using
              `get_dt_correlation`;
            * ``'stft'`` (default): uses an short-time Fourier transform (STFT)
              approach to identify the start of the pulse

        **kwargs
            Additional keyword arguments passed to the function that
            determines the pulse arrival times (see ``method``)

        Returns
        -------
        zenith : float
            The reconstructed zenith
        azimuth : float
            The reconstructed azimuth

        Notes
        -----
        The reconstructed parameters are stored in the ``station`` passed to the run function.
        Note also that an analytic plane wave fit is used: this means

        1. if <= 3 channels are used, always the solution coming from above is returned,
           rather than the one coming from below;
        2. If no analytic solution exists for the inferred pulse arrival times,
           the reconstruction will fail
        """

        channels = [station.get_channel(channel_id) for channel_id in use_channels]
        pos = np.array([
            det.get_relative_position(station.get_id(), channel_id)
            for channel_id in use_channels
        ])

        if method == 'stft':
            if not 'max_delta_t' in kwargs: # use a simple estimate
                max_delta_t = np.max([
                    np.linalg.norm(pi-pj) * ice.get_refractive_index(min(pi[2], pj[2]), det.get_site(station.get_id()))
                    for pi in pos for pj in pos]) / SPEED_OF_LIGHT
                dt = find_threshold_crossing_from_stft(
                    channels, max_delta_t=max_delta_t, **kwargs
                )
            else:
                dt = find_threshold_crossing_from_stft(channels, **kwargs)
        elif method == 'xcorr':
            dt = get_dt_correlation(channels, pos, **kwargs)
        elif method == 'simple_threshold':
            dt = find_threshold_crossing(channels, **kwargs)
        zenith, azimuth = geometryUtilities.analytic_plane_wave_fit(dt, pos)

        if np.isnan(zenith):
            return

        station[stationParameters.zenith] = zenith
        station[stationParameters.azimuth] = azimuth

        return zenith, azimuth

    def end(self):
        """Unused"""
