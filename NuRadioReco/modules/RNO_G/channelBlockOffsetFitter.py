"""
Module to remove 'block offsets' from RNO-G voltage traces.

The function `fit_block_offsets` can be used standalone to perform an out-of-band
fit to the block offsets. Alternatively, the `channelBlockOffsets` class contains convenience
``add_offsets`` (to add block offsets in simulation) and ``remove_offsets`` methods that can be run
directly on a NuRadioMC/imported ``Event``. The added/removed block offsets are stored per channel
in the `NuRadioReco.framework.parameters.channelParameters.block_offsets` parameter.

"""

from NuRadioReco.utilities import units, fft
from NuRadioReco.framework.base_trace import BaseTrace
from NuRadioReco.framework.parameters import channelParameters
from NuRadioReco.modules.base.module import register_run

import numpy as np
import scipy.optimize
import logging

logger = logging.getLogger('NuRadioReco.RNO_G.channelBlockOffsetFitter')

class channelBlockOffsets:

    def __init__(self, block_size=128, max_frequency=51*units.MHz):
        """
        Add or remove block offsets to channel traces

        This module adds, fits or removes 'block offsets' by fitting
        them in a specified out-of-band region in frequency space.

        Parameters
        ----------
        block_size: int (default: 128)
            The size (in samples) of the blocks
        max_frequency: float (default: 51 MHz)
            The maximum frequency to include in the out-of-band
            block offset fit

        """
        self.sampling_rate = None
        self.block_size = block_size # the size (in samples) of the blocks
        self._offset_fit = dict()
        self._offset_inject = dict()
        self._max_frequency = max_frequency

    def add_offsets(self, event, station, offsets=1*units.mV, channel_ids=None):
        """
        Add (simulated or reconstructed) block offsets to an event.

        Added block offsets for each channel are stored in the
        ``channelParameters.block_offsets`` parameter.

        Parameters
        ----------
        event : `NuRadioReco.framework.event.Event` | None
        station : `NuRadioReco.framework.station.Station`
            The station to add block offsets to
        offsets: float | array | dict
            offsets to add to the event. Default: 1 mV

            - if a float, add gaussian-distributed of amplitude ``offsets``
              to all channels specified;
            - if an array, the length should be the same as the number
              of blocks in a single trace, and the entries will be
              interpreted as the amplitudes of the offsets;
            - if a dict, the keys should be the channel ids, and each
              value should contain either a float or an array to add to
              each channel as specified above.

        channel_ids: list | None
            either a list of channel ids to apply the offsets to, or
            None to apply the offsets to all channels in the station
            (default: None).

        """

        if channel_ids is None:
            channel_ids = station.get_channel_ids()
        for channel_id in channel_ids:
            channel = station.get_channel(channel_id)
            if isinstance(offsets, dict):
                add_offsets = offsets[channel_id]
            else:
                add_offsets = offsets
            if len(np.atleast_1d(add_offsets)) == 1:
                add_offsets = np.random.normal(
                    0, add_offsets, (channel.get_number_of_samples() // self.block_size)
                )

            # save the added offsets as a channelParameter
            if channel.has_parameter(channelParameters.block_offsets):
                block_offsets_old = channel.get_parameter(channelParameters.block_offsets)
                channel.set_parameter(channelParameters.block_offsets, block_offsets_old + add_offsets)
            else:
                channel.set_parameter(channelParameters.block_offsets, add_offsets)

            channel.set_trace(
                channel.get_trace() + np.repeat(add_offsets, self.block_size),
                channel.get_sampling_rate()
            )

    def remove_offsets(self, event, station, mode='auto', channel_ids=None, maxiter=5):
        """
        Remove block offsets from an event

        Fits and removes the block offsets from an event. The removed
        offsets are stored in the ``channelParameters.block_offsets``
        parameter.

        Parameters
        ----------
        event : `NuRadioReco.framework.event.Event` | None
        station : `NuRadioReco.framework.station.Station`
            The station to remove the block offsets from
        mode: str {'auto', 'fit', 'approximate', 'stored', 'median'}, optional

            - 'fit': fit the block offsets with a minimizer
            - 'approximate' : use the first guess from the out-of-band component,
              without any fitting (slightly faster)
            - 'auto' (default): decide automatically between 'approximate' and 'fit'
              based on the estimated size of the block offsets.
            - 'stored': use the block offsets already stored in the
              ``channelParameters.block_offsets`` parameter. Will raise an error
              if this parameter is not present.
            - 'median': remove the block offsets by calculating the median
              of each block. This is faster than fitting, but less accurate.
              Not recommended!

        channel_ids: list | None
            List of channel ids to remove offsets from. If None (default),
            remove offsets from all channels in `station`
        maxiter: int, default 5
            (Only if mode=='fit') The maximum number of fit iterations.
            This can be increased to more accurately remove the block offsets
            at the cost of performance. (The default value removes 'most' offsets
            to about 1%)

        See Also
        --------
        run : alias of this method
        """
        if channel_ids  is None:
            channel_ids = station.get_channel_ids()

        offsets = {}
        if mode == 'stored': # remove offsets stored in channelParameters.block_offsets
            offsets = {
                channel_id: -station.get_channel(channel_id).get_parameter(channelParameters.block_offsets)
                for channel_id in channel_ids}
        else: # fit & remove offsets
            for channel_id in channel_ids:
                channel = station.get_channel(channel_id)
                trace = channel.get_trace()

                if mode == "median":
                    block_offsets = _calculate_block_offsets(
                        trace, block_size=self.block_size, func=np.median
                    )
                else:
                    block_offsets = fit_block_offsets(
                        trace, self.block_size,
                        channel.get_sampling_rate(), self._max_frequency,
                        mode=mode, maxiter=maxiter
                    )

                offsets[channel_id] = -block_offsets

        self.add_offsets(event, station, offsets, channel_ids)

    def begin(self):
        """(Unused)"""
        pass

    @register_run()
    def run(self, event, station, det=None, mode='auto', channel_ids=None, **kwargs):
        """
        Remove the block offsets from all channels of a station.

        Fits and removes the block offsets from an event. The removed offsets
        are stored in the ``channelParameters.block_offsets``
        parameter.

        This method is an alias of `remove_offsets`, with the only difference the inclusion
        of the (unused) `det` parameter, to be consistent with the `run` methods of other
        NuRadio classes.

        Parameters
        ----------
        event : `NuRadioReco.framework.event.Event` | None
        station : `NuRadioReco.framework.station.Station`
            The station to remove the block offsets from
        det : Detector object, optional
            Detector object (not used in this method,
            included to have the same signature as other NuRadio classes)
        mode : str {'auto', 'fit', 'approximate', 'stored', 'median'}, optional

            - 'fit': fit the block offsets with a minimizer
            - 'approximate' : use the first guess from the out-of-band component,
              without any fitting (slightly faster)
            - 'auto' (default): decide automatically between 'approximate' and 'fit'
              based on the estimated size of the block offsets.
            - 'stored': use the block offsets already stored in the
              ``channelParameters.block_offsets`` parameter. Will raise an error
              if this parameter is not present.
            - 'median': remove the block offsets by calculating the median
              of each block. This is faster than fitting, but less accurate.
              Not recommended to be used!

        channel_ids : list | None
            List of channel ids to remove offsets from. If None (default),
            remove offsets from all channels in `station`.
        **kwargs : keyword arguments
            Other keyword arguments to be passed to the `remove_offsets` function.

        See Also
        --------
        remove_offsets : alias of this method without the (unused) `det` parameter
        """
        self.remove_offsets(event, station, mode=mode, channel_ids=channel_ids, **kwargs)


    def end(self):
        """(Unused)"""
        pass


def fit_block_offsets(
        trace, block_size=128, sampling_rate=3.2*units.GHz,
        max_frequency=50*units.MHz, mode='auto', return_trace=False,
        maxiter=5, tol=1e-6):
    """
    Fit 'block' offsets for a voltage trace

    Fit block offsets ('rect'-shaped offsets from a baseline)
    using a fit to the out-of-band spectrum of a voltage trace.

    Parameters
    ----------
    trace: numpy Array
        the voltage trace
    block_size: int (default: 128)
        the number of samples in one block
    sampling_rate: float (default: 3.2 GHz)
        the sampling rate of the trace
    max_frequency: float (default: 50 MHz)
        the fit to the block offsets is performed
        in the frequency domain, in the band up to
        max_frequency
    mode : str {'auto', 'fit', 'approximate'}, optional
        Whether to fit the block offsets
        or just use the first guess from the out-of-band
        component (faster). By default ('auto'), decide
        automatically based on the size of the block offsets
        (only fit if the largest block offset exceeds 50% of the Vrms).
    return_trace: bool (default: False)
        if True, return the tuple (offsets, output_trace)
        where the output_trace is the input trace with
        fitted block offsets removed
    maxiter: int (default: 5)
        (Only if mode=='fit') The maximum number of fit iterations.
        This can be increased to more accurately remove the block offsets
        at the cost of performance. (The default value removes 'most' offsets
        to about 1%)

    Returns
    -------
    block_offsets: numpy array
        The fitted block offsets.
    output_trace: numpy array or None
        The input trace with the fitted block offsets removed.
        Returned only if return_trace=True

    Other Parameters
    ----------------
    tol: float (default: 1e-6)
        tolerance parameter passed on to scipy.optimize.minimize

    See Also
    --------
    channelBlockOffsets :
        Class that uses this function to automatically remove the block offsets for all
        channels in a station.
    """
    dt = 1. / sampling_rate
    spectrum = fft.time2freq(trace, sampling_rate)
    frequencies = np.fft.rfftfreq(len(trace), dt)
    n_blocks = len(trace) // block_size

    mask = (frequencies > 0) & (frequencies < max_frequency) #  a simple rectangular filter
    frequencies_oob = frequencies[mask]
    spectrum_oob = spectrum[mask]

    # we use the bandpass-filtered trace to get a first estimate of
    # the block offsets, by simply averaging over each block.
    filtered_trace_fft = np.copy(spectrum)
    filtered_trace_fft[~mask] = 0
    filtered_trace = fft.freq2time(filtered_trace_fft, sampling_rate)

    # obtain guesses for block offsets
    a_guess = np.mean(np.split(filtered_trace, n_blocks), axis=1)

    if mode == 'approximate':
        perform_fit = False
    elif mode == 'fit':
        perform_fit = True
    elif mode == 'auto':
        # continue to fitting step only if the largest block offset is more than 20% of the Vrms
        max_offset = np.max(np.abs(a_guess))
        vrms = np.std(trace)
        perform_fit = max_offset > 0.5 * vrms
        if perform_fit:
            logger.warning("Trace has large block offsets (>{:.0f}% of Vrms), removing by fitting.".format(100 * max_offset / vrms))
    else:
        raise ValueError(f'Invalid value for mode={mode}. Accepted values are {{"fit", "approximate"}}')

    if not perform_fit: # just return the first guess
        block_offsets = a_guess + np.mean(trace)
    else:
        # we can get rid of one parameter through a global shift
        a_guess = a_guess[:-1] - a_guess[-1]

        # we perform the fit out-of-band, in order to avoid
        # distorting any actual signal

        # most of the terms in the fit depend only on the frequencies,
        # sampling rate and number of blocks. We therefore calculate these
        # only once, outside the fit function.
        pre_factor_exponent = np.array([
            -2.j * np.pi * frequencies_oob * dt * ((j+.5) * block_size - .5)
                for j in range(len(a_guess))
        ])
        const_fft_term = (
                1 / sampling_rate * np.sqrt(2) # NuRadio FFT normalization
            * np.exp(pre_factor_exponent)
            * np.sin(np.pi * frequencies_oob * block_size * dt)[None]
            / np.sin(np.pi * frequencies_oob * dt)[None]
        )

        def pedestal_fit(a):
            fit = np.sum(a[:, None] * const_fft_term, axis=0)
            chi2 = np.sum(np.abs(fit - spectrum_oob)**2)
            return chi2

        res = scipy.optimize.minimize(pedestal_fit, a_guess, tol=tol, options=dict(maxiter=maxiter)).x
        logger.debug(
            "Fit shifted estimated block offsets by {:.2f} ({:.0f}%)".format(
                np.median(res - a_guess), 100 * np.median((res - a_guess) / res)))

        # the fit is not sensitive to an overall shift,
        # so we include the zero-meaning here
        block_offsets = np.zeros(len(res) + 1)
        block_offsets[:-1] = res
        block_offsets += np.mean(trace) - np.mean(block_offsets)


    if return_trace:
        output_trace = trace - np.repeat(block_offsets, block_size)
        return block_offsets, output_trace

    return block_offsets


def _calculate_block_offsets(traces, block_size=128, func=np.median, return_trace=False):
    """
    Simple baseline correction function.

    Determines baseline in discrete chunks of "block_size" with a
    configurable function (default: median).

    Parameters
    ----------
    traces: np.array(n_events, n_channels, n_samples)
        Waveforms of several events/channels.
    block_size: int (default: 128)
        Number of samples/bins in one "chunk". If None, calculate median/mean over entire trace.
    func: callable (default: np.median)
        Function to calculate pedestal.
    return_trace: bool (default: False)
        If True, return the corrected waveforms in addition to the block offsets.

    Returns
    -------
    wfs_corrected: np.array(n_events, n_channels, n_samples)
        Baseline/pedestal corrected waveforms
    baseline_values: np.array of shape (n_samples // block_size, n_events, n_channels)
        (Only if return_offsets==True) The baseline offsets

    See Also
    --------
    fit_block_offsets :
        Function that uses an FFT to do an out-of-band removal of block offsets.
        This is generally much more accurate than this function.
    """

    num_samples = traces.shape[-1]
    n_cuncks = num_samples // block_size

    # traces -> (n_events (optional), n_channels (optional), num_samples)
    # block_offsets -> (n_cuncks, n_events, n_channels) pedestal for each chunck
    block_offsets = func(np.split(traces, n_cuncks, axis=-1), axis=-1)

    # (n_cuncks, n_events, n_channels) -> (n_events, n_channels, n_cuncks)
    block_offsets = np.moveaxis(block_offsets, 0, -1)

    if return_trace:
        output_traces = traces - np.repeat(block_offsets, block_size, axis=-1)
        return block_offsets, output_traces

    return block_offsets