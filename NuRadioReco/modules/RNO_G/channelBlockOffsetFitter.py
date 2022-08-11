from NuRadioReco.utilities import units
from NuRadioReco.framework.base_trace import BaseTrace
# from iminuit import Minuit
import numpy as np
import scipy

class channelBlockOffsets:

    def __init__(self, block_size=128, max_frequency=51*units.MHz):
        """
        Initialize block offset fitter.

        Parameters:
        -----------
        block_size: int (default: 128)
            The size (in samples) of the blocks
        max_frequency: float (default: 51 MHz)
            The maximum frequency to include in the out-of-band
            block offset fit

        """
        self.sampling_rate = None
        self.block_size = block_size # the size (in samples) of the blocks
        self._offset_guess = dict()
        self._offset_fit = dict()
        self._offset_inject = dict()
        self._max_frequency = max_frequency

    def add_offsets(self, event, station, offsets=1*units.mV, channel_ids=None):
        """
        Add (simulated or reconstructed) block offsets to an event.

        Parameters
        ----------
        event: `NuRadioReco.framework.event.Event` | None
        station: `NuRadioReco.framework.station.Station`
            The station to add block offsets to
        offsets: float | array | dict
            offsets to add to the event. Default: 1 mV

            - if a float, add gaussian-distributed of amplitude `offsets`
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
            elif len(np.atleast_1d(offsets)) == 1:
                add_offsets = np.random.normal(
                    0, offsets, (channel.get_number_of_samples() // self.block_size)
                )
            else:
                add_offsets = offsets
            if channel_id in self._offset_inject.keys():
                self._offset_inject[channel_id] += add_offsets
            else:
                self._offset_inject[channel_id] = add_offsets

            channel.set_trace(
                channel.get_trace() + np.repeat(add_offsets, self.block_size),
                channel.get_sampling_rate()
            )

    def remove_offsets(self, event, station, offsets='fit', channel_ids=None):
        """
        Remove block offsets from an event

        Fits and removes the block offsets from an event.

        Parameters
        ----------
        event: `NuRadioReco.framework.event.Event` | None
        station: `NuRadioReco.framework.station.Station`
            The station to remove the block offsets from
        offsets: str
            How to remove the offsets. Options are:

            - 'fit': fit the offsets out of band
            - 'guess': similar to 'fit', but just take
              a first guess at the offsets from the out-of-band region
              without actually performing the fit.
            - 'injected': if offsets were injected using the `add_offsets`
              method, this removes those offsets. Otherwise, this does nothing.

            Default: 'fit'
        channel_ids: list | None
            List of channel ids to remove offsets from. If None (default),
            remove offsets from all channels in `station`

        """
        if offsets=='fit':
            if not len(self._offset_fit):
                self.fit_offsets(event, station, channel_ids)
            offsets = self._offset_fit
        elif offsets=='guess':
            offsets = self._offset_guess
        elif offsets=='injected':
            if not len(self._offset_inject):
                offsets = np.zeros(16) #TODO - ensure this works for different trace lengths
            else:
                offsets = self._offset_inject

        if isinstance(offsets, dict):
            remove_offsets = {key: -offsets[key] for key in offsets.keys()}
        else:
            remove_offsets = -offsets
        self.add_offsets(event, station, remove_offsets, channel_ids)

    def fit_offsets(self, event, station, channel_ids=None):
        """
        Fit the block offsets using an out-of-band fit

        This function fits the block offsets present in a given
        event / station using an out-of-band fit in frequency space.

        Parameters
        ----------
        event: `NuRadioReco.framework.event.Event` | None
        station: `NuRadioReco.framework.station.Station`
            The station to fit the block offsets to
        channel_ids: list | None
            List of channel ids to fit block offsets for. If None (default),
            fit offsets for all channels in `station`

        """
        block_size = self.block_size
        if channel_ids is None:
            channel_ids = station.get_channel_ids()
        for channel_id in channel_ids:
            channel = station.get_channel(channel_id)
            trace = channel.get_trace()

            # we use the bandpass-filtered trace to get a first estimate of
            # the block offsets, by simply averaging over each block.
            filtered_trace = channel.get_filtered_trace([0, self._max_frequency], 'rectangular')
            self.sampling_rate = channel.get_sampling_rate()
            n_blocks = len(trace) // block_size

            # obtain guesses for block offsets
            a_guess = np.array([
                np.mean(filtered_trace[i*block_size:(i+1)*block_size])
                for i in range(n_blocks)
            ])
            self._offset_guess[channel_id] = a_guess
            # we can get rid of one parameter through a global shift
            a_guess = a_guess[:-1] - a_guess[-1]
            frequencies = channel.get_frequencies()
            spectrum = channel.get_frequency_spectrum()

            # we perform the fit out-of-band, in order to avoid
            # distorting any actual signal
            mask = (frequencies > 0) & (frequencies < self._max_frequency)

            frequencies_oob = frequencies[mask]
            spectrum_oob = spectrum[mask]

            ns = self.block_size
            dt = 1/self.sampling_rate

            # most of the terms in the fit depend only on the frequencies,
            # sampling rate and number of blocks. We therefore calculate these
            # only once, outside the fit function.
            pre_factor_exponent = np.array([
                -2.j * np.pi * frequencies_oob * dt * ((j+.5) * ns - .5)
                 for j in range(len(a_guess))
            ])
            self._const_fft_term = (
                 1 / self.sampling_rate * np.sqrt(2) # NuRadio FFT normalization
                * np.exp(pre_factor_exponent)
                * np.sin(np.pi*frequencies_oob*ns*dt)[None]
                / np.sin(np.pi*frequencies_oob*dt)[None]
            )
            self._spectrum = spectrum_oob
            res = scipy.optimize.fmin(
                self._pedestal_fit, a_guess, disp=0)
            ### Minuit seems a lot quicker than scipy - not sure why?
            # m = Minuit(self._pedestal_fit_minuit, a_guess * nufft_conversion_factor)
            # m.errordef = 1
            # m.errors = 0.01 * np.ones_like(a_guess)
            # m.migrad(ncall=20000)
            # res = m.values
            block_offsets = np.zeros(len(res) + 1)
            block_offsets[:-1] = res
            # the fit is not sensitive to an overall shift,
            # so we include the zero-meaning here
            block_offsets += np.mean(trace) - np.mean(block_offsets)

            self._offset_fit[channel_id] = block_offsets

    def get_offsets(self, channel_id, offset_type='fit'):
        """
        Return the block offsets for a given channel.

        Parameters
        ----------
        channel_id: int
            channel id that specifies the channel to return block offsets for
        offset_type: str
            Options:

            - 'fit': return the fitted block offsets
            - 'guess': return the first guess for the block offsets
            - 'injected': return the block offsets that were injected
              using the `add_offsets` method.

        Returns
        -------
        trace: `BaseTrace`
            A `BaseTrace` object with the same length as the channel trace,
            containing only the block offsets.

        """
        trace = BaseTrace()
        if offset_type == 'fit':
            trace.set_trace(np.repeat(self._offset_fit[channel_id], self.block_size), self.sampling_rate)
        elif offset_type == 'guess':
            trace.set_trace(np.repeat(self._offset_guess[channel_id], self.block_size), self.sampling_rate)
        elif offset_type == 'injected':
            trace.set_trace(np.repeat(self._offset_inject[channel_id], self.block_size), self.sampling_rate)
        return trace

    def _pedestal_fit(self, a):
        fit = np.sum(a[:, None] * self._const_fft_term, axis=0)
        chi2 = np.sum(np.abs(fit-self._spectrum)**2)
        return chi2