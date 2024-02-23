from NuRadioReco.utilities import units
import NuRadioReco.framework.base_trace

from scipy import interpolate

import numpy as np
import logging
import datetime


class Response:
    """
    This class provides an interface to read-in and apply the complex response functions of the
    various components of the signal chain of a RNO-G channel. The class allows to combine the
    response of several components into one response by multiplying them.

    Examples
    --------

    .. code-block::

        response = det.get_signal_chain_response(station_id=24, channel_id=0)

        # Multipy the response to a trace. The multiply operator takes care of everything
        trace_at_readout = trace_at_antenna * response

        # getting the complex response as array
        freq = np.arange(50, 1000) * units.MHz
        complex_resp = response(freq)

    """

    def __init__(self, frequency, y, y_unit, time_delay=0, weight=1,
                 name="default", station_id=None, channel_id=None,
                 remove_time_delay=True, debug_plot=False,
                 log_level=logging.INFO):
        """
        Parameters
        ----------

        frequency : list(float)
            The frequency vector at which the response is measured. Unit has to be GHz.

        y : [list(float), list(float)]
            The measured response. First entry is the vector of the measured amplitude, the second entry is the measured phase.
            The unit of both entries is specified with the next argument.

        y_unit : [str, str]
            The first entry specifies the unit of the measured amplitude. Options are "dB", "MAG" and "mag".
            The second entry specifies the unit of the measured phase. Options are "rad" and "deg".

        time_delay : float (Default: 0)
            "Average" group delay. Read from database. Will be used to normalize the phase and stored alongside response

        weight : float (Default: 1)
            Specifies the weight with which this component reponse "adds" to the total signal-chain response or data.
            Its the exponent of the complex multiplicitive gain. That means that a value of 1 means to linear multiply this
            reponse while a value of -1 means to divide by this reponse.

        name : str (Default: "default")
            Give the response a name. This is only use for printing purposes.

        station_id : int (Default: None)
            Associated station id.

        channel_id : int (Default: None)
            Associated channel id.

        remove_time_delay : bool (Default: True)
            If True, remove `time_delay` from response.

        debug_plot : bool (Default: False)
            If True, produce a debug plot

        log_level : `logging.LOG_LEVEL` (Default: logging.INFO)
            Defines verbosity level of logger. Other options are: `logging.WARNING`, `logging.DEBUG`, ...
        """
        self.logger = logging.getLogger("NuRadioReco.Response")
        self.logger.setLevel(log_level)

        self.__names = [name]
        self._station_id = station_id
        self._channel_id = channel_id

        self._sanity_check = True # Tmp

        if self._station_id is None or self._channel_id is None:
            self.logger.error(f"Station and channel id were not defined for response {name}. Please do that.")

        self.__frequency = np.array(frequency) * units.GHz

        if y[0] is None or y[1] is None:
            raise ValueError("Data for response incomplete, detected \"None\"")

        y_ampl, y_phase = np.array(y)
        if y_unit[0] == "dB":
            gain = 10 ** (y_ampl / 20)
        elif y_unit[0].lower() == "mag":
            gain = y_ampl
        else:
            raise KeyError

        if y_unit[1].lower() == "deg":
            if np.max(np.abs(y_phase)) < 2 * np.pi:
                self.logger.warning("Is the phase really in deg? Does not look like it... "
                                    f"Do not convert to rad. Phase: {y_phase}")
            else:
                y_phase = np.deg2rad(y_phase)
        elif y_unit[1].lower() == "rad":
            y_phase = y_phase
        else:
            raise KeyError

        y_phase_orig = np.copy(y_phase)

        if remove_time_delay:
            if np.any(np.abs(2 * time_delay * np.diff(self.__frequency)) > 1):
                df_max = 1 / np.abs(2 * time_delay) * 0.8  # the factor of 0.8 is to be lower
                self.logger.warning(
                    f"The frequency binning (resolution) of {np.diff(self.__frequency)[0] * 1e3:.2f} MHz "
                    f"of the response function is to large/corse to correctly remove the time delay of {time_delay} ns. "
                    f"The response function is now upsampled to {df_max * 1e3:.2f} MHz.")
                new_frequencies = np.arange(self.__frequency[0], self.__frequency[-1], df_max)
                gain = np.interp(new_frequencies, self.__frequency, gain)
                y_phase = np.interp(new_frequencies, self.__frequency, y_phase)
                self.__frequency = new_frequencies

        # Remove the average group delay from response
        if remove_time_delay and time_delay:
            _response = subtract_time_delay_from_response(self.__frequency, gain, y_phase, time_delay)
            y_phase = np.angle(_response)
        else:
            time_delay = 0  # set time_delay to 0 if group delay is not removed

        y_phase = np.unwrap(y_phase)

        self.__gains = [interpolate.interp1d(
            self.__frequency, gain, kind="linear", bounds_error=False, fill_value=0)]

        self.__phases = [interpolate.interp1d(
            self.__frequency, y_phase, kind="linear", bounds_error=False, fill_value=0)]

        self.__weights = [weight]
        self.__time_delays = [weight * time_delay]

        if debug_plot:
            from matplotlib import pyplot as plt
            fig, axs = plt.subplots(3, 1, sharex=True)
            axs[0].set_title(name)
            frequency_interp = np.linspace(self.__frequency[0], self.__frequency[-1], 10000)
            axs[0].plot(self.__frequency, gain, "C0o", label="data", markersize=2)
            axs[0].plot(frequency_interp, self.__gains[0](frequency_interp), "C1--", lw=1, label="interpolation")

            if remove_time_delay:
                axs[1].plot(self.__frequency, y_phase_orig, "C0o", markersize=2, label="Original phase")
                ax2 = axs[1].twinx()
                ax2.plot(self.__frequency, y_phase, "C2o", markersize=2, label=f"Excl. time delay of {self.__time_delays[0]:.2f}ns")
                ax2.plot(frequency_interp, self.__phases[0](frequency_interp), "C3--", lw=1, label="interpolation")
                ax2.set_ylabel("norm. phase / rad")
                ax2.legend(fontsize=5)
            else:
                axs[1].plot(self.__frequency, y_phase, "C0o", markersize=2)
                axs[1].plot(frequency_interp, self.__phases[0](frequency_interp), "C1--", lw=1)

            axs[0].set_ylabel("gain")
            axs[1].set_ylabel("phase / rad")

            axs[0].legend(fontsize=5)

            group_delay = -np.diff(y_phase) / np.diff(self.__frequency)[0] / (2 * np.pi)
            axs[2].plot(self.__frequency[:-1], group_delay, "C0o", markersize=2)
            frequency_mask = np.all([50 * units.MHz < self.__frequency[:-1], self.__frequency[:-1] < 800 * units.MHz], axis=0)
            axs[2].set_ylim(np.amin(group_delay[frequency_mask]), np.amax(group_delay[frequency_mask]))
            axs[2].set_ylabel("group delay / ns")

            axs[-1].set_xlabel("frequency / GHz")
            fig.tight_layout()
            plt.savefig(f'{name}_{self._station_id}_{self._channel_id}_'
                        f'{datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")}_debug.png', transparent=False)
            plt.close()

    def __call__(self, freq, component_name=None):
        """
        Returns the complex response for a given frequency.

        Parameters
        ----------

        freq: list(float)
            The frequencies for which to get the response.

        Returns

        response: np.array(np.complex128)
            The complex response at the desired frequencies
        """
        response = np.ones_like(freq, dtype=np.complex128)

        for gain, phase, weight, name in zip(self.__gains, self.__phases, self.__weights, self.__names):

            if component_name is not None and component_name != name:
                continue

            _gain = gain(freq / units.GHz)

            # to avoid RunTime warning and NANs in total reponse
            if weight == -1:
                _gain = np.where(_gain > 0, _gain, 1e-6)

            response *= (_gain * np.exp(1j * phase(freq / units.GHz))) ** weight

        if np.allclose(response, np.ones_like(freq, dtype=np.complex128)):
            if component_name is not None:
                raise ValueError("Returned response is equal to 1. "
                                f"Did you requested a non-existing component ({component_name})? "
                                f"Options are: {self.__names}")
            else:
                self.logger.warning("Returned response is equal to 1.")

        return response

    def get_names(self):
        """ Get list of the names of all individual responses """
        return self.__names

    def __mul__(self, other):
        """
        Define multiplication operator for
            - Other objects of the same class
            - Objects of type NuRadioReco.framework.base_trace
        """

        if isinstance(other, Response):
            if self._station_id != other._station_id or \
                self._channel_id != other._channel_id:
                self.logger.error("It looks like you are combining responses from "
                                  f"two different channels: {self._station_id}.{self._channel_id} "
                                  f" vs {other._station_id}.{other._channel_id} (station_id.channel_id)")
            # Store each response individually: append/concatenate lists of gains and phases.
            # The multiplication happens in __call__.
            self.__names += other.__names
            self.__gains += other.__gains
            self.__phases += other.__phases
            self.__weights += other.__weights
            self.__time_delays += other.__time_delays
            return self

        elif isinstance(other, NuRadioReco.framework.base_trace.BaseTrace):

            if self._sanity_check:
                trace_length = other.get_number_of_samples() / other.get_sampling_rate()
                time_delay = self._calculate_time_delay()
                if time_delay > trace_length / 2:
                    self.logger.warning("The time shift appiled by the response is larger than half the trace length:\n\t"
                                        f"{time_delay:.2f} vs {trace_length:.2f}")

            spec = other.get_frequency_spectrum()
            freqs = other.get_frequencies()
            spec *= self(freqs)  # __call__
            other.add_trace_start_time(np.sum(self.__time_delays))
            other.set_frequency_spectrum(spec, sampling_rate="same")
            return other

        elif isinstance(other, np.ndarray):
            raise TypeError("You try to multiply a `Response` object with a numpy array, "
                            "only `Response` or `BaseTrace` is allowed. "
                            "Did you call `get_trace()` or `get_frequency_spectrum()` on `BaseTrace`?")

        else:
            raise TypeError(f"Response multiplied with unknown type: {type(other)}")

    def __rmul__(self, other):
        """ Same as mul """
        return self.__mul__(other)

    def __str__(self):
        ampl = 20 * np.log10(np.abs(self(0.5 * units.GHz)))
        return "Response of " + ", ".join([f"{name} ({weight})" for name, weight in zip(self.get_names(), self.__weights)]) \
            + f": |R(0.5 GHz)| = {ampl:.2f} dB (amplitude) ({np.sum(self.__time_delays):.2f} ns)"

    def plot(self, ax1=None, show=False, in_dB=True, plt_kwargs={}):
        import matplotlib.pyplot as plt

        freqs = np.linspace(0, 1.4) * units.GHz

        if ax1 is None:
            fig, ax = plt.subplots()
        else:
            ax = ax1

        for gain, weight, name in zip(self.__gains, self.__weights, self.__names):
            _gain = gain(freqs)

            name = name.replace("_", " ")
            ls = "-" if weight == 1 else "--"

            if in_dB:
                mask = _gain > 0  # to avoid RunTime warning
                ax.plot(freqs[mask] / units.MHz, 20 * np.log10(_gain[mask]), ls=ls, label=name + f": {weight}", **plt_kwargs)
            else:
                ax.plot(freqs / units.MHz, _gain, label=name + f": {weight}", ls=ls, **plt_kwargs)

        _gain = np.abs(self(freqs))
        if in_dB:
            mask = _gain > 0  # to avoid RunTime warning
            ax.plot(freqs[mask] / units.MHz, 20 * np.log10(_gain[mask]), color="k", label="total", **plt_kwargs)
        else:
            ax.plot(freqs / units.MHz, _gain, color="k", label="total", **plt_kwargs)

        ax.set_xlabel("frequency / MHz")
        if in_dB:
            ax.set_ylabel("gain / dB")
        else:
            ax.set_ylabel("gain")
            ax.set_yscale("log")

        ax.legend()
        ax.grid()

        if show:
            plt.show()
        elif ax1 is not None:
            pass
        else:
            return fig, ax

    def get_time_delay(self):
        """ Get time delay from DB """
        return np.sum(self.__time_delays)

    def _calculate_time_delay(self):
        """
        Calculate time delay from phase of the stored complex response function.
        This is not the time delay which is stored in the DB and which is used in
        the `__init__()` to normalize the response function. Rather, its the remaining
        group delay.

        The time delay is calculated as the mean between 195 and 205 MHz.

        Returns
        -------

        time_delay1 : float
            The time delay at ~ 200 MHz
        """

        freqs = np.arange(0.05, 1.2, 0.001) * units.GHz

        response = self(freqs)

        delta_freq = np.diff(freqs)

        phase = np.angle(response)
        time_delay = -np.diff(np.unwrap(phase)) / delta_freq / 2 / np.pi

        mask = np.all([195 * units.MHz < freqs, freqs < 250 * units.MHz], axis=0)[:-1]
        time_delay1 = np.mean(time_delay[mask])

        # fit the unwrapped phase with a linear function
        popt = np.polyfit(freqs, np.unwrap(phase), 1)
        time_delay2 = -popt[0] / (2 * np.pi)

        if np.abs(time_delay1 - time_delay2) > 0.1 * units.ns:
            self.logger.warning("Calculation of time delay. The two methods yield different results: "
                                f"{time_delay1:.1f} ns / {time_delay2:.1f} ns for {self.get_names()}. Return the former...")

        return time_delay1


def subtract_time_delay_from_response(frequencies, resp, phase=None, time_delay=None):
    """
    Remove a constant time delay from a complex response function

    Parameters
    ----------

    frequencies : array of floats
        Corresponding frequencies

    resp : array of (complex) floats
        Complex response function (if `phase is None`), Real gain of a complex response
        function (if `phase is not None`).

    phase : array of floats
        Phase of the complex response function (optional). (Default: None)

    time_delay : float
        Time delay to be removed

    Returns
    -------

    resp: array of complex floats
        Corrected response function
    """
    resp = np.copy(resp)

    if phase is not None:
        phase = np.copy(phase)
        gain = resp
        phase = np.unwrap(phase)  # double helps more
    else:
        gain = np.abs(resp)
        phase = np.unwrap(np.angle(resp))

    if time_delay is None:
        raise ValueError("You have to specify a time delay")

    if np.any(np.abs(2 * time_delay * np.diff(frequencies)) > 1):
        raise ValueError("The frequency binning (resolution) of the response function "
                         f"is to large/corse to correctly remove the time delay of {time_delay} ns. "
                         "You need to upsample the response function.")

    resp = gain * np.exp(1j * (phase + 2 * np.pi * time_delay * frequencies))

    return resp