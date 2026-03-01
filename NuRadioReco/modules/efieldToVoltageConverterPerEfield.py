import numpy as np
import time
import logging
from scipy.fft import next_fast_len

from NuRadioReco.modules.base.module import register_run
from NuRadioReco.detector import antennapattern
from NuRadioReco.utilities import units, signal_processing
from NuRadioReco.modules.efieldToVoltageConverter import calculate_time_shift_for_cosmic_ray

import NuRadioReco.framework.sim_channel
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import channelParameters as chp


class efieldToVoltageConverterPerEfield():
    """
    This module applies the antenna response to each electric field individually and stores the
    resulting voltage traces in the SimStationclass as SimChannel objects
    """

    def __init__(self, log_level=logging.NOTSET):
        self.__t = 0
        self.logger = logging.getLogger('NuRadioReco.efieldToVoltageConverterPerEfield')
        self.logger.setLevel(log_level)
        self.antenna_provider = antennapattern.AntennaPatternProvider()
        self._pre_pulse_time = 0.0
        self._post_pulse_time = 0.0

    def begin(self, pre_pulse_time=0.0, post_pulse_time=0.0):
        """
        Configure zero-padding for linear convolution.

        When padding > 0, the E-field trace is zero-padded before FFT so that
        the frequency-domain multiplication with the antenna response is
        equivalent to linear (not circular) convolution. The result is cropped
        back to the original trace length.

        Parameters
        ----------
        pre_pulse_time : float
            Time to zero-pad before the trace (in NuRadioReco units, i.e. ns)
        post_pulse_time : float
            Time to zero-pad after the trace (in NuRadioReco units, i.e. ns)
        """
        self._pre_pulse_time = float(pre_pulse_time)
        self._post_pulse_time = float(post_pulse_time)

    @register_run()
    def run(self, evt, station, det):
        """
        Converts simulated electric fields to voltage traces for each channel in a sim station.

        Parameters
        ----------
        evt : Event
            The event object.
        station : SimStation (or Station)
            The SimStation object. If a Station object is provided, it is detected and the
            SimStation is automatically retrived from the Station object
        det : Detector
            The detector object.

        Returns
        -------
        None

        Raises
        ------
        LookupError
            If the station has no electric fields.
        """
        t = time.time()

        # access simulated efield and high level parameters
        if isinstance(station, NuRadioReco.framework.station.Station):
            sim_station = station.get_sim_station()
        else:
            sim_station = station
        if(len(sim_station.get_electric_fields()) == 0):
            raise LookupError(f"station {station.get_id()} has no efields")

        for channel_id in det.get_channel_ids(sim_station.get_id()):
            # one channel might contain multiple channels to store the signals from multiple ray paths and showers,
            # so we loop over all simulated channels with the same id,
            self.logger.debug('channel id {}'.format(channel_id))
            for electric_field in sim_station.get_electric_fields_for_channels([channel_id]):
                sim_channel = NuRadioReco.framework.sim_channel.SimChannel(channel_id, shower_id=electric_field.get_shower_id(),
                                                                           ray_tracing_id=electric_field.get_ray_tracing_solution_id())
                sim_channel[chp.signal_ray_type] = electric_field[efp.ray_path_type]

                sr = electric_field.get_sampling_rate()
                dt = 1.0 / sr

                efield_trace = electric_field.get_trace()  # shape (3, N)
                N = efield_trace.shape[1]

                n_pre = int(np.ceil(self._pre_pulse_time * sr))
                n_post = int(np.ceil(self._post_pulse_time * sr))

                if n_pre == 0 and n_post == 0:
                    Npad = N
                else:
                    Npad = next_fast_len(N + n_pre + n_post)

                trace_pad = np.zeros((3, Npad), dtype=efield_trace.dtype)
                trace_pad[:, n_pre:n_pre + N] = efield_trace

                ff = np.fft.rfftfreq(Npad, d=dt)
                efield_fft = np.fft.rfft(trace_pad, axis=-1)

                zenith = electric_field[efp.zenith]
                azimuth = electric_field[efp.azimuth]

                VEL = signal_processing.get_efield_antenna_factor(sim_station, ff, [channel_id], det, zenith, azimuth, self.antenna_provider)

                if VEL is None:
                    voltage_fft = np.zeros_like(efield_fft[1])
                else:
                    VEL = VEL[0]
                    voltage_fft = np.sum(VEL * np.array([efield_fft[1], efield_fft[2]]), axis=0)

                voltage_fft[ff < 5 * units.MHz] = 0.0

                v_full = np.fft.irfft(voltage_fft, n=Npad)

                if n_pre > 0 or n_post > 0:
                    edge = max(1, int(np.ceil(200 * sr)))
                    edge_max = np.max(np.abs(np.r_[v_full[:edge], v_full[-edge:]]))
                    peak = np.max(np.abs(v_full))
                    if peak > 0 and edge_max > 2e-3 * peak:
                        self.logger.warning(
                            "PerEfield padding may be insufficient: edge_max/peak=%.3g "
                            "(ch %d, pre=%.0f ns, post=%.0f ns, Npad=%d)",
                            edge_max / peak, channel_id,
                            self._pre_pulse_time, self._post_pulse_time, Npad)

                v = v_full[n_pre:n_pre + N]

                dist_channel_efield = np.linalg.norm(det.get_relative_position(sim_station.get_id(), channel_id) - electric_field.get_position())
                if dist_channel_efield / units.mm > 0.01:
                    travel_time_shift = calculate_time_shift_for_cosmic_ray(
                        det, sim_station, electric_field, channel_id)
                else:
                    travel_time_shift = 0

                sim_channel.set_trace(v, sr)
                sim_channel.set_trace_start_time(electric_field.get_trace_start_time() + travel_time_shift)
                sim_station.add_channel(sim_channel)

        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        self.logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        self.logger.info("total time used by this module is {}".format(dt))
        return dt
