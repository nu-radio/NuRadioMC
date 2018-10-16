import numpy as np
import copy
from numpy import fft
from NuRadioReco.utilities import units
from scipy import signal
from numpy.polynomial import polynomial as poly
import matplotlib.pyplot as plt
from radiotools import helper as hp
from radiotools import coordinatesystems
from NuRadioReco.utilities import fft
import logging
logger = logging.getLogger('stationSignalReconstructor')
from NuRadioReco.framework.parameters import stationParameters as stnp

class stationSignalReconstructor:
    """
    Calculates basic signal parameters.
    """

    def __init__(self):
        self.__conversion_factor_integrated_signal = 2.65441729 * 1e-3 * 1.e-9 * 6.24150934 * 1e18  # to convert V**2/m**2 * s -> J/m**2 -> eV/m**2
        self.begin()

    def begin(self, signal_window_pre=10 * units.ns, signal_window_post=40 * units.ns, noise_window=100 * units.ns):
        self.__signal_window_pre = signal_window_pre
        self.__signal_window_post = signal_window_post
        self.__noise_window = noise_window

    def run(self, evt, station, det, debug=False):

        trace_copy = copy.copy(station.get_trace())

        # calculate hilbert envelope
        envelope = np.abs(signal.hilbert(trace_copy))
        envelope_mag = np.linalg.norm(envelope, axis=0)
        signal_time_bin = np.argmax(envelope_mag)
        signal_time = signal_time_bin / station.get_sampling_rate()
        station[stnp.signal_time] = signal_time

#
        low_pos = np.int(130 * units.ns * station.get_sampling_rate())
        up_pos = np.int(210 * units.ns * station.get_sampling_rate())
        if(debug):
            fig, ax = plt.subplots(1, 1)
            sc = ax.scatter(trace_copy[1, low_pos:up_pos], trace_copy[2, low_pos:up_pos], c=station.get_times()[low_pos:up_pos], s=5)
            fig.colorbar(sc, ax=ax)
            ax.set_aspect('equal')
            ax.set_xlabel("eTheta")
            ax.set_ylabel("ePhi")
            fig.tight_layout()
#

        low_pos, up_pos = hp.get_interval(envelope_mag, scale=0.5)
        v_start = trace_copy[:, signal_time_bin]
        v_avg = np.zeros(3)
        for i in range(low_pos, up_pos + 1):
            v = trace_copy[:, i]
            alpha = hp.get_angle(v_start, v)
            if(alpha > 0.5 * np.pi):
                v *= -1
            v_avg += v
        station[stnp.efield_vector] = v_avg
        pole = np.arctan2(np.abs(v_avg[2]), np.abs(v_avg[1]))
#         if(pole > 180 * units.deg):
#             pole -= 360 * units.deg
#         if(pole > 90 * units.deg):
#             pole = 180 * units.deg - pole
        station[stnp.efield_vector_polarization] = pole
        logger.info("average e-field vector = {:.4g}, {:.4g}, {:.4g} -> polarization = {:.1f}deg".format(v_avg[0], v_avg[1], v_avg[2], pole / units.deg))
        trace = station.get_trace()

        if(debug):
            fig, ax = plt.subplots(1, 1)
            tt = station.get_times()
            t0 = station.get_trace_start_time()
            dt = 1. / station.get_sampling_rate()
            ax.plot(tt / units.ns, trace[1] / units.mV * units.m)
            ax.plot(tt / units.ns, trace[2] / units.mV * units.m)
            ax.plot(tt / units.ns, envelope_mag / units.mV * units.m)
            ax.vlines([low_pos * dt + t0, up_pos * dt + t0], 0, envelope_mag.max() / units.mV * units.m)
            ax.vlines([signal_time - self.__signal_window_pre + t0, signal_time + self.__signal_window_post + t0], 0, envelope_mag.max() / units.mV * units.m, linestyles='dashed')

        t0 = station.get_trace_start_time()
        times = station.get_times()
        mask_signal_window = (times > (signal_time - self.__signal_window_pre + t0)) & (times < (signal_time + self.__signal_window_post + t0))
        mask_noise_window = np.zeros_like(mask_signal_window, dtype=np.bool)
        if(self.__noise_window > 0):
            mask_noise_window[np.int(np.round((-self.__noise_window - 141.) * station.get_sampling_rate())):np.int(np.round(-141. * station.get_sampling_rate()))] = np.ones(np.int(np.round(self.__noise_window * station.get_sampling_rate())), dtype=np.bool)  # the last n bins

        dt = times[1] - times[0]
        f_signal = np.sum(trace[:, mask_signal_window] ** 2, axis=1) * dt
        signal_energy_fluence = f_signal
        logger.debug('f signal {}'.format(f_signal))
        f_noise = np.zeros_like(f_signal)
        signal_energy_fluence_error = np.zeros(3)
        if(np.sum(mask_noise_window)):
            f_noise = np.sum(trace[:, mask_noise_window] ** 2, axis=1) * dt
            logger.debug('f_noise {},  {}/{} = {}'.format(f_noise * np.sum(mask_signal_window) / np.sum(mask_noise_window), np.sum(mask_signal_window), np.sum(mask_noise_window), 1. * np.sum(mask_signal_window) / np.sum(mask_noise_window)))
            signal_energy_fluence = f_signal - f_noise * np.sum(mask_signal_window) / np.sum(mask_noise_window)
            RMSNoise = np.sqrt(np.mean(trace[:, mask_noise_window] ** 2, axis=1))
            signal_energy_fluence_error = (4 * np.abs(signal_energy_fluence) * RMSNoise ** 2 * dt + 2 * (self.__signal_window_pre + self.__signal_window_post) * RMSNoise ** 4 * dt) ** 0.5

        signal_energy_fluence *= self.__conversion_factor_integrated_signal
        signal_energy_fluence_error *= self.__conversion_factor_integrated_signal
        station.set_parameter(stnp.signal_energy_fluence, signal_energy_fluence)
        station.set_parameter_error(stnp.signal_energy_fluence, signal_energy_fluence_error)

        logger.info("f = {} +- {}".format(signal_energy_fluence / units.eV * units.m2, signal_energy_fluence_error / units.eV * units.m2))

        # calculate polarization angle from energy fluence
        x = np.abs(signal_energy_fluence[1]) ** 0.5
        y = np.abs(signal_energy_fluence[2]) ** 0.5
        sx = signal_energy_fluence_error[1] * 0.5
        sy = signal_energy_fluence_error[2] * 0.5
        pol_angle = np.arctan2(y, x)
        pol_angle_error = 1. / (x ** 2 + y ** 2) * (y ** 2 * sx ** 2 + x ** 2 * sy ** 2) ** 0.5  # gaussian error propagation
        logger.info("polarization angle = {:.1f} +- {:.1f}".format(pol_angle / units.deg, pol_angle_error / units.deg))
        station.set_parameter(stnp.polarization_angle, pol_angle)
        station.set_parameter_error(stnp.polarization_angle, pol_angle_error)

        # compute expeted polarization
        site = det.get_site(station.get_id())
        exp_efield = hp.get_lorentzforce_vector(station[stnp.zenith], station[stnp.azimuth], hp.get_magnetic_field_vector(site))
        cs = coordinatesystems.cstrafo(station[stnp.zenith], station[stnp.azimuth], site=site)
        exp_efield_onsky = cs.transform_from_ground_to_onsky(exp_efield)
        exp_pol_angle = np.arctan2(np.abs(exp_efield_onsky[2]), np.abs(exp_efield_onsky[1]))
        logger.info("expected polarization angle = {:.1f}".format(exp_pol_angle / units.deg))
        station.set_parameter(stnp.polarization_angle_expectation, exp_pol_angle)

        return

        spec = np.abs(station.get_frequency_spectrum())
        spec_mag = np.linalg.norm(spec, axis=0)
        ff = station.get_frequencies()
        mask = (ff > 100 * units.MHz) & (ff < 500 * units.MHz)
        p1, statsp1 = poly.polyfit(ff[mask], np.log10(spec_mag[mask]), 1, full=True)
        p2, stats = poly.polyfit(ff[mask], np.log10(spec_mag[mask]), [0, 2], full=True)

        if debug:
            xxx = np.linspace(100, 500, 1000) * units.MHz
            fig, ax = plt.subplots(1, 1)
            ax.plot(station.get_frequencies(), spec[0])
            ax.plot(station.get_frequencies(), spec[1])
            ax.plot(station.get_frequencies(), spec[2])
            ax.plot(station.get_frequencies(), spec_mag)
            ax.plot(xxx, 10 ** poly.polyval(xxx, p1, "--"))
            ax.plot(xxx, 10 ** poly.polyval(xxx, p2, "--"))
            ax.semilogy(True)
            ax.set_ylim(spec_mag[mask].min() * 0.2, spec_mag[mask].max() * 1.2)
            ax.set_xlim(100 * units.MHz, 500 * units.MHz)

        def analytic_pulse(x, amp_p0, amp_p1, phase_p0, phase_p1, frequencies):
            amps = amp_p0 + frequencies * amp_p1
            # phases = phase_p0 + frequencies * phase_p1
            phases = phase_p0 + frequencies * phase_p1
            xx = amps * np.exp(phases * 1j)
            mask = (frequencies < 30) | (frequencies > 80)
            xx[mask] = 0
            return fft.freq2time(xx)

        # rotate into vB frame
        trace_vB = station.get_trace_vBvvB()[0]

        trace_vB_signal = trace_vB[mask_signal_window]
        envelope_vB = np.abs(signal.hilbert(trace_vB_signal))
        maxpos = np.argmax(envelope_vB)
        trace_vB_signal = np.roll(trace_vB_signal, -maxpos)
        spec2 = fft.time2freq(trace_vB_signal)
        spec_signal = np.abs(spec2)
        spec_signal_phase = np.unwrap(np.angle(spec2))
        ff_signal = np.fft.rfftfreq(np.sum(mask_signal_window), dt)
        spec_noise = np.abs(fft.time2freq(trace_vB[mask_noise_window]))
        ff_noise = np.fft.rfftfreq(np.sum(mask_noise_window), dt)

        if debug:
            fig, (ax, ax2) = plt.subplots(1, 2)
            ax.plot(ff_signal / units.MHz, spec_signal, "o-")
            ax.plot(ff_noise / units.MHz, spec_noise, "o-")
            ax.set_xlim(0, 700)
            ax2.plot(ff_signal / units.MHz, np.rad2deg(spec_signal_phase), "o-")
            ax2.set_xlim(0, 700)
            ax2.set_ylabel("phase")
            plt.tight_layout()

            fig, ax = plt.subplots(1, 1)
    #         ax.plot(times, envelope, "C0--")
            ax.plot(times, trace_vB, "-C0")
    #         ax.plot(times, envelope, "C1--")
    #         ax.plot(times, trace_vvB[1], "-C1")
    #         ax.plot(times, trace_vvB[2], "-C2")
            ax.plot(times, envelope_mag, "--C2")
            ax.axvspan(signal_time - self.__signal_window * 0.5, signal_time + self.__signal_window * 0.5, alpha=0.5, color='green')
            ax.axvspan(times[-1] - self.__noise_window, times[-1], alpha=0.5, color='red')
            plt.show()

    def end(self):
        pass
