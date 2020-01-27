import numpy as np
from numpy.polynomial import polynomial as pol
import copy
import matplotlib.pyplot as plt
import logging
import datetime
import os

from radiotools import plthelpers as php

from NuRadioReco.utilities import units
import NuRadioReco.framework.channel
import NuRadioReco.framework.station
import NuRadioReco.framework.event
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
import NuRadioReco.modules.channelGenericNoiseAdder

import NuRadioReco.modules.ARA.triggerSimulator
import NuRadioReco.modules.trigger.highLowThreshold
from NuRadioMC.SignalGen import parametrizations as signalgen

from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import electricFieldParameters as efp

from NuRadioReco.detector import detector

from NuRadioReco.modules.base import module
logger = module.setup_logger(level=logging.DEBUG)

#fig = plt.gcf()
#fig.canvas.manager.window.tkraise()


def get_ARA_power_mean_rms(sampling_rate, Vrms, min_freq, max_freq):
    """
    helper function to calculate the mean and rms power of the ARA tunnel diode
    for a given Vrms, sampling rate and frequency content

    Parameters
    ----------
    sampling_rate: float
        the sampling rate e.g. 1GHz
    Vrms: float
        the RMS of noise in the time domain
    min_freq: float
        the lower bandpass frequency
    max_freq: float
        the upper bandpass frequency
    """
    triggerSimulator = NuRadioReco.modules.ARA.triggerSimulator.triggerSimulator()
    channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()

    noise = NuRadioReco.framework.channel.Channel(0)

    long_noise = channelGenericNoiseAdder.bandlimited_noise(min_freq=min_freq,
                                                            max_freq=max_freq,
                                                            n_samples=2 ** 20,
                                                            sampling_rate=sampling_rate,
                                                            amplitude=Vrms,
                                                            type='perfect_white')
    long_noise *= Vrms / long_noise.std()

    print(long_noise.std())

    noise.set_trace(long_noise, sampling_rate)

    power_noise = triggerSimulator.tunnel_diode(noise)

    power_mean = np.mean(power_noise)
    power_rms = np.sqrt(np.mean(power_noise ** 2))
    return power_mean, power_rms


det = detector.Detector(json_filename='../example_data/dummy_detector.json')
det.update(datetime.datetime(2018, 10, 1))

efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
hardwareResponseIncorporator = NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()

logger.setLevel(logging.WARNING)


n_index = 1.78
cherenkov_angle = np.arccos(1. / n_index)
dt = 1. * units.ns
N = 2 ** 8

Vrms = 11 * units.micro * units.V

NN = 1000
min_freq, max_freq = 100 * units.MHz, .5 * units.GHz
if 0:
    # now calculate the relation between the ARA SNR (integrated power ratio) and the ARIANNA SNR (Vp2p/2/Vrms)
    import NuRadioReco.modules.ARA.triggerSimulator
    triggerSimulator = NuRadioReco.modules.ARA.triggerSimulator.triggerSimulator()
    power_mean, power_rms = get_ARA_power_mean_rms(1. / dt, Vrms, min_freq, max_freq)
    print("power mean/rms = {:.4g}/{:.4g}".format(power_mean, power_rms))
    counter = -1
    SNRp2p_bicone = np.zeros(NN)
    SNRara_bicone = np.zeros(NN)
    for E in 10 ** np.linspace(15.5, 17, 100):
        for theta in np.linspace(cherenkov_angle - 0.1 * units.deg, cherenkov_angle + 0.1 * units.deg, 10):
            counter += 1
            pulse = signalgen.get_time_trace(E, theta, N, dt, 'HAD', n_index, 1000 * units.m, 'Alvarez2000')
            event = NuRadioReco.framework.event.Event(1, 1)
            station = NuRadioReco.framework.station.Station(101)
            trace = np.zeros((3, N))
            trace[1] = pulse
            trace[2] = pulse
            sim_station = NuRadioReco.framework.sim_station.SimStation(101)
            electric_field = NuRadioReco.framework.electric_field.ElectricField([2])
            electric_field.set_trace(sampling_rate=1. / dt, trace=trace)
            electric_field[efp.azimuth] = 0
            electric_field[efp.zenith] = (90 + 45) * units.deg
            electric_field[efp.ray_path_type] = 'direct'
            sim_station.add_electric_field(electric_field)
            sim_station[stnp.zenith] = (90 + 45) * units.deg
            sim_station[stnp.zenith] = 0
            station.set_sim_station(sim_station)
            event.set_station(station)

            efieldToVoltageConverter.run(event, station, det)
            channelBandPassFilter.run(event, station, det, passband=[min_freq, max_freq])

            trace_bicone = station.get_channel(2).get_trace()

            SNRp2p_bicone[counter] = (trace_bicone.max() - trace_bicone.min()) / 2. / Vrms

            after_tunnel_diode = np.abs(triggerSimulator.tunnel_diode(station.get_channel(2)))
            power_mean = 0
            SNRara_bicone[counter] = np.max((after_tunnel_diode - power_mean) / power_rms)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(SNRp2p_bicone, SNRara_bicone, s=20, alpha=0.5)
    ax.set_xlabel("ARIANNA SNR (Vp2p/2/Vrms)")
    ax.set_ylabel("ARA SNR (tunnel diode output/noise power RMS)")
    ax.set_title("for ARA bicone response")
    fig.tight_layout()
    fig.savefig('plots/SNR_ARA_ARIANNA.png')
    plt.show()

long_noise = channelGenericNoiseAdder.bandlimited_noise(min_freq=min_freq,
                                            max_freq=max_freq,
                                            n_samples=2 ** 20,
                                            sampling_rate=1 / dt,
                                            amplitude=Vrms,
                                            type='perfect_white')
NN = 10000
SS_LPDA = np.zeros(NN)
SS_LPDA_amp = np.zeros(NN)
Vp2p_LPDA = np.zeros(NN)
Vp2p_LPDA_noise_1 = np.zeros(NN)
Vp2p_LPDA_noise_2 = np.zeros(NN)
Vp2p_LPDA_amp = np.zeros(NN)
SS_bicone = np.zeros(NN)
Vp2p_bicone = np.zeros(NN)
Vp2p_bicone_noise_1 = np.zeros(NN)
Vp2p_bicone_noise_2 = np.zeros(NN)
counter = -1
if not os.path.isdir('plots'):
    os.mkdir('plots')
for signal_scaling in np.linspace(1, 100, NN):
    for theta in np.linspace(cherenkov_angle - 3 * units.deg, cherenkov_angle * units.deg, 1):
        counter += 1
        print(counter)
        pulse = signalgen.get_time_trace(1e18, theta, N, dt, 'HAD', n_index, 1000 * units.m, 'Alvarez2000')
        event = NuRadioReco.framework.event.Event(1, 1)
        station = NuRadioReco.framework.station.Station(101)
        trace = np.zeros((3, N))
        trace[1] = pulse
        trace[2] = pulse
        trace = trace / np.max(np.abs(trace)) * signal_scaling * Vrms
        sim_station = NuRadioReco.framework.sim_station.SimStation(101)
        electric_field = NuRadioReco.framework.electric_field.ElectricField([2,3,4,5])
        electric_field.set_trace(sampling_rate=1. / dt, trace=trace)
        electric_field[efp.azimuth] = 0
        electric_field[efp.zenith] = (90 + 45) * units.deg
        electric_field[efp.ray_path_type] = 'direct'
        electric_field.set_trace_start_time(0)
        electric_field[efp.zenith] = (90 + 45) * units.deg
        electric_field[efp.azimuth] = 0
        sim_station.add_electric_field(electric_field)
        station.set_sim_station(sim_station)
        event.set_station(station)

        efieldToVoltageConverter.run(event, station, det)
        channelBandPassFilter.run(event, station, det, passband=[80 * units.MHz, 500 * units.MHz])

        trace_LPDA = copy.copy(station.get_channel(3).get_trace())
        trace_bicone = station.get_channel(2).get_trace()

        def get_Vp2p(trace, dt=dt, coincidence_window=5 * units.ns):
            tstart = 20 * units.ns
            tstop = 50 * units.ns
            maximum = 0
            for t in np.arange(tstart, tstop, dt):
                n_bins = np.int(np.ceil(coincidence_window / dt))
                bin_start = np.int(np.ceil(t / dt))
                maximum = max(maximum, np.max(trace[bin_start:(bin_start+n_bins)]) - np.min(trace[bin_start:(bin_start+n_bins)]))
            return maximum
#             return np.max(trace) - np.min(trace)
#             index = np.argmax(np.abs(trace))
#             n_bins = np.int(np.ceil(coincidence_window / dt))
#             maximum = trace[index]
#             sign = np.sign(maximum)
#             minimum = np.max(-1 * sign * trace[max(0, (index - n_bins)):min((index + n_bins), len(trace))])
#             return np.abs(maximum) + np.abs(minimum)

        Vp2p_LPDA[counter] = get_Vp2p(trace_LPDA)
        Vp2p_bicone[counter] = get_Vp2p(trace_bicone)

        SS_LPDA[counter] = np.sum(trace_LPDA ** 2) * dt
        SS_bicone[counter] = np.sum(trace_bicone ** 2) * dt

        if(counter < 0):  # plot some example traces
            fig, ax = plt.subplots(1, 3, sharey=True, figsize=(20, 7))
            ax = np.array(ax).flatten()
            tt = station.get_channel(3).get_times()
            ax[0].plot(tt / units.ns, pulse / units.micro / units.V)
            ax[1].plot(tt / units.ns, trace_bicone / units.micro / units.V)
            ax[2].plot(tt / units.ns, trace_LPDA / units.micro / units.V)
            ax[0].set_ylabel('voltage [$\mu$V]')
            ax[2].set_ylabel('voltage [mV]')
            ax[0].set_title("no antenna")
            ax[1].set_title("bicone")
            ax[2].set_title("LPDA")
            ax[0].set_xlabel("time [ns]")
            ax[1].set_xlabel("time [ns]")
            ax[2].set_xlabel("time [ns]")
            fig.tight_layout()
            fig.suptitle(r"$\theta$ = {:.0f}, dCherenkov = {:.0f}".format(135, (theta - cherenkov_angle) / units.deg))
            fig.subplots_adjust(top=0.9)
            fig.savefig("plots/trace_LPDA_bicone_{:d}.png".format(counter))
            plt.show()
            plt.close("all")

        channelGenericNoiseAdder.run(event, station, det, amplitude=Vrms,
                                     min_freq=100 * units.MHz,
                                     max_freq=500 * units.MHz,
                                     type='perfect_white')
        trace_LPDA = station.get_channel(3).get_trace()
        trace_bicone = station.get_channel(2).get_trace()
        Vp2p_LPDA_noise_1[counter] = get_Vp2p(trace_LPDA)
        Vp2p_bicone_noise_1[counter] = get_Vp2p(trace_bicone)

        trace_LPDA = station.get_channel(5).get_trace()
        trace_bicone = station.get_channel(4).get_trace()
        Vp2p_LPDA_noise_2[counter] = get_Vp2p(trace_LPDA)
        Vp2p_bicone_noise_2[counter] = get_Vp2p(trace_bicone)

        hardwareResponseIncorporator.run(event, station, det, sim_to_data=True)
        trace_LPDA_amp = station.get_channel(3).get_trace()
        Vp2p_LPDA_amp[counter] = get_Vp2p(trace_LPDA_amp)
        SS_LPDA_amp[counter] = np.sum(trace_LPDA_amp ** 2) * dt


fig, (ax, ax2) = plt.subplots(1, 2)
ax.scatter(Vp2p_LPDA / Vrms / 2, SS_LPDA / Vrms ** 2, label='LPDA')
ax.scatter(Vp2p_bicone / Vrms / 2, SS_bicone / Vrms ** 2, label='bicone')
ax.set_ylabel(r"$\sum A_i^2 \Delta t / V_\mathrm{RMS}^2 [ns]$")
ax.set_xlabel(r"$V_\mathrm{p2p}/2/V_\mathrm{RMS}$")
ax.semilogx(True)
ax.semilogy(True)
ax2.set_ylabel(r"$\sqrt{\sum A_i^2 \Delta t/ V_\mathrm{RMS}^2 [ns]}$")
ax2.set_xlabel(r"$V_\mathrm{p2p}/2/V_\mathrm{RMS}$")
ax2.scatter(Vp2p_LPDA / Vrms / 2, (SS_LPDA / units.ns / Vrms ** 2) ** 0.5, label='LPDA', s=15)
ax2.scatter(Vp2p_bicone / Vrms / 2, (SS_bicone / units.ns / Vrms ** 2) ** 0.5, label='bicone', s=15)
coef_bicone = pol.polyfit(Vp2p_bicone / Vrms / 2, (SS_bicone / units.ns / Vrms ** 2) ** 0.5, deg=[1])
coef_LPDA = pol.polyfit(Vp2p_LPDA / Vrms / 2, (SS_LPDA / units.ns / Vrms ** 2) ** 0.5, deg=[1])
print("slope LPDA: SNR_new = ({:.5f} * SNR_p2p + {:.5f} ".format(coef_LPDA[1], coef_LPDA[0]))
print("slope bicone: SNR_new = ({:.5f} * SNR_p2p + {:.5f} ".format(coef_bicone[1], coef_bicone[0]))
xx = np.linspace(0, 150)
ax2.plot(xx, pol.polyval(xx, coef_bicone), '--C3')
ax2.plot(xx, pol.polyval(xx, coef_LPDA), '--C2')
ax2.text(5, 7, "LPDA: SNR_new = ({:.2f} * SNR_p2p)^2".format(coef_LPDA[1]), fontsize='x-small')
ax2.text(5, 4, "bicone: SNR_new = ({:.2f} * SNR_p2p)^2".format(coef_bicone[1]), fontsize='x-small')
ax.legend()
ax2.set_ylim(0, 250 / 10)
ax2.set_xlim(0, 150 / 10)
ax2.legend()
fig.tight_layout()
fig.savefig("plots/SNRcomparison.png".format(counter))
# plt.show()

# calculate trigger efficiency of ARIANNA Vp2p trigger
bins = np.arange(0, 150, 5)
bins_center = 0.5 * (bins[1:] + bins[:-1])

SNR_bicone = SS_bicone / Vrms ** 2 / units.ns
SNR_ARIANNA_bicone = Vp2p_bicone / Vrms / 2.
weights = (SNR_ARIANNA_bicone >= 3) * np.ones(len(SNR_ARIANNA_bicone))
Hcount, edges = np.histogram(SNR_bicone, bins=bins)
Hweight, edges = np.histogram(SNR_bicone, bins=bins, weights=weights)
trigger_efficiency_bicone = Hweight / Hcount

SNR_ARIANNA_LPDA = Vp2p_LPDA / Vrms / 2.
SNR_LPDA = SS_LPDA / Vrms ** 2 / units.ns
weights = (SNR_ARIANNA_LPDA >= 3) * np.ones(len(SNR_ARIANNA_LPDA))
Hcount, edges = np.histogram(SNR_LPDA, bins=bins)
Hweight, edges = np.histogram(SNR_LPDA, bins=bins, weights=weights)
trigger_efficiency_LPDA = Hweight / Hcount

Vrms_amp = (SS_LPDA_amp / SS_LPDA).mean() ** 0.5 * Vrms  # quick and dirty hack to get the Vrms after the amp response
SNR_ARIANNA_LPDA_amp = Vp2p_LPDA_amp / Vrms_amp / 2.
SNR_LPDA_amp = SS_LPDA_amp / Vrms_amp ** 2 / units.ns
weights = (SNR_ARIANNA_LPDA_amp >= 3) * np.ones(len(SNR_ARIANNA_LPDA_amp))
Hcount, edges = np.histogram(SNR_LPDA_amp, bins=bins)
Hweight, edges = np.histogram(SNR_LPDA_amp, bins=bins, weights=weights)
trigger_efficiency_LPDA_amp = Hweight / Hcount

fig, ax = plt.subplots(1, 1)
ax.plot(bins_center, trigger_efficiency_bicone, 'o-', label='bicone')
# ax.plot(bins_center, trigger_efficiency_LPDA, 's-', label='LPDA')
# ax.plot(bins_center, trigger_efficiency_LPDA_amp, 'd--', label='LPDA + amp')
ax.set_xticks(np.arange(0, 180, 20))
ax.set_xlim(0, 140)
ax.set_xlabel("new SNR")
ax.set_ylabel("trigger efficiency")
ax.set_title("trigger settings: Vp2p/Vrms/2 > 3, (3/8 trigger rate ~10mHz)")
ax.legend()
fig.tight_layout()
fig.savefig("plots/ARIANNAtriggerefficiency_allcangles.png".format(counter))

# with noise
SNR_ARIANNA_bicone_1 = Vp2p_bicone_noise_1 / Vrms / 2.
SNR_ARIANNA_bicone_2 = Vp2p_bicone_noise_2 / Vrms / 2.
weights_1 = (SNR_ARIANNA_bicone_1 >= 3) * np.ones(len(SNR_ARIANNA_bicone_1))
weights_2 = (SNR_ARIANNA_bicone_2 >= 3) * np.ones(len(SNR_ARIANNA_bicone_2))
weights = weights_1 * weights_2
Hcount, edges = np.histogram(SNR_bicone, bins=bins)
Hweight, edges = np.histogram(SNR_bicone, bins=bins, weights=weights)
trigger_efficiency_bicone = Hweight / Hcount
mask = (SNR_ARIANNA_bicone_1 >= 3) & (SNR_ARIANNA_bicone_2 >= 3)
bis = np.arange(0, 150, 5)
fig, ax = php.get_histograms([SNR_bicone, SNR_bicone[mask]], xlabels=['SNR all', 'SNR triggered'],
                             bins=[bins, bins])
fig.savefig('plots/ARIANNAtriggereffSNR_5ns.png')

weights_1 = (SNR_ARIANNA_bicone_1 >= 4) * np.ones(len(SNR_ARIANNA_bicone_1))
weights_2 = (SNR_ARIANNA_bicone_2 >= 4) * np.ones(len(SNR_ARIANNA_bicone_2))
weights = weights_1 * weights_2
Hcount, edges = np.histogram(SNR_bicone, bins=bins)
Hweight, edges = np.histogram(SNR_bicone, bins=bins, weights=weights)
trigger_efficiency_bicone_4s = Hweight / Hcount

weights_1 = (SNR_ARIANNA_bicone_1 >= 5) * np.ones(len(SNR_ARIANNA_bicone_1))
weights_2 = (SNR_ARIANNA_bicone_2 >= 5) * np.ones(len(SNR_ARIANNA_bicone_2))
weights = weights_1 * weights_2
Hcount, edges = np.histogram(SNR_bicone, bins=bins)
Hweight, edges = np.histogram(SNR_bicone, bins=bins, weights=weights)
trigger_efficiency_bicone_5s = Hweight / Hcount

SNR_ARIANNA_LPDA_1 = Vp2p_LPDA_noise_1 / Vrms / 2.
SNR_ARIANNA_LPDA_2 = Vp2p_LPDA_noise_2 / Vrms / 2.
weights_1 = (SNR_ARIANNA_LPDA_1 >= 3) * np.ones(len(SNR_ARIANNA_LPDA_1))
weights_2 = (SNR_ARIANNA_LPDA_2 >= 3) * np.ones(len(SNR_ARIANNA_LPDA_2))
weights = weights_1 * weights_2
Hcount, edges = np.histogram(SNR_LPDA, bins=bins)
Hweight, edges = np.histogram(SNR_LPDA, bins=bins, weights=weights)
trigger_efficiency_LPDA = Hweight / Hcount

weights_1 = (SNR_ARIANNA_LPDA_1 >= 4) * np.ones(len(SNR_ARIANNA_LPDA_1))
weights_2 = (SNR_ARIANNA_LPDA_2 >= 4) * np.ones(len(SNR_ARIANNA_LPDA_2))
weights = weights_1 * weights_2
Hcount, edges = np.histogram(SNR_LPDA, bins=bins)
Hweight, edges = np.histogram(SNR_LPDA, bins=bins, weights=weights)
trigger_efficiency_LPDA_4s = Hweight / Hcount

fig, ax = plt.subplots(1, 1)
ax.plot(bins_center, trigger_efficiency_bicone, 'o-', label='bicone 3sigma (with noise)')
ax.plot(bins_center, trigger_efficiency_bicone_4s, 'o-', label='bicone 4sigma (with noise)')
ax.plot(bins_center, trigger_efficiency_bicone_5s, 'o-', label='bicone 5sigma (with noise)')
# ax.plot(bins_center, trigger_efficiency_LPDA, 's--', label='LPDA 3sigma (with noise)')
# ax.plot(bins_center, trigger_efficiency_LPDA_4s, 's--', label='LPDA 4sigma (with noise)')
# ax.plot(bins_center, trigger_efficiency_LPDA_amp, 'd--', label='LPDA + amp')
ax.set_xticks(np.arange(0, 180, 20))
ax.set_xlim(0, 140)
ax.set_xlabel("new SNR")
ax.set_ylabel("trigger efficiency")
ax.set_title("trigger settings: Vp2p/Vrms/2 > N, 2/2 trigger")
ax.legend()
fig.tight_layout()
fig.savefig("plots/ARIANNAtriggerefficiency_allcangles_noise_5ns.png".format(counter))
plt.show()
