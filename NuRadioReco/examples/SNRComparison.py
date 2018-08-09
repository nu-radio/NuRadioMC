import numpy as np
import matplotlib.pyplot as plt
import logging
from NuRadioReco.utilities import units
import NuRadioReco.framework.channel
import NuRadioReco.framework.station
import NuRadioReco.framework.event
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.efieldToVoltageConverter
from NuRadioReco.modules.channelGenericNoiseAdder import channelGenericNoiseAdder

import NuRadioReco.modules.ARA.triggerSimulator
import NuRadioReco.modules.ARIANNA.triggerSimulator
from NuRadioMC.SignalGen import parametrizations as signalgen

from numpy.polynomial import polynomial as pol

from NuRadioReco.detector import detector


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


det = detector.Detector(json_filename='example_data/dummy_detector.json')
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("TriggerComparison")

n_index = 1.78
cherenkov_angle = np.arccos(1. / n_index)
dt = 1. * units.ns
N = 2 ** 8

Vrms = 11 * units.micro * units.V

NN = 1000
# now calculate the relation between the ARA SNR (integrated power ratio) and the ARIANNA SNR (Vp2p/2/Vrms)
import NuRadioReco.modules.ARA.triggerSimulator
triggerSimulator = NuRadioReco.modules.ARA.triggerSimulator.triggerSimulator()
min_freq, max_freq = 100 * units.MHz, .5 * units.GHz
power_mean, power_rms = get_ARA_power_mean_rms(1. / dt, Vrms, min_freq, max_freq)
print("power mean/rms = {:.4g}/{:.4g}".format(power_mean, power_rms))
counter = -1
SNRp2p_bicone = np.zeros(NN)
SNRara_bicone = np.zeros(NN)
for E in 10 ** np.linspace(15.5, 17, 100):
    for theta in np.linspace(cherenkov_angle - 0.1 * units.deg, cherenkov_angle + 0.1 * units.deg, 10):
        counter += 1
        pulse = signalgen.get_time_trace(E, theta, N, dt, 0, n_index, 1000 * units.m, 'Alvarez2000')
        event = NuRadioReco.framework.event.Event(1, 1)
        station = NuRadioReco.framework.station.Station(101)
        trace = np.zeros((3, N))
        trace[1] = pulse
        trace[2] = pulse
        sim_station = NuRadioReco.framework.sim_station.SimStation(101, sampling_rate=1. / dt, trace=trace)
        sim_station['zenith'] = (90 + 45) * units.deg
        sim_station['azimuth'] = 0
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
ax.scatter(SNRara_bicone, SNRp2p_bicone, s=20, alpha=0.5)
ax.set_xlabel("ARIANNA SNR (Vp2p/2/Vrms)")
ax.set_ylabel("ARA SNR (tunnel diode output/noise power RMS)")
ax.set_title("for ARA bicone response")
fig.tight_layout()
fig.savefig('plots/SNR_ARA_ARIANNA.png')
plt.show()

long_noise = channelGenericNoiseAdder().bandlimited_noise(min_freq=min_freq,
                                            max_freq=max_freq,
                                            n_samples=2 ** 20,
                                            sampling_rate=1 / dt,
                                            amplitude=Vrms,
                                            type='perfect_white')
a = 1 / 0

SS_LPDA = np.zeros(NN)
Vp2p_LPDA = np.zeros(NN)
SS_bicone = np.zeros(NN)
Vp2p_bicone = np.zeros(NN)
counter = -1
for E in 10 ** np.linspace(15.5, 17, 10):
    for theta in np.linspace(cherenkov_angle - 5 * units.deg, cherenkov_angle + 5 * units.deg, 10):
        counter += 1
        pulse = signalgen.get_time_trace(E, theta, N, dt, 0, n_index, 1000 * units.m, 'Alvarez2000')
        event = NuRadioReco.framework.event.Event(1, 1)
        station = NuRadioReco.framework.station.Station(101)
        trace = np.zeros((3, N))
        trace[1] = pulse
        trace[2] = pulse
        sim_station = NuRadioReco.framework.sim_station.SimStation(101, sampling_rate=1. / dt, trace=trace)
        sim_station['zenith'] = (90 + 45) * units.deg
        sim_station['azimuth'] = 0
        station.set_sim_station(sim_station)
        event.set_station(station)

        efieldToVoltageConverter.run(event, station, det)
        channelBandPassFilter.run(event, station, det, passband=[100 * units.MHz, 500 * units.MHz])

        trace_LPDA = station.get_channel(3).get_trace()
        trace_bicone = station.get_channel(2).get_trace()

        Vp2p_LPDA[counter] = (trace_LPDA.max() - trace_LPDA.min())
        Vp2p_bicone[counter] = (trace_bicone.max() - trace_bicone.min())

        SS_LPDA[counter] = np.sum(trace_LPDA ** 2) * dt
        SS_bicone[counter] = np.sum(trace_bicone ** 2) * dt

        if(counter < 0):  # plot some example traces
            fig, ax = plt.subplots(1, 2, sharey=True)
            ax = np.array(ax).flatten()
            tt = station.get_channel(3).get_times()
            ax[0].plot(tt / units.ns, trace_LPDA / units.micro / units.V)
            ax[1].plot(tt / units.ns, trace_bicone / units.micro / units.V)
            ax[0].set_ylabel('voltage [$\mu$V]')
            ax[0].set_title("LPDA")
            ax[1].set_title("bicone")
            ax[0].set_xlabel("time [ns]")
            ax[1].set_xlabel("time [ns]")
            fig.tight_layout()
            fig.suptitle(r"$\theta$ = {:.0f}, dCherenkov = {:.0f}".format(135, (theta - cherenkov_angle) / units.deg))
            fig.subplots_adjust(top=0.9)
            fig.savefig("plots/trace_LPDA_bicone_{:d}.png".format(counter))
#         plt.show()
            plt.close("all")

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
plt.close("all")

