import numpy as np
import matplotlib.pyplot as plt
import logging
from NuRadioReco.utilities import units
import NuRadioReco.framework.channel
import NuRadioReco.framework.station
import NuRadioReco.framework.event
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.channelLengthAdjuster
import NuRadioReco.utilities.diodeSimulator
import NuRadioReco.modules.ARA.triggerSimulator
import NuRadioReco.modules.trigger.highLowThreshold
from NuRadioReco.modules.base import module
from NuRadioReco.detector import detector
from NuRadioReco.framework.parameters import channelParameters as chp
det = detector.Detector(json_filename='../example_data/dummy_detector.json')

logger = module.setup_logger(level=logging.WARNING)

channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin(debug=False)
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
channelSignalReconstructor.begin(debug=False)
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelLengthAdjuster = NuRadioReco.modules.channelLengthAdjuster.channelLengthAdjuster()
channelLengthAdjuster.begin(number_of_samples=400, offset=50)


diodeSimulator = NuRadioReco.utilities.diodeSimulator.diodeSimulator()
noise_mean, noise_std = diodeSimulator.calculate_noise_parameters(
    amplitude=20 * units.mV,
    min_freq=50 * units.MHz,
    max_freq=1000 * units.MHz,
    sampling_rate=1. * units.GHz
)
triggerSimulator_ARA = NuRadioReco.modules.ARA.triggerSimulator.triggerSimulator()
triggerSimulator_ARIANNA = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
triggerSimulator_ARIANNA.begin()

event_ARA = NuRadioReco.framework.event.Event(1, 1)
event_ARIANNA = NuRadioReco.framework.event.Event(1, 1)

station_ARA = NuRadioReco.framework.station.Station(101)
channel_ARA = NuRadioReco.framework.channel.Channel(0)

station_ARIANNA = NuRadioReco.framework.station.Station(101)
channel_ARIANNA = NuRadioReco.framework.channel.Channel(0)

# Switch between cosmic ray pulse (CoREAS) and Askaryan parameterization
CR = False

# TYPE_SNR = 'integrated_power'
TYPE_SNR = 'peak_amplitude'

if CR:
    # Use numpy array with band-pass limited pulse [30-1000] MHz, samples at 10 GHz
    data = np.load("example_data/Test_data_8.npy")
    test_pulse = data[2, :]
    # Normalize test pulse to 1
else:
    # PARAMEETERS for Askaryan Pulse
    energy = 1e19 * units.eV
    fhad = 0.5
    viewing_angle = 54 * units.degree
    n_samples = 2**12
    dt = 0.1 * units.ns
    n_index = 1.5
    R = 100 * units.m

    from NuRadioMC.SignalGen import parametrizations as signalgen
    test_pulse = signalgen.get_time_trace(energy * fhad, viewing_angle, n_samples, dt, 'HAD', n_index, R, 'Alvarez2000')


test_pulse /= np.max(np.abs(test_pulse))

n_scaling = 50
result_ARA = np.zeros((n_scaling, 2))
result_ARIANNA = np.zeros((n_scaling, 2))

i = 0
n_iter = 20

for scaling in np.linspace(10 * units.mV, 200 * units.mV, n_scaling):
    test_pulse_sc = test_pulse * scaling

    n_trigger_ARA = 0
    n_trigger_ARIANNA = 0

    SNR_ARA = 0
    SNR_ARIANNA = 0

    max = []
    for n in range(n_iter):

        channel_ARA.set_trace(test_pulse_sc, 10 * units.GHz)
        station_ARA.add_channel(channel_ARA)
        station_ARA.remove_triggers()

        channel_ARIANNA.set_trace(test_pulse_sc, 10 * units.GHz)
        station_ARIANNA.add_channel(channel_ARIANNA)
        station_ARIANNA.remove_triggers()

        channelBandPassFilter.run(
            event_ARIANNA,
            station_ARIANNA,
            det, passband=[50 * units.MHz, 1000 * units.MHz],
            filter_type='rectangular'
        )
        channelBandPassFilter.run(
            event_ARA,
            station_ARA,
            det,
            passband=[50 * units.MHz, 1000 * units.MHz],
            filter_type='rectangular'
        )

        channelGenericNoiseAdder.run(
            event_ARA,
            station_ARA,
            det,
            amplitude=20 * units.mV,
            min_freq=50 * units.MHz,
            max_freq=1000 * units.MHz,
            type='perfect_white'
        )
        channelGenericNoiseAdder.run(
            event_ARIANNA,
            station_ARIANNA,
            det,
            amplitude=20 * units.mV,
            min_freq=50 * units.MHz,
            max_freq=1000 * units.MHz,
            type='perfect_white'
        )

        channelResampler.run(event_ARA, station_ARA, det, sampling_rate=1 * units.GHz)
        channelResampler.run(event_ARIANNA, station_ARIANNA, det, sampling_rate=1 * units.GHz)

        channelSignalReconstructor.run(event_ARIANNA, station_ARIANNA, det)
        channelSignalReconstructor.run(event_ARA, station_ARA, det)

        channelLengthAdjuster.run(event_ARIANNA, station_ARIANNA, det)
        channelLengthAdjuster.run(event_ARA, station_ARA, det)

        triggerSimulator_ARA.run(
            event_ARA,
            station_ARA,
            det,
            power_threshold=6.5,
            coinc_window=110 * units.ns,
            number_concidences=1,
            triggered_channels=[0, 1, 2, 3, 4, 5, 6, 7],
            power_mean=noise_mean,
            power_std=noise_std
        )

        triggerSimulator_ARIANNA.run(
            event_ARIANNA,
            station_ARIANNA,
            det,
            threshold_high=36 * units.mV,
            threshold_low=-36 * units.mV,
            high_low_window=20 * units.ns,
            coinc_window=32 * units.ns,
            number_concidences=1,
            triggered_channels=[0, 1, 2, 3]
        )

#
        SNR_ARA += station_ARA.get_channel(0)[chp.SNR][TYPE_SNR]

        SNR_ARIANNA += station_ARIANNA.get_channel(0)[chp.SNR][TYPE_SNR]

        max.append(np.max(np.abs(station_ARA.get_channel(0).get_trace())))

        if station_ARA.has_triggered():
            n_trigger_ARA += 1.
        if station_ARIANNA.has_triggered():
            n_trigger_ARIANNA += 1.

    result_ARA[i, 0] = SNR_ARA / n_iter
    result_ARA[i, 1] = n_trigger_ARA / n_iter

    result_ARIANNA[i, 0] = SNR_ARIANNA / n_iter
    result_ARIANNA[i, 1] = n_trigger_ARIANNA / n_iter
    i += 1


plt.figure()
plt.plot(result_ARA[:, 0], result_ARA[:, 1], linestyle='None', marker='o', label="ARA, power_threshold 6.5")
plt.plot(result_ARIANNA[:, 0], result_ARIANNA[:, 1], linestyle='None', marker='s', label="ARIANNA, 3 sigma")
plt.ylabel("Trigger efficiency on one antenna")
plt.xlabel(TYPE_SNR)
plt.legend()
plt.show()
