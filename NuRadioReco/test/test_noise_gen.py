import numpy as np

from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.utilities.noise import thermalNoiseGenerator
from NuRadioReco.utilities import units, fft


n_samples = 1000
sampling_rate = 1 * units.GHz
Vrms = 1
threshold = Vrms * 2
time_coincidence = 5 * units.ns
n_majority = 2
time_coincidence_majority = 40 * units.ns
n_channels = 10
trigger_time = 0.2 * n_samples / sampling_rate

filt = np.zeros

cBPF = channelBandPassFilter()
frequencies = np.fft.rfftfreq(n_samples, 1.0 / sampling_rate)
filt = cBPF.get_filter(
    frequencies, None, 0, None, passband=[sampling_rate * 0.1, sampling_rate * 0.3], filter_type="butter", order=10
)


generator = thermalNoiseGenerator(
    n_samples,
    sampling_rate,
    Vrms,
    threshold,
    time_coincidence,
    n_majority,
    time_coincidence_majority,
    n_channels,
    trigger_time,
    filt,
    noise_type="rayleigh",
    keep_full_band=False,
)
traces_baseline = generator.generate_noise()
specs_baseline = np.median(np.abs(fft.time2freq(traces_baseline, sampling_rate)), axis=0)


generator = thermalNoiseGenerator(
    n_samples,
    sampling_rate,
    Vrms,
    threshold,
    time_coincidence,
    n_majority,
    time_coincidence_majority,
    n_channels,
    trigger_time,
    filt,
    noise_type="rayleigh",
    keep_full_band=True,
)
traces_fb = generator.generate_noise()
specs_fb = np.median(np.abs(fft.time2freq(traces_fb, sampling_rate)), axis=0)

icheck = int(len(specs_fb) * 0.8)
assert np.sum(specs_baseline[icheck:] > specs_fb[icheck:]) == 0
