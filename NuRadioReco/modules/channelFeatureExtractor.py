from NuRadioReco.modules.base.module import register_run
from NuRadioReco.framework.parameters import channelParameters
from NuRadioReco.utilities import units
import NuRadioReco.utilities.trace_utilities as trace_utils

import numpy as np

import logging
logger = logging.getLogger("NuRadioReco.channelFeatureExtractor")


class channelFeatureExtractor:
    """
    Per-channel feature extraction module.

    Wraps the per-channel primitives in
    ``NuRadioReco.utilities.trace_utilities`` in the standard
    ``begin``/``run``/``end`` module interface. Computes a configurable
    set of scalar features for every channel on the station (or a
    user-specified subset) and optionally writes the "standard" ones
    onto the channel via ``channelParameters``.

    This module is experiment-agnostic. Aggregation across channel
    groupings (coherent sums over a phased array, per-string averages,
    HDF5 column naming conventions) belongs in analysis-specific driver
    scripts; see, e.g.,
    ``NuRadioReco/examples/RNOG/feature_extraction_ex/feature_extraction.py``
    for the RNO-G wrapping of this module.

    Feature groups (controlled by ``feature_groups`` in the config):

    - ``snr``: split-trace noise RMS and peak-amplitude SNR
    - ``rpr``: root-power ratio (needs ``snr`` to have run for noise RMS)
    - ``max_amplitude``: std-normalized peak-to-peak amplitude and
      Hilbert-envelope peak
    - ``impulsivity``: NRMC impulsivity + extended impulsivity (slope,
      R^2, intercept, KS statistic) on the raw trace
    - ``kurtosis_entropy``: kurtosis and Shannon entropy
    - ``spectral``: centroid, bandwidth, skewness, kurtosis, entropy,
      slope, 90% rolloff, flatness, low-band fraction
    - ``impulse_correlations``: max correlation with delta, bipolar,
      gaussian, bipolar_wide, sinc templates

    Output: a ``{channel_id: {feature_name: value}}`` dict, also stored
    as ``self.results`` and retrievable via ``get_results()``.

    See Also
    --------
    NuRadioReco.utilities.trace_utilities
    """

    _DEFAULT_CONFIG = {
        "feature_groups": None,
        "simple_noise_rms": False,
        "n_entropy_bins": 50,
        "spectral_fmin": 0.08 * units.GHz,
        "spectral_fmax": 0.6 * units.GHz,
        "spectral_low_band_boundary": 0.1 * units.GHz,
        "set_channel_parameters": True,
    }

    AVAILABLE_GROUPS = (
        "snr",
        "rpr",
        "max_amplitude",
        "impulsivity",
        "kurtosis_entropy",
        "spectral",
        "impulse_correlations",
    )

    def __init__(self):
        self._config = dict(self._DEFAULT_CONFIG)
        self.results = {}

    def begin(self, config=None):
        """Merge user config over defaults.

        Parameters
        ----------
        config : dict, optional
            Overrides for ``_DEFAULT_CONFIG``. Unknown keys are
            preserved for inspection but ignored by this module.
        """
        self._config = dict(self._DEFAULT_CONFIG)
        if config:
            self._config.update(config)

    @register_run()
    def run(self, event, station, det=None, channel_ids=None):
        """Compute features for every (selected) channel on the station.

        Parameters
        ----------
        event : NuRadioReco.framework.event.Event
        station : NuRadioReco.framework.station.Station
        det : Detector, optional
            Accepted for interface symmetry; not used.
        channel_ids : iterable of int, optional
            If given, restrict feature extraction to these channel IDs.
            Otherwise, all channels on the station are processed.

        Returns
        -------
        dict
            ``{channel_id: {feature_name: value}}``.
        """
        cfg = self._config
        groups = cfg.get("feature_groups")
        compute_all = groups is None
        groups = set(groups) if groups is not None else set()
        selected = None if channel_ids is None else set(channel_ids)

        self.results = {}
        for channel in station.iter_channels():
            ch_id = channel.get_id()
            if selected is not None and ch_id not in selected:
                continue

            trace = channel.get_trace()
            sampling_rate = channel.get_sampling_rate()

            features = self._compute_channel_features(
                trace, sampling_rate, cfg, compute_all, groups)
            self.results[ch_id] = features

            if cfg["set_channel_parameters"]:
                self._set_channel_parameters(channel, features)

        return self.results

    def end(self):
        """No-op."""
        pass

    def get_results(self):
        """Return the per-channel feature dict from the most recent run."""
        return self.results

    def _compute_channel_features(self, trace, sampling_rate, cfg,
                                  compute_all, groups):
        """Compute features for a single channel trace."""
        row = {}

        noise_rms = None
        if compute_all or "snr" in groups or "rpr" in groups:
            if cfg["simple_noise_rms"]:
                noise_rms = float(np.std(trace))
            else:
                noise_rms = float(trace_utils.get_split_trace_noise_RMS(trace))
            row["noise_rms"] = noise_rms

        if compute_all or "snr" in groups:
            row["snr"] = float(
                trace_utils.get_signal_to_noise_ratio(trace, noise_rms))

        if compute_all or "rpr" in groups:
            times = np.arange(len(trace)) / sampling_rate
            row["root_power_ratio"] = float(
                trace_utils.get_root_power_ratio(trace, times, noise_rms))

        need_envelope = (compute_all
                         or "max_amplitude" in groups
                         or "impulsivity" in groups)
        envelope = trace_utils.get_hilbert_envelope(trace) if need_envelope else None

        if compute_all or "max_amplitude" in groups:
            std = np.std(trace)
            normalized = trace / std if std > 0 else trace
            row["max_amplitude_norm"] = float(np.amax(
                trace_utils.get_maximum_peak_to_peak_amplitude(normalized)))
            row["max_amplitude_envelope"] = float(np.amax(envelope))

        if compute_all or "impulsivity" in groups:
            row["impulsivity_nrmc"] = float(
                trace_utils.get_impulsivity(trace, envelope=envelope))
            ext = trace_utils.get_extended_impulsivity(trace, envelope=envelope)
            row["impulsivity"] = ext["impulsivity_custom"]
            row["impulsivity_r_squared"] = ext["impulsivity_r_squared"]
            row["impulsivity_slope"] = ext["impulsivity_slope"]
            row["impulsivity_intercept"] = ext["impulsivity_intercept"]
            row["impulsivity_ks_statistic"] = ext["impulsivity_ks_statistic"]

        if compute_all or "kurtosis_entropy" in groups:
            row["kurtosis"] = float(trace_utils.get_kurtosis(trace))
            row["entropy"] = float(trace_utils.get_entropy(
                trace, n_hist_bins=cfg["n_entropy_bins"]))

        if compute_all or "spectral" in groups:
            spec = trace_utils.get_spectral_features(
                trace, sampling_rate,
                fmin=cfg["spectral_fmin"],
                fmax=cfg["spectral_fmax"],
                low_band_boundary=cfg["spectral_low_band_boundary"],
            )
            for key, val in spec.items():
                row[key] = float(val)

        if compute_all or "impulse_correlations" in groups:
            corrs = trace_utils.get_impulse_template_correlations(
                trace, sampling_rate)
            for name, val in corrs.items():
                row[f"impulse_corr_{name}"] = float(val)

        return row

    def _set_channel_parameters(self, channel, features):
        """Store standard per-channel features via ``channelParameters``."""
        if "snr" in features:
            channel.set_parameter(channelParameters.SNR,
                                  {"peak_2_peak_amplitude_split_noise_rms": features["snr"]})
        if "noise_rms" in features:
            channel.set_parameter(channelParameters.noise_rms,
                                  features["noise_rms"])
        if "max_amplitude_envelope" in features:
            channel.set_parameter(channelParameters.maximum_amplitude_envelope,
                                  features["max_amplitude_envelope"])
        if "root_power_ratio" in features:
            channel.set_parameter(channelParameters.root_power_ratio,
                                  features["root_power_ratio"])
        if "impulsivity" in features:
            channel.set_parameter(channelParameters.impulsivity,
                                  features["impulsivity"])
        if "entropy" in features:
            channel.set_parameter(channelParameters.entropy,
                                  features["entropy"])
        if "kurtosis" in features:
            channel.set_parameter(channelParameters.kurtosis,
                                  features["kurtosis"])
