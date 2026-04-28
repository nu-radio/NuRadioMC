from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units

import NuRadioReco.modules.RNO_G.channelBlockOffsetFitter
import NuRadioReco.modules.RNO_G.channelGlitchDetector
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.channelAddCableDelay
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelSinewaveSubtraction
import NuRadioReco.modules.channelResampler

import logging
logger = logging.getLogger('NuRadioReco.RNO_G.channelPreprocessor')


class channelPreprocessor:
    """
    RNO-G waveform preprocessing chain as a single composable module.

    Wraps the standard sequence of per-event processing steps so that the
    chain lives in one place and can be reused across different readers
    (``dataProviderRNOG`` for ROOT, ``dataProviderNuRadio`` for NUR, or
    ad-hoc pipelines). Each step is gated by a flag in the config passed
    to ``begin`` so callers can opt in or out without reimplementing the
    wiring.

    Steps, in order:

    1. ``channelBlockOffsetFitter`` (fit + subtract LAB4D block offsets)
    2. ``channelGlitchDetector`` (flag scrambled readout blocks; does not
       fix them, only sets ``channelParameter.glitch``)
    3. ``channelAddCableDelay`` (subtract cable delays)
    4. ``hardwareResponseIncorporator`` (invert hardware phase response;
       angle-independent, unlike antenna dedispersion which is reco-only)
    5. ``channelResampler`` (upsample to a target rate, typically 5 GHz)
    6. ``channelSinewaveSubtraction`` (CW peak removal)
    7. ``channelBandPassFilter`` (apply analysis passband)

    Block-offset removal is on by default. Glitch detection and steps
    4-7 are off by default.

    See Also
    --------
    NuRadioReco.modules.RNO_G.dataProviderRNOG
    NuRadioReco.modules.RNO_G.channelBlockOffsetFitter
    NuRadioReco.modules.RNO_G.channelGlitchDetector
    NuRadioReco.modules.channelAddCableDelay
    NuRadioReco.modules.channelResampler
    NuRadioReco.modules.channelSinewaveSubtraction
    NuRadioReco.modules.channelBandPassFilter
    """

    _DEFAULT_CONFIG = {
        "apply_block_offset_removal": True,
        "apply_glitch_detection": False,
        "apply_cable_delay": True,
        "cable_delay_mode": "subtract",
        "apply_hw_phase_removal": False,
        "hw_phase_mode": "phase_only",
        "hw_phase_sim_to_data": False,
        "apply_upsampling": False,
        "target_sampling_rate": 5.0 * units.GHz,
        "apply_cw_removal": False,
        "cw_peak_prominence": 4.0,
        "cw_freq_band": (0.1, 0.6),
        "cw_algorithm": "sliding",
        "apply_bandpass": False,
        "bandpass_band": (0.1 * units.GHz, 0.7 * units.GHz),
        "bandpass_filter_type": "butter",
        "bandpass_order": 10,
        "glitch_cut_value": 0.0,
    }

    def __init__(self):
        self._block_offset = NuRadioReco.modules.RNO_G.channelBlockOffsetFitter.channelBlockOffsets()
        self._glitch_detector = None
        self._cable_delay = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()
        self._hw_response = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
        self._resampler = NuRadioReco.modules.channelResampler.channelResampler()
        self._cw_filter = NuRadioReco.modules.channelSinewaveSubtraction.channelSinewaveSubtraction()
        self._bandpass = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
        self._config = dict(self._DEFAULT_CONFIG)

    def begin(self, config=None):
        """Initialize submodules with merged defaults + user config.

        Parameters
        ----------
        config : dict, optional
            Per-step flags and parameters. Keys override the class
            defaults (`_DEFAULT_CONFIG`). Unknown keys are ignored by
            this module but preserved on `self._config` for inspection.
        """
        if config:
            self._config.update(config)
        cfg = self._config

        self._glitch_detector = NuRadioReco.modules.RNO_G.channelGlitchDetector.channelGlitchDetector(
            cut_value=cfg["glitch_cut_value"]
        )

        self._block_offset.begin()
        self._glitch_detector.begin()
        self._cable_delay.begin()
        self._hw_response.begin()
        self._resampler.begin()
        self._bandpass.begin()
        self._cw_filter.begin(
            save_filtered_freqs=False,
            freq_band=tuple(cfg["cw_freq_band"]),
        )

    def end(self):
        """Call end on submodules that maintain per-run state."""
        self._block_offset.end()
        if self._glitch_detector is not None:
            self._glitch_detector.end()
        self._resampler.end()

    @register_run()
    def run(self, event, station, det):
        """Apply the enabled preprocessing steps in order.

        Parameters
        ----------
        event : NuRadioReco.framework.event.Event
        station : NuRadioReco.framework.station.Station
        det : Detector
        """
        cfg = self._config

        if cfg["apply_block_offset_removal"]:
            self._block_offset.run(event, station, det)

        if cfg["apply_glitch_detection"]:
            self._glitch_detector.run(event, station, det)

        if cfg["apply_cable_delay"]:
            self._cable_delay.run(event, station, det, mode=cfg["cable_delay_mode"])

        if cfg["apply_hw_phase_removal"]:
            self._hw_response.run(
                event, station, det,
                sim_to_data=cfg["hw_phase_sim_to_data"],
                mode=cfg["hw_phase_mode"],
            )

        if cfg["apply_upsampling"]:
            self._resampler.run(event, station, det,
                                sampling_rate=cfg["target_sampling_rate"])

        if cfg["apply_cw_removal"]:
            self._cw_filter.run(event, station, det,
                                algorithm=cfg["cw_algorithm"],
                                peak_prominence=cfg["cw_peak_prominence"])

        if cfg["apply_bandpass"]:
            self._bandpass.run(
                event, station, det,
                passband=tuple(cfg["bandpass_band"]),
                filter_type=cfg["bandpass_filter_type"],
                order=cfg["bandpass_order"],
            )
