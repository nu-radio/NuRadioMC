"""
Run NuRadioMC emitter simulation for RNO-G calibration pulser.

Propagates the calibration pulse through Greenland ice, applies the full
RNO-G hardware response chain (amplifiers, filters, cable delays), and
triggers with the RNO-G deep high-low threshold trigger.

Usage:
    python A02RunSimulation.py events.hdf5 config.yaml output.hdf5 [output.nur]
    python A02RunSimulation.py events.hdf5 config.yaml output.hdf5 output.nur --detector-file det.json
"""

import argparse
import logging
from datetime import datetime

import numpy as np

from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator, triggerBoardResponse
from NuRadioReco.modules.trigger import highLowThreshold
from NuRadioReco.utilities import units, signal_processing
from NuRadioMC.simulation import simulation

logger = logging.getLogger(__name__)

TRIGGER_CHANNELS = [0, 1, 2, 3]


def high_low_threshold_from_rate(log_rate_per_hz):
    """Convert a target trigger rate to a high-low threshold in sigma.

    Parameterization of the RNO-G FLOWER board high-low trigger threshold
    for VPol antennas with iglu + flower_lp hardware chain, as a function
    of log10 singles rate (per Hz). Same parameterization used in the
    official RNO-G trigger simulation example
    (NuRadioMC/examples/RNO_G_trigger_simulation/simulate.py).

    From Alan Coleman's trigger simulation study:
    https://radio.uchicago.edu/wiki/images/e/e6/2023.10.11_Simulating_RNO-G_Trigger.pdf

    Returns
    -------
    float
        Threshold in units of noise sigma.
    """
    return (-859 + np.sqrt(39392706 - 3602500 * log_rate_per_hz)) / 1441.0


class RNOGPulserSimulation(simulation.simulation):
    """NuRadioMC simulation subclass with RNO-G hardware response and trigger."""

    def __init__(self, *args, **kwargs):
        self._hw = hardwareResponseIncorporator.hardwareResponseIncorporator()
        self._hw.begin(trigger_channels=TRIGGER_CHANNELS)

        self._adc = triggerBoardResponse.triggerBoardResponse()
        self._adc.begin(clock_offset=0.0, adc_output="counts")

        self._trigger = highLowThreshold.triggerSimulator()

        super().__init__(*args, **kwargs)

    def _detector_simulation_filter_amp(self, evt, station, det):
        self._hw.run(evt, station, det, sim_to_data=True)

    def _detector_simulation_trigger(self, evt, station, det):
        # Compute thermal noise Vrms on trigger channels from detector response
        trigger_vrms = []
        for ch_id in TRIGGER_CHANNELS:
            resp = det.get_signal_chain_response(station.get_id(), ch_id, trigger=True)
            trigger_vrms.append(
                signal_processing.calculate_vrms_from_temperature(
                    temperature=300, response=resp))
        trigger_vrms = np.array(trigger_vrms)

        # Apply FLOWER trigger board response (filter + ADC digitization)
        vrms_after_gain = self._adc.run(
            evt, station, det,
            trigger_channels=TRIGGER_CHANNELS,
            vrms=trigger_vrms,
            digitize_trace=True,
        )

        # 1 Hz threshold in sigma
        threshold_sigma = high_low_threshold_from_rate(0)
        flower_rate = station.get_trigger_channel(
            TRIGGER_CHANNELS[0]).get_sampling_rate()

        threshold_high = {ch: int(round(threshold_sigma * vrms))
                          for ch, vrms in zip(TRIGGER_CHANNELS, vrms_after_gain)}
        threshold_low = {ch: int(round(-threshold_sigma * vrms))
                         for ch, vrms in zip(TRIGGER_CHANNELS, vrms_after_gain)}

        self._trigger.run(
            evt, station, det,
            threshold_high=threshold_high,
            threshold_low=threshold_low,
            use_digitization=False,
            high_low_window=6 / flower_rate,
            coinc_window=20 / flower_rate,
            number_concidences=2,
            triggered_channels=TRIGGER_CHANNELS,
            trigger_name="deep_high_low_1Hz",
            pre_trigger_time=200 * units.ns,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RNO-G pulser simulation')
    parser.add_argument('inputfilename', type=str,
                        help='Path to HDF5 input events (from A01generate_pulser_events.py)')
    parser.add_argument('config', type=str,
                        help='NuRadioMC YAML config file')
    parser.add_argument('outputfilename', type=str,
                        help='HDF5 output filename')
    parser.add_argument('outputfilenameNuRadioReco', type=str, nargs='?', default=None,
                        help='Output .nur filename for full waveform data')
    parser.add_argument('--detector-file', type=str, default=None,
                        help='Path to exported RNO-G detector JSON file. '
                             'If not provided, queries MongoDB directly.')
    parser.add_argument('--station', type=int, default=None,
                        help='Station ID to select (speeds up detector loading)')
    args = parser.parse_args()

    from NuRadioReco.detector.RNO_G.rnog_detector import Detector
    det_kwargs = dict(log_level=logging.INFO)
    if args.detector_file:
        det_kwargs['detector_file'] = args.detector_file
    if args.station is not None:
        det_kwargs['select_stations'] = args.station

    try:
        det = Detector(**det_kwargs)
    except Exception as e:
        if args.detector_file:
            raise
        print(f"MongoDB connection failed: {e}")
        print("Try using --detector-file with an exported detector JSON.")
        raise SystemExit(1)

    det.update(datetime(2022, 8, 1))

    sim = RNOGPulserSimulation(
        inputfilename=args.inputfilename,
        outputfilename=args.outputfilename,
        det=det,
        outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
        config_file=args.config,
        evt_time=datetime(2022, 8, 1),
        file_overwrite=True,
    )

    n_triggered = sim.run()
    print(f"Triggered: {n_triggered}")
