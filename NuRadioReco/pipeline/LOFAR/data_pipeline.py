#!/usr/bin/env python3
"""
NuRadioReco-native LOFAR data pipeline.

The base LOFAR pipeline takes a LOFAR event ID and a detector description JSON file as input, and produces a .nur file containing the reconstructed shower parameters and station parameters. The pipeline also produces debug plots if requested.

It uses the IFT reconstruction module to reconstruct the shower parameters from the LOFAR data. The IFT reconstruction module uses the NIFTy library to perform the reconstruction. This is the base data pipeline for LOFAR data analysis, and can be used as a template for other pipelines.

.. moduleauthor:: Karen Terveer <karen.terveer@fau.de> & Keito Watanabe <keito.watanabe@kit.edu>
"""

import argparse
import logging
import os

try:
    import jax

    # Double precision has to be enabled before any other module creates a JAX array,
    # so the imports below deliberately do not sit at the top of the file (PEP 8 E402).
    jax.config.update("jax_enable_x64", True)
except ImportError:
    jax = None

import numpy as np  # noqa: E402

import NuRadioReco.detector.detector  # noqa: E402
from NuRadioReco.framework.parameters import stationParameters, showerParameters  # noqa: E402
from NuRadioReco.modules.LOFAR import dataEventGenerator  # noqa: E402
from NuRadioReco.modules import voltageToEfieldConverter  # noqa: E402
from NuRadioReco.modules.LOFAR import pipelineVisualizer_LOFAR  # noqa: E402
import NuRadioReco.modules.io.eventWriter  # noqa: E402
from NuRadioReco.modules.LOFAR import iftReconstructor  # noqa: E402
from NuRadioReco.modules.LOFAR.iftReconstructor import (  # noqa: E402
    _DEFAULT_N_VI_ITERATIONS, _DEFAULT_N_SAMPLES,
    _EARLY_ABORT_XMAX_STD_GCM2, _EARLY_ABORT_XMAX_AFTER_ITERS, _EARLY_ABORT_MAX_FLUENCE,
)
from NuRadioReco.utilities.LOFAR.iftDataHelpers import MAX_SIGNAL_SNR_THRESHOLD  # noqa: E402
from NuRadioReco.utilities.LOFAR.macros import GDAS_ATMOSPHERE_DIRECTORY  # noqa: E402
from NuRadioReco.utilities import units  # noqa: E402

LOGGER = logging.getLogger("NuRadioReco.pipeline.LOFAR.data_pipeline")


def _convert_voltage_to_efield(event, detector):
    converter = voltageToEfieldConverter.voltageToEfieldConverter()
    converter.begin()

    for station in event.get_stations():
        if not station.has_parameter(stationParameters.zenith) or \
                not station.has_parameter(stationParameters.azimuth):
            shower = next(event.get_showers())
            if shower.has_parameter(showerParameters.zenith) and shower.has_parameter(showerParameters.azimuth):
                station.set_parameter(stationParameters.zenith, shower.get_parameter(showerParameters.zenith))
                station.set_parameter(stationParameters.azimuth, shower.get_parameter(showerParameters.azimuth))
            else:
                station.set_parameter(stationParameters.zenith, 0.0 * units.radian)
                station.set_parameter(stationParameters.azimuth, 0.0 * units.radian)

        for group_id in station.get_channel_ids(return_group_ids=True):
            use_channels = [channel.get_id() for channel in station.iter_channel_group(group_id)]
            if len(use_channels) < 2:
                continue
            try:
                converter.run(event, station, detector, use_channels=use_channels)
            except Exception as exc:
                LOGGER.warning(
                    "Skipping e-field conversion for station %s group %s: %s",
                    station.get_id(), group_id, exc
                )

    converter.end()

def _write_params_nur(event, output_dir):
    """Write a .nur file containing only shower/station parameters (no channel traces or e-fields)."""
    for station in event.get_stations():
        for ch_id in list(station.get_channel_ids()):
            station.remove_channel(ch_id)
        station.set_electric_fields([])
    path = os.path.join(output_dir, f"{event.get_id()}.nur")
    writer = NuRadioReco.modules.io.eventWriter.eventWriter()
    writer.begin(path)
    writer.run(event)
    writer.end()
    LOGGER.info("Wrote parameters-only .nur to %s", path)


def _save_pipeline_visualizer_plots(event, detector, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    visualizer = pipelineVisualizer_LOFAR.pipelineVisualizer()
    visualizer.begin()
    try:
        visualizer.run(event, detector, save_dir=output_dir, polarization=True, direction=True)
    except Exception as exc:
        LOGGER.warning("Skipping final LOFAR visualizer debug plots: %s", exc)
    finally:
        visualizer.end()
    

def run_pipeline(args):

    if jax is None:
        raise RuntimeError("JAX is required for the LOFAR pipeline, but it is not installed.")

    # manually generate detector with manual json file
    detector = NuRadioReco.detector.detector.Detector(
        args.detector,
        source="json",
        antenna_by_depth=False,
    )

    # process the event with the dataEventGenerator module
    # note that the event is not written to a .nur file yet, since we want to run the IFT reconstructor on it first. The processed event is returned by the dataEventGenerator module.
    LOGGER.info("Running dataEventGenerator module for event %d", args.event_id)
    data_event_generator = dataEventGenerator(detector, output_directory = args.output_dir)
    processed_event = dataEventGenerator.process_event(args.event_id, save_debug_plots = args.debug_plots, write_event = False)

    # for now fixed, in the future we should be able to replace this
    reconstructor = iftReconstructor.iftReconstructor()


    LOGGER.info("Running voltage-to-electric-field conversion")
    _convert_voltage_to_efield(processed_event, detector)
    if args.debug_plots:
        _save_pipeline_visualizer_plots(processed_event, detector, data_event_generator.debug_dir)

    LOGGER.info("Running IFT reconstruction module")
    recon_kwargs = dict(
        enable_fluence_correlated_field=args.enable_fluence_correlated_field,
        enable_timing_correlated_field=args.enable_timing_correlated_field,
        export_posterior_samples=args.export_posterior_samples,
        output_directory=args.output_dir,
        debug_plots=args.debug_plots,
        debug_plot_dir=data_event_generator.debug_dir,
        run_nifty=not args.no_nifty,
        step_deg=1.0,
        atmosphere_dir=args.atmosphere_dir,
        gdas_cache_dir=args.gdas_cache_dir,
        max_signal_fallback=args.max_signal_fallback,
        max_signal_snr_threshold=args.max_signal_snr,
        early_abort_xmax_std_gcm2=args.early_abort_xmax_std,
        early_abort_max_fluence=args.early_abort_max_fluence,
    )
    if args.ift_iterations is not None:
        recon_kwargs["n_iterations"] = args.ift_iterations
    if args.ift_samples is not None:
        recon_kwargs["n_samples"] = args.ift_samples
    reconstructor.begin(**recon_kwargs)
    reconstructor.run(processed_event, detector)
    reconstructor.end()

    if args.output_nur:
        writer = NuRadioReco.modules.io.eventWriter.eventWriter()
        writer.begin(args.output_nur)
        writer.run(processed_event)
        writer.end()

    os.makedirs(args.output_dir, exist_ok=True)
    _write_params_nur(processed_event, args.output_dir)

    return processed_event


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("event_id", type=int)

    parser.add_argument("--ift-iterations", type=int, default=None,
                        help=f"Number of VI iterations (default: {_DEFAULT_N_VI_ITERATIONS})")
    parser.add_argument("--ift-samples", type=int, default=None,
                        help=f"Number of VI samples per iteration (default: {_DEFAULT_N_SAMPLES})")
    parser.add_argument("--enable-fluence-correlated-field",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-timing-correlated-field",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-signal-fallback",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="When both direction searches find no timing data at all, "
                             "search every channel blind for its largest excursion "
                             "instead of aborting. Only ever runs where the pipeline "
                             "would otherwise produce nothing.")
    parser.add_argument("--max-signal-snr", type=float,
                        default=MAX_SIGNAL_SNR_THRESHOLD,
                        help="Per-antenna envelope SNR floor for the blind max-signal "
                             "fallback (default: %(default)s, calibrated so that ~1.5%% "
                             "of pure-noise antennas pass).")

    parser.add_argument("--early-abort-xmax-std", type=float,
                        default=_EARLY_ABORT_XMAX_STD_GCM2,
                        help="Abort the event if the Xmax posterior is still wider than "
                             "this (g/cm2) after %d VI iterations (default: %%(default)s). "
                             "0 disables the check." % _EARLY_ABORT_XMAX_AFTER_ITERS)
    parser.add_argument("--early-abort-max-fluence", type=float,
                        default=_EARLY_ABORT_MAX_FLUENCE,
                        help="Abort the event if any input fluence exceeds this or is "
                             "not finite (default: %(default)s). 0 disables the check.")

    parser.add_argument("--output-dir", default=os.getcwd(),
                        help="Directory for output files and debug plots")
    parser.add_argument("--output-nur", default=None,
                        help="Write the processed event to this .nur file")
    parser.add_argument("--export-posterior-samples", action="store_true",
                        help="Save all IFT posterior samples, trigger decisions, and summary to a .npz file")

    parser.add_argument("--atmosphere-dir",
                        default=GDAS_ATMOSPHERE_DIRECTORY,
                        help="Directory to search for pre-computed ATMOSPHERE_{event_id}.DAT files "
                             "(may be read-only). If a file is not found it is generated and written "
                             "to --gdas-cache-dir.")
    parser.add_argument("--gdas-cache-dir",
                        default=os.path.join(os.path.expanduser("~"), ".cache", "lofar_gdas"),
                        help="Writable directory for downloaded GDAS binaries and newly generated "
                             "ATMOSPHERE_*.DAT files. Defaults to ~/.cache/lofar_gdas.")

    parser.add_argument("--no-nifty", action="store_true",
                        help="Skip the NIFTy/VI reconstruction (preprocessing and debug plots only)")
    parser.add_argument("--debug-plots", action="store_true")
    parser.add_argument("--log-level", default="INFO")

    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    event = run_pipeline(args)
    if event is None:
        raise RuntimeError(f"Pipeline did not produce event {args.event_id}")
    LOGGER.info("Finished event %s with %d stations", event.get_id(), len(event.get_station_ids()))


if __name__ == "__main__":
    main()
