#!/usr/bin/env python3
"""
NuRadioReco-native LOFAR preprocessing and IFT pipeline.

.. moduleauthor:: Karen Terveer <karen.terveer@fau.de>
"""

import argparse
import logging
import os

import jax

# Double precision has to be enabled before any other module creates a JAX array,
# so the imports below deliberately do not sit at the top of the file (PEP 8 E402).
jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

import NuRadioReco.detector.detector  # noqa: E402
import NuRadioReco.modules.eventTypeIdentifier  # noqa: E402
import NuRadioReco.modules.io.eventWriter  # noqa: E402
from NuRadioReco.framework.parameters import stationParameters, showerParameters  # noqa: E402
from NuRadioReco.modules import channelBandPassFilter  # noqa: E402
from NuRadioReco.modules import voltageToEfieldConverter  # noqa: E402
from NuRadioReco.modules.LOFAR import planeWaveDirectionFitter_LOFAR  # noqa: E402
from NuRadioReco.modules.LOFAR import pipelineVisualizer_LOFAR  # noqa: E402
from NuRadioReco.modules.LOFAR import stationGalacticCalibrator  # noqa: E402
from NuRadioReco.modules.LOFAR import stationPulseFinder  # noqa: E402
from NuRadioReco.modules.LOFAR import stationRFIFilter  # noqa: E402
from NuRadioReco.modules.LOFAR import iftReconstructor  # noqa: E402
from NuRadioReco.modules.LOFAR.iftReconstructor import (  # noqa: E402
    _DEFAULT_N_VI_ITERATIONS, _DEFAULT_N_SAMPLES,
    _EARLY_ABORT_XMAX_STD_GCM2, _EARLY_ABORT_XMAX_AFTER_ITERS, _EARLY_ABORT_MAX_FLUENCE,
)
from NuRadioReco.modules.LOFAR.utilities.iftDataHelpers import MAX_SIGNAL_SNR_THRESHOLD  # noqa: E402
from NuRadioReco.modules.io.LOFAR import readLOFARData  # noqa: E402
from NuRadioReco.utilities import units  # noqa: E402


LOGGER = logging.getLogger("NuRadioReco.LOFAR.pipeline")


DEFAULT_STATIONS = [
    "CS001", "CS002", "CS003", "CS004", "CS005", "CS006", "CS007",
    "CS011", "CS013", "CS017",
]


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


def _save_trace_snapshot(event, output_dir, stage):
    import matplotlib.pyplot as plt

    max_channels = 8
    os.makedirs(output_dir, exist_ok=True)
    for station in event.get_stations():
        channels = list(station.iter_channels())[:max_channels]
        if not channels:
            continue

        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        ytick_positions = []
        ytick_labels = []
        for i_ch, channel in enumerate(channels):
            trace = channel.get_trace()
            if len(trace) == 0:
                continue
            stride = max(1, len(trace) // 4096)
            samples = np.arange(0, len(trace), stride)
            scale = np.nanmax(np.abs(trace)) or 1.0
            ax.plot(samples, trace[::stride] / scale + i_ch, lw=0.7, alpha=0.8)
            ytick_positions.append(i_ch)
            ytick_labels.append(str(channel.get_id()))

        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Channel ID")
        ax.set_title(f"{stage}: CS{station.get_id():03d} first {len(channels)} channels")
        fig.savefig(
            os.path.join(output_dir, f"{stage}_traces_CS{station.get_id():03d}_{event.get_id()}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


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
    detector = NuRadioReco.detector.detector.Detector(
        args.detector,
        source="json",
        antenna_by_depth=False,
    )

    reader = readLOFARData.readLOFARData(
        restricted_station_set=DEFAULT_STATIONS,
        tbb_directory=args.tbb_directory,
        json_directory=args.json_directory,
        metadata_directory=args.metadata_directory,
    )
    reader.begin(args.event_id, block_number_file=args.block_number_file)

    event_type_identifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()
    rfi_filter = stationRFIFilter.stationRFIFilter()
    bandpass_filter = channelBandPassFilter.channelBandPassFilter()
    calibrator = stationGalacticCalibrator.stationGalacticCalibrator()
    pulse_finder = stationPulseFinder.stationPulseFinder()
    direction_fitter = planeWaveDirectionFitter_LOFAR.planeWaveDirectionFitter()
    reconstructor = iftReconstructor.iftReconstructor()

    debug_dir = os.path.join(args.output_dir, "debug_plots", str(args.event_id)) if args.debug_plots else None

    processed_event = None
    for event in reader.run(detector, trace_length=65536):
        if args.debug_plots:
            _save_trace_snapshot(event, debug_dir, "01_reader")

        event_type_identifier.begin()
        for station in event.get_stations():
            event_type_identifier.run(event, station, mode="forced", forced_event_type="cosmic_ray")

        LOGGER.info("Running LOFAR RFI filter")
        rfi_filter.begin(
            reader=reader,
            rfi_cleaning_trace_length=8192,
            debug_plot_dir=debug_dir,
        )
        rfi_filter.run(event)
        rfi_filter.end(event if args.debug_plots else None)
        if args.debug_plots:
            _save_trace_snapshot(event, debug_dir, "02_rfi")

        LOGGER.info("Running bandpass filter")
        for station in event.get_stations():
            bandpass_filter.run(
                event, station, detector,
                passband=[30.0 * units.MHz, 80.0 * units.MHz],
                filter_type="gaussian_tapered",
                roll_width=2.5 * units.MHz,
            )
            bandpass_filter.run(
                event, station, detector,
                passband=[30.0 * units.MHz, 80.0 * units.MHz],
                filter_type="hann_tapered",
                half_hann_percent=0.1,
            )
        if args.debug_plots:
            _save_trace_snapshot(event, debug_dir, "03_bandpass")

        LOGGER.info("Running galactic calibration")
        calibrator.begin()
        calibrator.run(event, detector)
        calibrator.end()
        if args.debug_plots:
            _save_trace_snapshot(event, debug_dir, "04_galactic_calibration")

        LOGGER.info("Running pulse finder")
        pulse_finder.begin(cr_snr=6.5, good_channels=6)
        pulse_finder.run(event, detector)
        pulse_finder.end()

        LOGGER.info("Running plane-wave direction fitter")
        direction_fitter.begin(
            debug=args.debug_plots,
            debug_plot_dir=debug_dir,
        )
        direction_fitter.run(event, detector)
        direction_fitter.end()

        LOGGER.info("Running voltage-to-electric-field conversion")
        _convert_voltage_to_efield(event, detector)
        if args.debug_plots:
            _save_pipeline_visualizer_plots(event, detector, debug_dir)

        LOGGER.info("Running IFT reconstruction module")
        recon_kwargs = dict(
            enable_fluence_correlated_field=args.enable_fluence_correlated_field,
            enable_timing_correlated_field=args.enable_timing_correlated_field,
            export_posterior_samples=args.export_posterior_samples,
            output_directory=args.output_dir,
            debug_plots=args.debug_plots,
            debug_plot_dir=debug_dir,
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
        reconstructor.run(event, detector)
        reconstructor.end()

        if args.output_nur:
            writer = NuRadioReco.modules.io.eventWriter.eventWriter()
            writer.begin(args.output_nur)
            writer.run(event)
            writer.end()

        os.makedirs(args.output_dir, exist_ok=True)
        _write_params_nur(event, args.output_dir)

        processed_event = event
        break

    return processed_event


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("event_id", type=int)

    parser.add_argument("--tbb-directory",
                        default="/vol/astro5/lofar/astro3/vhecr/lora_triggered/data/")
    parser.add_argument("--json-directory",
                        default="/vol/astro7/lofar/kratos_files/json")
    parser.add_argument("--metadata-directory",
                        default="/vol/astro7/lofar/vhecr/kratos/data/")
    parser.add_argument("--detector",
                        default="LOFAR/LOFAR.json")
    parser.add_argument("--block-number-file",
                        default="/vol/astro5/lofar/astro3/vhecr/lora_triggered/LORA/LORAtime4")

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
                        default="/vol/astro7/lofar/sim/pipeline/atmosphere_files",
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
