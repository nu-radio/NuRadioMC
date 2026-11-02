import socket

try:
    import jax

    # Double precision has to be enabled before any other module creates a JAX array,
    # so the imports below deliberately do not sit at the top of the file (PEP 8 E402).
    jax.config.update("jax_enable_x64", True)
except ImportError:
    jax = None

from astropy.time import Time
from pathlib import Path

import numpy as np  # noqa: E402
import radiotools
import subprocess

import NuRadioReco.modules.io.eventReader
from NuRadioReco.utilities.dataservers import download_from_dataserver
from NuRadioReco.utilities.LOFAR.iftDataHelpers import MAX_SIGNAL_SNR_THRESHOLD  # noqa: E402
from NuRadioReco.framework.parameters import stationParameters, channelParameters, showerParameters

from NuRadioReco.pipeline.LOFAR import data_pipeline

def validate_reconstruction(test_event, true_event):
    """Compare the reconstruction results of the test event with the pre-saved true event."""
    # Compare the reconstructed parameters
    test_shower = test_event.get_first_shower()
    true_shower = true_event.get_first_shower()
    
    assert np.isclose(np.log10(test_shower.get_parameter(showerParameters.energy)), np.log10(true_shower.get_parameter(showerParameters.energy)), rtol=0.1)
    assert np.isclose(test_shower.get_parameter(showerParameters.zenith), true_shower.get_parameter(showerParameters.zenith), rtol=0.1)
    assert np.isclose(test_shower.get_parameter(showerParameters.azimuth), true_shower.get_parameter(showerParameters.azimuth), rtol=0.1)
    assert np.isclose(test_shower.get_parameter(showerParameters.shower_maximum), true_shower.get_parameter(showerParameters.shower_maximum), rtol=0.1)
    assert np.isclose(test_shower.get_parameter(showerParameters.core)[0], true_shower.get_parameter(showerParameters.core)[0], rtol=0.1)
    assert np.isclose(test_shower.get_parameter(showerParameters.core)[1], true_shower.get_parameter(showerParameters.core)[1], rtol=0.1)
    assert np.isclose(test_shower.get_parameter(showerParameters.core)[2], true_shower.get_parameter(showerParameters.core)[2], rtol=0.1)
    print("Reconstruction validation passed: test event matches pre-saved true event.")

# Use the public sample by default. For your own event, edit these paths and
# event_id below
data_path = Path("./test-data/data")
output_path = Path("./test-data/output-data-pipeline")
tbb_directory = data_path / "tbb"
json_directory = data_path / "lora/json"
metadata_directory = data_path / "metadata"
block_number_file = None  # sample has LORA JSON; optionally supply a LORAtime4 path
download_from_dataserver(
    remote_path="lofar_share/lofar-sample-event.tar.gz",
    target_path=str(data_path / "sample-data.tar.gz"),
)
event_id = 92380604

pipeline_path = Path("./test-data/pipeline")
atmosphere_dir = Path("./test-data/pipeline/atmosphere")
download_from_dataserver(
    remote_path=f"lofar_share/pipeline-results-{event_id}.tar.gz",
    target_path=str(pipeline_path / f"pipeline-results-{event_id}.tar.gz"),
)

# Start from the actual CLI defaults
# parse_args([]) would require event_id; do not parse the notebook kernel's argv.
pipeline_args = data_pipeline.build_arg_parser().parse_args([str(event_id)])
pipeline_args.tbb_dir = str(tbb_directory)
pipeline_args.json_dir = str(json_directory)
pipeline_args.metadata_dir = str(metadata_directory)
pipeline_args.block_number_file = block_number_file
pipeline_args.output_dir = str(output_path)
pipeline_args.output_nur = None # do NOT save for the test, as we want to compare the results with the pre-saved ones
pipeline_args.atmosphere_dir = atmosphere_dir
pipeline_args.gdas_cache_dir = str(output_path / "gdas_cache")
pipeline_args.ift_iterations = 2        # increase for more refined reconstruction
pipeline_args.ift_samples = 40          # increase for more refined reconstruction
pipeline_args.enable_fluence_correlated_field = False       # disabled for demonstration, setting to True might increase runtime and memory usage 
pipeline_args.enable_timing_correlated_field = False        # disabled for demonstration, setting to True might increase runtime and memory usage 
pipeline_args.export_posterior_samples = False       # do not produce posterior samples for the validation run, set to True to produce posterior samples
pipeline_args.dry_run = False           # Set to True to skip reconstruction step

# run the pipeline
test_event = data_pipeline.run_pipeline(pipeline_args)

# validate with the pre-saved results from the pipeline run on the same event
pregen_event_path = pipeline_path / f"data-results-{event_id}.nur"
full_reader = NuRadioReco.modules.io.eventReader.eventReader()
full_reader.begin(str(pregen_event_path))

true_event = next(full_reader.run())

validate_reconstruction(test_event, true_event)
