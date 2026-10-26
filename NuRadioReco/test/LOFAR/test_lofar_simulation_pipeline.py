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

from NuRadioReco.pipeline.LOFAR import simulation_pipeline

core_spread = 10.0 # Set to a smaller value for testing; adjust as needed

def validate_reconstruction(test_event, true_event):
    """Compare the reconstruction results of the test event with the pre-saved true event."""
    # Compare the reconstructed parameters
    test_shower = test_event.get_first_shower()
    true_shower = true_event.get_first_shower()
    
    assert np.isclose(np.log10(test_shower.get_parameter(showerParameters.energy)), np.log10(true_shower.get_parameter(showerParameters.energy)), rtol=0.1)
    assert np.isclose(test_shower.get_parameter(showerParameters.zenith), true_shower.get_parameter(showerParameters.zenith), rtol=0.1)
    assert np.isclose(test_shower.get_parameter(showerParameters.azimuth), true_shower.get_parameter(showerParameters.azimuth), rtol=0.1)
    assert np.isclose(test_shower.get_parameter(showerParameters.shower_maximum), true_shower.get_parameter(showerParameters.shower_maximum), rtol=0.1)
    # absolute tolerance for the core is set such that it should be within 1 std of the randomised core spread (10 m) used in the simulation
    assert np.isclose(test_shower.get_parameter(showerParameters.core)[0], true_shower.get_parameter(showerParameters.core)[0], atol=core_spread) 
    assert np.isclose(test_shower.get_parameter(showerParameters.core)[1], true_shower.get_parameter(showerParameters.core)[1], atol=core_spread)
    assert np.isclose(test_shower.get_parameter(showerParameters.core)[2], true_shower.get_parameter(showerParameters.core)[2], rtol=0.1)
    
def validate_reco_vs_mc_truth(test_event):
    """Compare the reconstruction results of the test event with the MC truth parameters."""
    # Compare the reconstructed parameters with MC truth
    mc_shower = test_event.get_first_sim_shower()
    test_shower = test_event.get_first_shower()
    assert np.isclose(np.log10(test_shower.get_parameter(showerParameters.energy)), np.log10(mc_shower.get_parameter(showerParameters.energy)), rtol=0.1)
    assert np.isclose(test_shower.get_parameter(showerParameters.zenith), mc_shower.get_parameter(showerParameters.zenith), rtol=0.1)
    assert np.isclose(test_shower.get_parameter(showerParameters.azimuth), mc_shower.get_parameter(showerParameters.azimuth), rtol=0.1)
    assert np.isclose(test_shower.get_parameter(showerParameters.shower_maximum), mc_shower.get_parameter(showerParameters.shower_maximum), rtol=0.1)
    # absolute tolerance for the core is set such that it should be within 1 std of the randomised core spread (10 m) used in the simulation
    assert np.isclose(test_shower.get_parameter(showerParameters.core)[0], mc_shower.get_parameter(showerParameters.core)[0], atol=core_spread)
    assert np.isclose(test_shower.get_parameter(showerParameters.core)[1], mc_shower.get_parameter(showerParameters.core)[1], atol=core_spread)
    assert np.isclose(test_shower.get_parameter(showerParameters.core)[2], mc_shower.get_parameter(showerParameters.core)[2], rtol=0.1)
    print("Reconstruction validation passed: test event matches MC truth.")

# Use the public sample by default. For your own event, edit these paths and
# event_id below
sim_data_path = Path("./test-data/hdf5_sims")
output_path = Path("./test-data/output-sim-pipeline")
download_from_dataserver(
    remote_path="lofar_share/lofar-sample-event-sim.tar.gz",
    target_path=str(sim_data_path / "sample-sim.tar.gz"),
)
event_id = 92380604
mass = "proton"
coreas_sim_id = "000082"
coreas_hdf5_file = f"SIM{coreas_sim_id}.hdf5"  # Replace with the actual HDF5 file name for the simulated event

pipeline_path = Path("./test-data/pipeline")
atmosphere_dir = Path("./test-data/pipeline/atmosphere")
noise_library_dir = pipeline_path / "noise_library"
download_from_dataserver(
    remote_path=f"lofar_share/pipeline-results-{event_id}.tar.gz",
    target_path=str(pipeline_path / f"pipeline-results-{event_id}.tar.gz"),
)

# Start from the actual CLI defaults
# parse_args([]) would require event_id; do not parse the notebook kernel's argv.
pipeline_args = simulation_pipeline.build_arg_parser().parse_args([str(event_id), mass, coreas_hdf5_file])
pipeline_args.coreas_dir = str(sim_data_path)
pipeline_args.noise_library = str(noise_library_dir / "lofar_real_noise_library.npy")
pipeline_args.noise_library_nur = str(noise_library_dir / "lofar_real_noise_library.nur")
pipeline_args.output_dir = str(output_path)
pipeline_args.output_nur = None # do NOT save for the test, as we want to compare the results with the pre-saved ones
pipeline_args.core_spread = core_spread # see above
pipeline_args.atmosphere_dir = atmosphere_dir
pipeline_args.gdas_cache_dir = str(output_path / "gdas_cache")
pipeline_args.ift_iterations = 2        # increase for more refined reconstruction
pipeline_args.ift_samples = 40          # increase for more refined reconstruction
pipeline_args.enable_fluence_correlated_field = False       # disabled for demonstration, setting to True might increase runtime and memory usage 
pipeline_args.enable_timing_correlated_field = False        # disabled for demonstration, setting to True might increase runtime and memory usage 
pipeline_args.export_posterior_samples = False       # do not produce posterior samples for the validation run, set to True to produce posterior samples
pipeline_args.dry_run = False           # Set to True to skip reconstruction step

# run the pipeline
test_event = simulation_pipeline.run_pipeline(pipeline_args)

# validate with the pre-saved results from the pipeline run on the same event
pregen_event_path = pipeline_path / f"simulation-results-{event_id}-{mass}-{coreas_sim_id}.nur"
full_reader = NuRadioReco.modules.io.eventReader.eventReader()
full_reader.begin(str(pregen_event_path))

true_event = next(full_reader.run())

validate_reconstruction(test_event, true_event)

validate_reco_vs_mc_truth(test_event)
