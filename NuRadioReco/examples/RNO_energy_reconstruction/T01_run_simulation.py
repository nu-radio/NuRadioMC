from __future__ import absolute_import, division, print_function
import argparse
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("runstrawman")
# from NuRadioReco.detector.RNO_G import rnog_detector as RNO_G_detector
from NuRadioReco.detector.RNO_G import rnog_detector_mod as RNO_G_detector
from astropy.time import Time

# Detector= RNO_G_detector.Detector(select_stations=11)
# # Set detector time to now
# Detector.update(time=Time.now())


Detector = RNO_G_detector.ModDetector(
    select_stations=11,
    always_query_entire_description=True,
    signal_chain_measurement_name="calibrated_impulse_response_v0"
)
Detector.update(time=Time("2023-08-01"))
# Detector.modify_station_description(11, ["signal_digitizer_config", "number_of_samples"], 2048 * 4)

for chennel_id in Detector.get_channel_ids(11):
    delay = Detector.get_cable_delay(11, chennel_id)
    print(f"Channel {chennel_id} has a cable delay of {delay*1e9:.2f} ns")
    Detector.add_manual_time_delay(11, chennel_id, delay, weight=-1)


station_id = 11
print("Original station position:", Detector.get_absolute_position(station_id))
Detector.modify_station_description(
    station_id=station_id,
    keys=["station_position", "position"],   # ✅ correct nested path
    value=[0.0, 0.0, 0.0]
)
print("Modified station position:", Detector.get_absolute_position(station_id))



parser = argparse.ArgumentParser(
    description='We start by creating some data to do reconstruction on, but feel free to use your own simulations!'
)
parser.add_argument(
    'input_file',
    type=str,
    help='Path to the HDF5 file containing generated neutrino events to be simulated.'
)
parser.add_argument(
    '--output_file',
    type=str,
    default='simulated_final_with_normal_trace_without_noise.nur',
    help='Name of the .nur file the simulated events will be written into.'
)
# parser.add_argument(
#     '--detector_file',
#     type=str,
#     default='../../detector/RNO_G/RNO_single_station.json',
#     help='Path to the JSON file containing the detector description.'
# )
parser.add_argument(
    '--config_file',
    type=str,
    default='config.yaml',
    help='Path to the .yaml file containing the simulation configuration.'
)
parser.add_argument(
    '--noise_level',
    type=float,
    default=10.,
    help='Root mean square (in millivolt) of the noise to be simulated. Note that this noise should include the amplifier'
         ' response.'
)

args = parser.parse_args()

# initialize detector sim modules
highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
hardware_response = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
noise_level = args.noise_level * units.mV


class mySimulation(simulation.simulation):

    def _detector_simulation_filter_amp(self, evt, station, det):
        hardware_response.run(evt, station, det, sim_to_data=True)            #check whats sim_to_data does

    def _detector_simulation_trigger(self, evt, station, det):
        highLowThreshold.run(evt, station, det,
                                    threshold_high=2. * noise_level,
                                    threshold_low=-2. * noise_level,
                                    triggered_channels=[0, 1, 2, 3],
                                    number_concidences=2,  # 2/4 majority logic
                                    trigger_name='main_trigger',
                                    pre_trigger_time=100 * units.ns
                             )
        trigger = station.get_primary_trigger()
        if trigger is not None and trigger.has_triggered():
            print(f"Trigger time: {trigger.get_trigger_time()}")
            print(f"Pre-trigger time ch0: {trigger.get_pre_trigger_time_channel(0)}")
            print(f"Pre-trigger time ch5: {trigger.get_pre_trigger_time_channel(5)}")


sim = mySimulation(
    inputfilename=args.input_file,
    outputfilename='T01_output.hdf5',
     # detectorfile=args.detector_file,
    det=Detector,
    outputfilenameNuRadioReco=args.output_file,
    config_file=args.config_file,
    file_overwrite=True,
    write_detector=True,
    evt_time=Time("2023-08-01"),
    debug=True
)
sim.run()
