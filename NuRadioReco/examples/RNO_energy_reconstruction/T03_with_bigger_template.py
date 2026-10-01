import numpy as np
import NuRadioReco.modules.io.NuRadioRecoio
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.detector.generic_detector
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.electricFieldResampler
import NuRadioReco.modules.channelTimeOffsetCalculator
import NuRadioReco.modules.channelSignalPropertiesFromNeighbors
import NuRadioReco.modules.iftElectricFieldReconstructor.iftElectricFieldReconstructor
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.electricFieldBandPassFilter
import NuRadioReco.modules.channelPulseFinderSimulator
from NuRadioReco.utilities import units, bandpass_filter
import NuRadioMC.utilities.medium
import NuRadioReco.framework.base_trace
import argparse
import os

from NuRadioReco.detector.RNO_G import rnog_detector_mod as RNO_G_detector
from astropy.time import Time

Detector = RNO_G_detector.ModDetector(
    select_stations=11,
    always_query_entire_description=True,
    signal_chain_measurement_name="calibrated_impulse_response_v0"
)
Detector.update(time=Time("2023-08-01"))

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
    'Run the IFT electric field reconstruction.'
)
parser.add_argument(
    '--input_file',
    type=str,
    default='reconstructed_vertex_with_normal_trace_10_events.nur',
    help='Name of the input file. Should be output file of T02_run_vertex_reco.py'
)
parser.add_argument('--output_file', type=str, default='reconstructed_efield_normal_trace_same_size.nur', help='Filename into which results are written')
# parser.add_argument(
#     '--detector_file',
#     type=str,
#     default='../../detector/RNO_G/RNO_single_station.json',
#     help='JSON file containing the detector description. Here, we assume it is written for the GenericDetector class.'

# )
parser.add_argument('--noise_level', type=float, default=10.0, help='RMS of the noise in the channel traces, in mV.')
args = parser.parse_args()

noise_level = args.noise_level * units.mV
sampling_rate = 5. * units.GHz
vertex_reco_passband = [.1, .3]
efield_reco_passband = [.13, .5]
ice = NuRadioMC.utilities.medium.get_ice_model('greenland_simple')
event_reader = NuRadioReco.modules.io.NuRadioRecoio.NuRadioRecoio([args.input_file])
channel_resampler = NuRadioReco.modules.channelResampler.channelResampler()
efield_resampler = NuRadioReco.modules.electricFieldResampler.electricFieldResampler()
event_writer = NuRadioReco.modules.io.eventWriter.eventWriter()
event_writer.begin(args.output_file)
det=Detector
# det = NuRadioReco.detector.generic_detector.GenericDetector(
#     json_filename=args.detector_file,
#     antenna_by_depth=False
# )
channel_bandpass_filter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
efield_bandpass_filter = NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()
channel_pulse_finder = NuRadioReco.modules.channelPulseFinderSimulator.channelPulseFinderSimulator()
channel_pulse_finder.begin(
    noise_level=noise_level,
    min_snr=2.5
)
"""
We create an electric field template to be used to find the radio pulse. Pretty much any
short pulse will do, so we create a delta pulse by setting a constant spectrum, apply a filter, and shift the pulse to
be in the middle of the trace.
"""

spec = np.ones(int(160*sampling_rate+1)) * bandpass_filter.get_filter_response(np.fft.rfftfreq(int(320*sampling_rate), 1. / sampling_rate),    [.13, .5] , 'butter', 10) #changed the filtering
print("printing the length of the spec")
print(len(spec))

efield_template = NuRadioReco.framework.base_trace.BaseTrace()
efield_template.set_frequency_spectrum(spec, sampling_rate)

efield_template.apply_time_shift(20. * units.ns, True)

if not os.path.isdir('plots/e_reco_with_same_size_theta_lag'):
    os.makedirs('plots/e_reco_with_same_size_theta_lag')
ift_efield_reconstructor = NuRadioReco.modules.iftElectricFieldReconstructor.iftElectricFieldReconstructor.IftElectricFieldReconstructor()
ift_efield_reconstructor.begin(
    electric_field_template=efield_template,
    passband=efield_reco_passband,
    n_samples=10,
    n_iterations=5,
    phase_slope='both',
    energy_fluence_passbands=[
        [.13, .2],
        [.13, .25],
        [.13, .3]
    ],
    slope_passbands=[
        [[.13, .2], [.2, .3]],
        [[.13, .25], [.25, .5]],
        [[.13, .3], [.3, .5]],
    ],
    debug=True,
    plot_folder='plots/e_reco_with_same_size_theta_lag'
)
# ift_efield_reconstructor.make_priors_plot()
time_offset_calculator = NuRadioReco.modules.channelTimeOffsetCalculator.channelTimeOffsetCalculator()       #this calculates the time offset between channels using the simulated electric field as a template, and applies it to the data channels. This is not something you would do in a real analysis, but it allows us to check that the reconstruction works when the timing between channels is perfect.
time_offset_calculator.begin(
    electric_field_template=efield_template,
    medium=ice,
    use_sim=True
)
channel_props_from_neighbor = NuRadioReco.modules.channelSignalPropertiesFromNeighbors.channelSignalPropertiesFromNeighbors() # this module uses the timing information from the channels for which we calculated the time offset to calculate the timing of the other channels, which do not have a direct time offset calculation because they do not have a clear pulse. Again, this is not something you would do in a real analysis, but it allows us to check that the reconstruction works when the timing between channels is perfect.
for i_event, event in enumerate(event_reader.get_events()):
    print('Event {}, Run={}, ID={}'.format(i_event, event.get_run_number(), event.get_id()))
    station = event.get_station(11)
    sim_station = station.get_sim_station()
    channel_resampler.run(event, station, det, sampling_rate=sampling_rate)
    channel_pulse_finder.run(event, station, det)
    channel_bandpass_filter.run(event, station, det, passband=efield_reco_passband, filter_type='butter', order=10)
    channel_bandpass_filter.run(event, sim_station, det, passband=efield_reco_passband, filter_type='butter', order=10)
    efield_bandpass_filter.run(event, sim_station, det, passband=efield_reco_passband, filter_type='butter', order=10)
    time_offset_calculator.run(event, station, det, [0, 1, 2, 3], passband=vertex_reco_passband)
    channel_props_from_neighbor.run(event, station, det, channel_groups=[[0, 1, 2, 3]])
    # channel_props_from_neighbor.run(event, station, det, channel_groups=[[9, 10, 11]])
    # channel_props_from_neighbor.run(event, station, det, channel_groups=[[21, 22, 23]])
    for ray_type in range(1):
        # ift_efield_reconstructor.make_priors_plot()
        ift_efield_reconstructor.run(
            event,
            station,
            det,
            channel_ids=[0, 1, 2, 3],
            efield_scaling=True,
            ray_type=ray_type+1,
            plot_title='',
            polarization='pol'
        )
        
    channel_resampler.run(event, station, det, sampling_rate=2.)
    efield_resampler.run(event, station, det, sampling_rate=2.)
    efield_resampler.run(event, station.get_sim_station(), det, sampling_rate=2.)
    event_writer.run(event)
