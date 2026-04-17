import numpy as np
import NuRadioReco.modules.io.eventReader
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.detector.generic_detector
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.neutrinoVertexReconstructor.neutrino3DVertexReconstructor
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.channelGenericNoiseAdder
from NuRadioReco.utilities import units, bandpass_filter
import NuRadioReco.framework.base_trace
import argparse
import os

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
for chennel_id in Detector.get_channel_ids(11):
    delay = Detector.get_cable_delay(11, chennel_id)
    print(f"Channel {chennel_id} has a cable delay of {delay*1e9:.2f} ns")
    Detector.add_manual_time_delay(11, chennel_id, delay, weight=-1)


parser = argparse.ArgumentParser(
    description='Run the vertex reconstruction used for the RNO-G energy reconstruction'
)
parser.add_argument(
    'lookup_tables',
    type=str,
    help='Folder containing the lookup tables for the vertex reconstructor. To create lookup tables, run '
         'NuRadioReco/modules/neutrinoVertexReconstructor/create_lookup_table.py'
         'You need a lookup table for each antenna depth used in the reconstruction.'
)
parser.add_argument('--input_file', type=str, default='simulated_final_with_normal_trace_without_noise.nur', help='File to run the reconstruction on.')
parser.add_argument('--output_file', type=str, default='reconstructed_vertex_with_normal_trace_20_events.nur', help='Filename into which results are written')
# parser.add_argument(
#     '--detector_file',
#     type=str,
#     default='../../detector/RNO_G/RNO_single_station.json',
#     help='JSON file containing the detector description. Here, we assume it is written for the GenericDetector class.'
# )
parser.add_argument('--noise_level', type=float, default=10., help='RMS of the noise to be added, in mV.')
args = parser.parse_args()
noise_level = args.noise_level * units.mV
sampling_rate = 5. * units.GHz

"""
IDs of the channels to be used for the vertex reconstruction, assuming you used the RNO-G detector description set as
the default. The shorter list saves time, but results may be less accurate.
"""
#vertex_channel_ids = [0, 1, 6, 7, 9, 21]
vertex_channel_ids = [0, 1, 2, 3, 5, 6, 7, 9, 10, 23, 22]               #5 is vpol
# vertex_channel_ids = [ 2, 3, 6, 7, 8, 21, 22]      #just for
"""
Passband of the filter that is applied to the channels for the vertex reconstruction.
"""
vertex_reco_passband = [.1, .3]

"""
Set up modules and detector class
"""
signal_reconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
event_reader = NuRadioReco.modules.io.eventReader.eventReader()
event_reader.begin([args.input_file])
channel_resampler = NuRadioReco.modules.channelResampler.channelResampler()
event_writer = NuRadioReco.modules.io.eventWriter.eventWriter()
event_writer.begin(args.output_file)
noise_adder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()          
# det = NuRadioReco.detector.generic_detector.GenericDetector(
#     json_filename=args.detector_file,
#     antenna_by_depth=False
# )

"""
We create an electric field template to be used when calculating the timing difference between channels. Pretty much any
short pulse will do, so we create a delta pulse by setting a constant spectrum, apply a filter, and shift the pulse to
be in the middle of the trace.
"""
spec = np.ones(int(128 * sampling_rate + 1)) * bandpass_filter.get_filter_response(
    np.fft.rfftfreq(int(256 * sampling_rate), 1. / sampling_rate), vertex_reco_passband, 'butter', 10)
efield_template = NuRadioReco.framework.base_trace.BaseTrace()
efield_template.set_frequency_spectrum(spec, sampling_rate)
efield_template.apply_time_shift(20. * units.ns, True)

if not os.path.isdir('plots/Vertex_reco_with_normal_trace_20'):
    os.makedirs('plots/Vertex_reco_with_normal_trace_20')
"""
Set up vertex reconstruction modules
"""
vertex_reconstructor = NuRadioReco.modules.neutrinoVertexReconstructor.neutrino3DVertexReconstructor.neutrino3DVertexReconstructor(
    lookup_table_location=args.lookup_tables
)
"""
These settings are a compromise between accuracy and saving time. Reduce grid size for better results. But be careful
when creating plots: A very fine grid can cause memory problems for matplotlib.
"""
vertex_reconstructor.begin(
    station_id=11,
    channel_ids=vertex_channel_ids,
    detector=Detector,
    template=efield_template,
    distances_2d=np.arange(10, 600, 10),
    distance_step_3d=2,
    z_step_3d=2,
    widths_3d=np.arange(-50, 50, 2),
    passband=vertex_reco_passband,
    z_coordinates_2d=np.arange(-600, -10, 10),
    debug_folder='plots/Vertex_reco_with_normal_trace_20'
)
#run={}               #list of events to run the reconstruction on. Here we just run it on the first event, but feel free to change this.

for i_event, event in enumerate(event_reader.run()):
    if i_event == 20:       #just run on the first 10 events to save time, but feel free to change this
        break
    print('Event {}, ID={}, Run={}'.format(i_event, event.get_id(), event.get_run_number()))
    station = event.get_station(11)
    station.set_is_neutrino()
    channel_resampler.run(event, station, Detector, sampling_rate=sampling_rate)
    noise_adder.run(event, station, Detector, amplitude=noise_level, type='rayleigh')
    vertex_reconstructor.run(
        event,
        station,
        Detector,
        debug=True
    )
    channel_resampler.run(event, station, Detector, sampling_rate=2. * units.GHz)
    signal_reconstructor.run(event, station, Detector)
    event_writer.run(event)
