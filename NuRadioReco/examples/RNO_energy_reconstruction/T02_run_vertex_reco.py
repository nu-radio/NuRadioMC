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
parser.add_argument('--input_file', type=str, default='simulated_events.nur', help='File to run the reconstruction on.')
parser.add_argument('--output_file', type=str, default='reconstructed_vertex.nur', help='Filename into which results are written')
parser.add_argument(
    '--detector_file',
    type=str,
    default='../../detector/RNO_G/RNO_single_station.json',
    help='JSON file containing the detector description. Here, we assume it is written for the GenericDetector class.'
)
parser.add_argument('--noise_level', type=float, default=10., help='RMS of the noise to be added, in mV.')
args = parser.parse_args()
noise_level = args.noise_level * units.mV
sampling_rate = 5. * units.GHz

"""
IDs of the channels to be used for the vertex reconstruction, assuming you used the RNO-G detector description set as
the default. The shorter list saves time, but results may be less accurate.
"""
# vertex_channel_ids = [0, 1, 6, 7, 8, 9, 21]
vertex_channel_ids = [0, 1, 2, 3, 6, 7, 8, 9, 10, 21, 22]
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
det = NuRadioReco.detector.generic_detector.GenericDetector(
    json_filename=args.detector_file,
    antenna_by_depth=False
)

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

if not os.path.isdir('plots/vertex_reco'):
    os.makedirs('plots/vertex_reco')
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
    detector=det,
    template=efield_template,
    distances_2d=np.arange(100, 3600, 200),
    distance_step_3d=10,
    z_step_3d=10,
    widths_3d=np.arange(-50, 50, 10),
    passband=vertex_reco_passband,
    z_coordinates_2d=np.arange(-3000, -100, 25),
    debug_folder='plots/vertex_reco'
)

for i_event, event in enumerate(event_reader.run()):
    print('Event {}, ID={}, Run={}'.format(i_event, event.get_id(), event.get_run_number()))
    station = event.get_station(11)
    station.set_is_neutrino()
    noise_adder.run(event, station, det, amplitude=noise_level, type='rayleigh')
    channel_resampler.run(event, station, det, sampling_rate=sampling_rate)
    vertex_reconstructor.run(
        event,
        station,
        det,
        debug=True
    )
    channel_resampler.run(event, station, det, sampling_rate=2.)
    signal_reconstructor.run(event, station, det)
    event_writer.run(event)
