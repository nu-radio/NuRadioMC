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

parser = argparse.ArgumentParser(
    'Run the IFT electric field reconstruction.'
)
parser.add_argument(
    '--input_file',
    type=str,
    default='reconstructed_vertex.nur',
    help='Name of the input file. Should be output file of T02_run_vertex_reco.py'
)
parser.add_argument('--output_file', type=str, default='reconstructed_efield.nur', help='Filename into which results are written')
parser.add_argument(
    '--detector_file',
    type=str,
    default='../../detector/RNO_G/RNO_single_station.json',
    help='JSON file containing the detector description. Here, we assume it is written for the GenericDetector class.'

)
parser.add_argument('--noise_level', type=float, default=10., help='RMS of the noise in the channel traces, in mV.')
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
det = NuRadioReco.detector.generic_detector.GenericDetector(
    json_filename=args.detector_file,
    antenna_by_depth=False
)
channel_bandpass_filter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
efield_bandpass_filter = NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()
channel_pulse_finder = NuRadioReco.modules.channelPulseFinderSimulator.channelPulseFinderSimulator()
channel_pulse_finder.begin(noise_level=noise_level)
"""
We create an electric field template to be used to find the radio pulse. Pretty much any
short pulse will do, so we create a delta pulse by setting a constant spectrum, apply a filter, and shift the pulse to
be in the middle of the trace.
"""
spec = np.ones(int(128 * sampling_rate + 1)) * bandpass_filter.get_filter_response(np.fft.rfftfreq(int(256 * sampling_rate), 1. / sampling_rate),    [.1, .3], 'butter', 10)
efield_template = NuRadioReco.framework.base_trace.BaseTrace()
efield_template.set_frequency_spectrum(spec, sampling_rate)
efield_template.apply_time_shift(20. * units.ns, True)

if not os.path.isdir('plots/efield_reco'):
    os.makedirs('plots/efield_reco')
ift_efield_reconstructor = NuRadioReco.modules.iftElectricFieldReconstructor.iftElectricFieldReconstructor.IftElectricFieldReconstructor()
ift_efield_reconstructor.begin(
    electric_field_template=efield_template,

    passband=efield_reco_passband,
    n_samples=10,
    n_iterations=5,
    phase_slope='negative',
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
    plot_folder='plots/efield_reco'
)
time_offset_calculator = NuRadioReco.modules.channelTimeOffsetCalculator.channelTimeOffsetCalculator()
time_offset_calculator.begin(
    electric_field_template=efield_template,
    medium=ice
)
channel_props_from_neighbor = NuRadioReco.modules.channelSignalPropertiesFromNeighbors.channelSignalPropertiesFromNeighbors()
for i_event, event in enumerate(event_reader.get_events()):
    print('Event {}, Run={}, ID={}'.format(i_event, event.get_run_number(), event.get_id()))
    station = event.get_station(11)
    sim_station = station.get_sim_station()
    channel_resampler.run(event, station, det, sampling_rate=sampling_rate)
    channel_pulse_finder.run(event, station, det)
    channel_bandpass_filter.run(event, station, det, passband=efield_reco_passband, filter_type='butter', order=10)
    channel_bandpass_filter.run(event, sim_station, det, passband=efield_reco_passband, filter_type='butter', order=10)
    efield_bandpass_filter.run(event, sim_station, det, passband=efield_reco_passband, filter_type='butter', order=10)
    time_offset_calculator.run(event, station, det, range(6), passband=vertex_reco_passband)
    channel_props_from_neighbor.run(event, station, det, channel_groups=[[0, 1, 2, 3, 4, 5]])
    # channel_props_from_neighbor.run(event, station, det, channel_groups=[[9, 10, 11]])
    # channel_props_from_neighbor.run(event, station, det, channel_groups=[[21, 22, 23]])
    for ray_type in range(3):
        ift_efield_reconstructor.run(
            event,
            station,
            det,
            channel_ids=[0, 1, 2, 3, 4, 5],
            efield_scaling=True,
            ray_type=ray_type + 1,
            plot_title='',
            polarization='pol'
        )
    channel_resampler.run(event, station, det, sampling_rate=2.)
    efield_resampler.run(event, station, det, sampling_rate=2.)
    efield_resampler.run(event, station.get_sim_station(), det, sampling_rate=2.)
    event_writer.run(event)
