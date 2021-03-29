import os
import datetime
import astropy
import argparse
import numpy as np
from NuRadioReco.utilities import units, bandpass_filter, geometryUtilities
from NuRadioReco.detector import detector
from NuRadioReco.modules.base import module
import NuRadioReco.modules.io.coreas.readCoREAS
import NuRadioReco.modules.io.coreas.simulationSelector
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.eventTypeIdentifier
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.electricFieldResampler
import NuRadioReco.modules.electricFieldBandPassFilter
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.iftElectricFieldReconstructor.iftElectricFieldReconstructor
import NuRadioReco.framework.base_trace
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import stationParameters as stnp
import logging
logger = module.setup_logger(name='NuRadioReco', level=logging.WARNING)


parser = argparse.ArgumentParser(
    'Example script for an electric field reconstruction'
    'using Information Field Theory'
)
parser.add_argument(
    '--coreas_file',
    type=str,
    default='example_data/example_event.h5',
    help='Path to the CoREAS file'
)
parser.add_argument(
    '--detector_description',
    type=str,
    default='example_data/arianna_station_32.json',
    help='Path to the JSON file with the detector description'
)
parser.add_argument(
    '--station_id',
    type=int,
    default=32,
    help='ID if the station for which to run the reconstruction'
)
parser.add_argument(
    '--noise_level',
    type=float,
    default=100. * units.mV,
    help='RMS of the noise'
)
args = parser.parse_args()

det = detector.Detector(json_filename=args.detector_description)    # detector file
det.update(datetime.datetime(2018, 10, 1))

dir_path = os.path.dirname(os.path.realpath(__file__))  # get the directory of this file

# initialize all modules that are needed for processing
# provide input parameters that are to remain constant during processung
readCoREAS = NuRadioReco.modules.io.coreas.readCoREAS.readCoREAS()
readCoREAS.begin([args.coreas_file], args.station_id, n_cores=3, max_distance=None, seed=123)
simulationSelector = NuRadioReco.modules.io.coreas.simulationSelector.simulationSelector()
simulationSelector.begin()
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False)
efieldBandPassFilter = NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()
hardwareResponseIncorporator = NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
triggerSimulator = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
triggerSimulator.begin()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()
eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()

electricFieldResampler = NuRadioReco.modules.electricFieldResampler.electricFieldResampler()
electricFieldResampler.begin()

channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelResampler.begin()

eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
output_filename = "IFT_example_station_{}.nur".format(args.station_id)
eventWriter.begin(output_filename)

passband = [80. * units.MHz, 500. * units.MHz]
sampling_rate = 5. * units.GHz

# Create template for the IFT reconstructor
# It is only used to determine the pulse timing, so
# we can keep it simple and just use a
# bandpass-filtered delta pulse
spec = np.ones(int(128 * sampling_rate + 1)) * bandpass_filter.get_filter_response(np.fft.rfftfreq(int(256 * sampling_rate), 1. / sampling_rate), passband, 'butter', 10)
efield_template = NuRadioReco.framework.base_trace.BaseTrace()
efield_template.set_frequency_spectrum(spec, sampling_rate)
efield_template.apply_time_shift(20. * units.ns, True)

ift_efield_reconstructor = NuRadioReco.modules.iftElectricFieldReconstructor.iftElectricFieldReconstructor.IftElectricFieldReconstructor()
ift_efield_reconstructor.begin(
    electric_field_template=efield_template,
    passband=passband,
    pulse_time_prior=20. * units.ns,
    phase_slope='positive',
    debug=True
)

# Loop over all events in file as initialized in readCoRREAS and perform analysis
for iE, evt in enumerate(readCoREAS.run(detector=det)):
    logger.info("processing event {:d} with id {:d}".format(iE, evt.get_id()))
    station = evt.get_station(args.station_id)
    station.set_station_time(astropy.time.Time('2019-01-01T00:00:00'))
    sim_station = station.get_sim_station()
    det.update(station.get_station_time())
    if simulationSelector.run(evt, station.get_sim_station(), det):
        electricFieldResampler.run(evt, station, det, sampling_rate)
        efieldToVoltageConverter.run(evt, station, det)
        hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)
        channelGenericNoiseAdder.run(evt, station, det, type="rayleigh", amplitude=args.noise_level)
        triggerSimulator.run(evt, station, det, number_concidences=2, threshold=100 * units.mV)
        if station.get_trigger('default_simple_threshold').has_triggered():
            channelBandPassFilter.run(evt, station, det, passband=passband, filter_type='butter', order=10)
            efieldBandPassFilter.run(evt, sim_station, det, passband, filter_type='butter', order=10)
            eventTypeIdentifier.run(evt, station, "forced", 'cosmic_ray')
            # The time differences between channels have to be stored in the channels
            # We can just calculate it from the CR direction.
            for channel in station.iter_channels():
                channel.set_parameter(chp.signal_receiving_zenith, sim_station.get_parameter(stnp.zenith))
                channel.set_parameter(chp.signal_receiving_azimuth, sim_station.get_parameter(stnp.azimuth))
                time_offset = geometryUtilities.get_time_delay_from_direction(
                    sim_station.get_parameter(stnp.zenith),
                    sim_station.get_parameter(stnp.azimuth),
                    det.get_relative_position(station.get_id(), channel.get_id()),
                    1.
                )
                channel.set_parameter(chp.signal_time_offset, time_offset)
            ift_efield_reconstructor.run(evt, station, det, [0, 1, 2, 3], False, use_sim=False)
            channelResampler.run(evt, station, det, sampling_rate=1 * units.GHz)
            electricFieldResampler.run(evt, station, det, sampling_rate=1 * units.GHz)
            eventWriter.run(evt, det)

