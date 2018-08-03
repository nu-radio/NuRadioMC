import logging
import numpy as np
import argparse
import matplotlib.pyplot as plt

from NuRadioReco.utilities import units
import NuRadioReco.detector.detector as detector
import NuRadioReco.modules.io.eventReader
from NuRadioReco.utilities import templates
import NuRadioReco.modules.channelTemplateCorrelation
import NuRadioReco.modules.voltageToEfieldConverter
import NuRadioReco.modules.io.eventWriter



# Logging level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SimpleReconstruction')

# Parse eventfile as argument
parser = argparse.ArgumentParser(description='NuRadioSim file')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC simulation result')
parser.add_argument('detectordescription', type=str,
                    help='path to detectordescription')
args = parser.parse_args()


# read in detector positions (this is a dummy detector)
det = detector.Detector(json_filename=args.detectordescription)
station_id = 101

# initialize modules
eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
channelTemplateCorrelation = NuRadioReco.modules.channelTemplateCorrelation.channelTemplateCorrelation()
voltageToEfieldConverter = NuRadioReco.modules.voltageToEfieldConverter.voltageToEfieldConverter()
eventReader = NuRadioReco.modules.io.eventReader.eventReader()

# Name outputfile
output_filename = "Simple_reconstruction_results.ari"
eventReader.begin(args.inputfilename)
channelTemplateCorrelation.begin(debug=False)
voltageToEfieldConverter.begin()
eventWriter.begin(output_filename, max_file_size=1000)

# starting angles for reconstruction
starting_zenith = 110 * units.degree
starting_azimuth = 10 * units.degree

i_events_saved = 0
for iE, event in enumerate(eventReader.run()):
    logger.info("Event ID {}".format(event.get_id()))
    stations = event.get_stations()
    for st, station in enumerate(stations):
        logger.info("Station ID {}".format(station.get_id()))
        station['zenith'] = starting_zenith
        station['azimuth'] = starting_azimuth
#         for ch, channel in enumerate(station.get_channels()):
#             logger.info("Channel ID {}".format(channel.get_id()))
#         channelTemplateCorrelation.run(event, station, det, channels_to_use=[0, 1, 2], cosmic_ray=False,
#             n_templates=1)
#
#         ref_template = templates.Templates().get_nu_ref_template(station_id)
#
#         plt.figure()
#         for ch, channel in enumerate(station.get_channels()):
#             if ch == 1:
#                 plt.plot(ref_template/np.max(np.abs(ref_template))*np.max(np.abs(channel.get_trace())),label='template')
#             plt.plot(channel.get_times(),channel.get_trace(),label='Channel {}'.format(ch))
#             print "Channel {0}, corr {1} at {2}".format(ch,channel['nu_ref_xcorr'],channel['nu_ref_xcorr_time'])
#         plt.legend()
#         plt.show()


        voltageToEfieldConverter.run(event, station, det)

    eventWriter.run(event)


    i_events_saved += 1
    if((i_events_saved % 100) == 0):
        print("saving event {}".format(i_events_saved))

