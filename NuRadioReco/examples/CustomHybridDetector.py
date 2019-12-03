import argparse
import NuRadioReco.modules.io.coreas.readCoREAS
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.framework.hybrid_shower
import NuRadioReco.detector.detector
import pickle
import datetime
import numpy
"""
To accomodate radio experiments that are part of a hybrid detector, NuRadioReco
can support custom detector classes and write them into  NuRadioReco event files.
This example shows what such a custom detector class should look like and how it
can be added to an event file.
"""
parser = argparse.ArgumentParser(description='Example showing how a custom hybrid detector can be added to an event file.')
parser.add_argument('--inputfilename', type=str, help='path to CoREAS file', default='example_data/example_event.h5')
parser.add_argument('--detectorfilename', type=str, help='path to detector description', default='example_data/arianna_detector_db.json')
parser.add_argument('--stationID', type=int, help='ID of the station to simulate', default=32)
parser.add_argument('--outputfilename', type=str, help='name of the output file', default='custom_detector.nur')
args = parser.parse_args()

#setting up stuff we need to read the CoREAS file
coreas_reader = NuRadioReco.modules.io.coreas.readCoREAS.readCoREAS()
coreas_reader.begin([args.inputfilename], args.stationID)
event_writer = NuRadioReco.modules.io.eventWriter.eventWriter()
event_writer.begin(args.outputfilename)
detector = NuRadioReco.detector.detector.Detector(args.detectorfilename)
detector.update(datetime.datetime(2019,1,1))


#define a custom detector class we want to store in the event
class CustomDetector():
    #the custom detector's creator must not require any arguments
    def __init__(self):
        self.__data = None

    #define some custom functions we want our detector to have
    def set_data(self, data):
        self.__data = data
    def get_data(self):
        return self.__data

    #the custom detector must know how to serialize and deserialize itself
    def serialize(self):
        data_pkl = pickle.dumps(self.__data, protocol=4)
        return data_pkl

    def deserialize(self, data_pkl):
        self.__data = pickle.loads(data_pkl)

for event in coreas_reader.run(detector):
    #This import is needed so we can read it later
    from CustomHybridDetector import CustomDetector
    #Create custom detector
    custom_detector = CustomDetector()
    #generate some data for the custom detector to store
    custom_data = numpy.random.random(10)
    custom_detector.set_data(custom_data)
    #create hybrid shower to hold the detector
    shower = NuRadioReco.framework.hybrid_shower.HybridShower('CustomShower')
    shower.set_hybrid_detector(custom_detector)
    #add shower to the hybrid information
    event.get_hybrid_information().add_hybrid_shower(shower)
    event_writer.run(event)
