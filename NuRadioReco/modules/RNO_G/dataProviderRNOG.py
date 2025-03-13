from NuRadioReco.modules.base.module import register_run

import NuRadioReco.modules.io.RNO_G.readRNOGDataMattak
import NuRadioReco.modules.RNO_G.channelGlitchDetector
import NuRadioReco.modules.RNO_G.channelBlockOffsetFitter

import NuRadioReco.modules.channelAddCableDelay

import logging
logger = logging.getLogger('NuRadioReco.RNO_G.dataProviderRNOG')

class dataProviderRNOG:
    """
    This module provides an easy access to "processed" RNO-G data.

    It does so by running a set of "processing" modules on the data.
    This module is a wrapper around the following modules (in this order):

    #. `NuRadioReco.modules.io.RNO_G.readRNOGDataMattak`
    #. `NuRadioReco.modules.RNO_G.channelGlitchDetector`
    #. `NuRadioReco.modules.RNO_G.channelBlockOffsetFitter`
    #. `NuRadioReco.modules.channelAddCableDelay`

    The module reads RNO-G data, applies a glitch detection algorithm (does not remove/fix them!),
    fits block offsets (and removes them!) and subtracts cable delays. The voltage calibration is applied
    in the readRNOGDataMattak module. The module also updates the detector object with the station time.

    The readRNOGDataMattak module has two different modes to apply the voltage calibration:

    - If ``read_calibrated_data==True`` (default: False), the "bias scan-based" voltage calibration is applied by mattak.
    - If ``convert_to_voltage==True`` (default: True), a "pseudo" fully-linear voltage calibration is applied.

    Hence, by default only the "pseudo" fully-linear voltage calibration is applied. If you want to apply the
    "bias scan-based" voltage calibration, set ``read_calibrated_data=True`` in the `begin` function.
    You have to make sure that mattak finds the necessary calibration files. If you want to retrieve the raw data, set both to False.

    Please refer to he documentation of the individual modules for more information. In particular the readRNOGDataMattak
    which performs the actual reading of the data and applies the voltage calibration. The reader module is
    based on the mattak package (https://github.com/RNO-G/mattak).

    See Also
    --------
    NuRadioReco.modules.io.RNO_G.readRNOGDataMattak
    NuRadioReco.modules.RNO_G.channelGlitchDetector
    NuRadioReco.modules.RNO_G.channelBlockOffsetFitter
    NuRadioReco.modules.channelAddCableDelay
    """

    def __init__(self):
        self.reader = NuRadioReco.modules.io.RNO_G.readRNOGDataMattak.readRNOGData()

        self.channelGlitchDetector = NuRadioReco.modules.RNO_G.channelGlitchDetector.channelGlitchDetector()
        self.channelBlockOffsetFitter = NuRadioReco.modules.RNO_G.channelBlockOffsetFitter.channelBlockOffsets()
        self.channelCableDelayAdder = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()

    def begin(self, files, det, reader_kwargs={}):
        """ Call the begin method of the modules.

        Parameters
        ----------
        files: list of str
            List of files to read (are passed to the readRNOGDataMattak module).
        det: Detector
            Detector object.
        reader_kwargs: dict (default: {})
            Keyword arguments passed to the reader module readRNOGDataMattak.
        """
        self.files = files
        self.detector = det

        self.reader.begin(self.files, **reader_kwargs)

        self.channelBlockOffsetFitter.begin()
        self.channelGlitchDetector.begin()
        self.channelCableDelayAdder.begin()

    def end(self):
        """ Call the end method of the modules """
        self.reader.end()
        self.channelGlitchDetector.end()

    @register_run()
    def run(self):
        """ Run the modules

        Yields
        ------
        event: Event
            The processed event
        """

        for event in self.reader.run():

            # This will throw an error if the event has more than one station
            station = event.get_station()
            self.detector.update(station.get_station_time())

            self.channelGlitchDetector.run(event, station, self.detector)
            self.channelBlockOffsetFitter.run(event, station, self.detector)
            self.channelCableDelayAdder.run(event, station, self.detector, mode='subtract')

            yield event
