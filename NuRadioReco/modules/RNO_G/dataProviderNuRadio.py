from NuRadioReco.modules.base.module import register_run

import NuRadioReco.modules.io.eventReader
import NuRadioReco.modules.RNO_G.channelPreprocessor

import logging
logger = logging.getLogger('NuRadioReco.RNO_G.dataProviderNuRadio')


class dataProviderNuRadio:
    """
    NUR-file counterpart to ``dataProviderRNOG``.

    Reads NuRadioReco-universal ``.nur`` files (typically simulation
    output, e.g. the RNO-G CR simulation set) with ``eventReader`` and
    then runs the shared ``channelPreprocessor`` chain. The detector is
    updated to each event's station time before preprocessing so the
    downstream behavior matches the ROOT path.

    Use this class in analyses that need the same processed view of
    both real data (ROOT via ``dataProviderRNOG``) and simulation (NUR
    via this module), with no divergence in which steps are applied.

    See Also
    --------
    NuRadioReco.modules.RNO_G.dataProviderRNOG
    NuRadioReco.modules.RNO_G.channelPreprocessor
    NuRadioReco.modules.io.eventReader
    """

    def __init__(self):
        self.reader = NuRadioReco.modules.io.eventReader.eventReader()
        self.preprocessor = NuRadioReco.modules.RNO_G.channelPreprocessor.channelPreprocessor()

    def begin(self, files, det, reader_kwargs=None, preprocessor_config=None):
        """Initialize the reader and preprocessor.

        Parameters
        ----------
        files : str or list of str
            Path(s) to ``.nur`` file(s) to read.
        det : Detector
            Detector object. Updated to each event's station time
            before preprocessing runs.
        reader_kwargs : dict, optional
            Forwarded to ``eventReader.begin`` (``read_detector``,
            ``log_level``).
        preprocessor_config : dict, optional
            Overrides for ``channelPreprocessor._DEFAULT_CONFIG``.
        """
        self.files = files
        self.detector = det

        self.reader.begin(files, **(reader_kwargs or {}))
        self.preprocessor.begin(config=preprocessor_config)

    def end(self):
        """End the reader and preprocessor."""
        self.reader.end()
        self.preprocessor.end()

    @register_run()
    def run(self):
        """Iterate events, update detector, apply preprocessing, yield.

        Yields
        ------
        event : NuRadioReco.framework.event.Event
            The processed event.
        """
        for event in self.reader.run():
            station = event.get_station()
            self.detector.update(station.get_station_time())

            self.preprocessor.run(event, station, self.detector)

            yield event
