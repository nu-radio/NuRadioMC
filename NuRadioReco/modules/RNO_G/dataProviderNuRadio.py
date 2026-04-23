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
            yield self._preprocess_and_return(event)

    def get_event_ids(self):
        """List ``(run_number, event_number)`` pairs in the opened file(s).

        For callers that want random access via ``get_event`` instead of
        iterating the generator from ``run()``.
        """
        return self._fin().get_event_ids()

    def get_event(self, run_number, event_number):
        """Return a single preprocessed event by ``(run, event_number)``.

        Returns
        -------
        event : NuRadioReco.framework.event.Event or None
            The preprocessed event, or None if the reader has no event
            with that id.

        Notes
        -----
        Random access avoids walking the file for callers that only need
        a subset of events (e.g. chunked reco re-reading specific events
        between passes). The returned event has the same preprocessing
        as events yielded by ``run()``.
        """
        event = self._fin().get_event(event_id=(run_number, event_number))
        if event is None:
            return None
        return self._preprocess_and_return(event)

    def _fin(self):
        # eventReader's underlying NuRadioRecoio handle. The `_eventReader__fin`
        # attribute is name-mangled; accessing it from outside the class is
        # what the rest of the NuRadioReco RNO-G examples do. A proper
        # `get_event` / `get_event_ids` on eventReader itself would be
        # cleaner; for now we keep the access here so callers don't have to.
        return self.reader._eventReader__fin

    def _preprocess_and_return(self, event):
        station = event.get_station()
        self.detector.update(station.get_station_time())
        self.preprocessor.run(event, station, self.detector)
        return event
