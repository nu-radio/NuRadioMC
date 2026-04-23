from NuRadioReco.modules.base.module import register_run

import NuRadioReco.modules.io.RNO_G.readRNOGDataMattak
import NuRadioReco.modules.RNO_G.channelPreprocessor

import logging
logger = logging.getLogger('NuRadioReco.RNO_G.dataProviderRNOG')

class dataProviderRNOG:
    """
    This module provides an easy access to "processed" RNO-G ROOT data.

    It reads RNO-G data with ``readRNOGDataMattak`` and then runs the
    shared ``channelPreprocessor`` chain (block-offset removal, glitch
    detection, cable-delay subtraction, and optionally resampling, CW
    removal and bandpass filtering). The voltage calibration is applied
    inside the reader. The detector is updated to the event's station
    time before preprocessing.

    The ``readRNOGDataMattak`` module has two different modes to apply the voltage calibration:

    - If ``read_calibrated_data==True`` (default: False), the "bias scan-based" voltage calibration is applied by mattak.
    - If ``convert_to_voltage==True`` (default: True), a "pseudo" fully-linear voltage calibration is applied.

    Hence, by default only the "pseudo" fully-linear voltage calibration is applied. If you want to apply the
    "bias scan-based" voltage calibration, set ``read_calibrated_data=True`` in the `begin` function.
    You have to make sure that mattak finds the necessary calibration files. If you want to retrieve the raw data, set both to False.

    The per-channel preprocessing chain and its flags live in
    ``channelPreprocessor``. See that module's ``_DEFAULT_CONFIG`` for
    available knobs.

    See Also
    --------
    NuRadioReco.modules.io.RNO_G.readRNOGDataMattak
    NuRadioReco.modules.RNO_G.channelPreprocessor
    """

    def __init__(self):
        self.reader = NuRadioReco.modules.io.RNO_G.readRNOGDataMattak.readRNOGData()
        self.preprocessor = NuRadioReco.modules.RNO_G.channelPreprocessor.channelPreprocessor()

    def begin(self, files, det, reader_kwargs={}, preprocessor_config=None):
        """ Call the begin method of the reader and preprocessor.

        Parameters
        ----------
        files: list of str
            List of files to read (are passed to the readRNOGDataMattak module).
        det: Detector
            Detector object.
        reader_kwargs: dict (default: {})
            Keyword arguments passed to the reader module `NuRadioReco.modules.io.RNO_G.readRNOGDataMattak`.
        preprocessor_config: dict, optional
            Overrides for ``channelPreprocessor._DEFAULT_CONFIG`` (e.g.
            ``{"apply_bandpass": True, "bandpass_band": (0.1, 0.6)}``).
        """
        self.files = files
        self.detector = det

        apply_baseline_correction = reader_kwargs.pop('apply_baseline_correction', None)
        if apply_baseline_correction is not None:
            logger.warning(
                "The 'apply_baseline_correction' argument is kwargs will be ignored. "
                "Instead the 'channelBlockOffsetFitter' is used explicitly in the module sequence.")

        self.reader.begin(self.files, apply_baseline_correction=None, **reader_kwargs)
        self.preprocessor.begin(config=preprocessor_config)

    def end(self):
        """ Call the end method of the modules """
        self.reader.end()
        self.preprocessor.end()

    @register_run()
    def run(self):
        """ Run the modules

        Yields
        ------
        event: Event
            The processed event
        """

        for event in self.reader.run():
            yield self._preprocess_and_return(event)

    def get_event_ids(self):
        """List ``(run_number, event_number)`` pairs in the opened file(s).

        For callers that want random access via ``get_event`` instead of
        iterating the generator from ``run()``.
        """
        return self.reader.get_event_ids()

    def get_event(self, run_number, event_number):
        """Return a single preprocessed event by ``(run, event_number)``.

        Returns
        -------
        event : NuRadioReco.framework.event.Event or None
            The preprocessed event, or None if the reader returned None
            (event not present in the opened file(s)).

        Notes
        -----
        Random access is appreciably faster than iterating ``run()`` and
        skipping for workloads that only touch a subset of events in the
        file(s) (e.g. chunked reco, burn-sample filtering). The returned
        event has the same preprocessing applied as events yielded by
        ``run()``.
        """
        event = self.reader.get_event(run_number, event_number)
        if event is None:
            return None
        return self._preprocess_and_return(event)

    def _preprocess_and_return(self, event):
        # This will throw an error if the event has more than one station
        station = event.get_station()
        self.detector.update(station.get_station_time())
        self.preprocessor.run(event, station, self.detector)
        return event
