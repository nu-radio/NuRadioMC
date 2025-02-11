import logging
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelCWNotchFilter
import NuRadioReco.modules.channelSignalReconstructor

import NuRadioReco.modules.RNO_G.dataProviderRNOG
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.io.eventWriter

import NuRadioReco.detector.RNO_G.rnog_detector
from NuRadioReco.utilities import units


logger = logging.getLogger("NuRadioReco.example.RNOG.rnog_standard_data_processing")
logger.setLevel(logging.INFO)

channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelResampler.begin()

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()

channelCWNotchFilter = NuRadioReco.modules.channelCWNotchFilter.channelCWNotchFilter()
channelCWNotchFilter.begin()

hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
hardwareResponseIncorporator.begin()

channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor(log_level=logging.WARNING)
channelSignalReconstructor.begin()


def process_event(evt, det):
    """
    Recommended preprocessing for RNO-G events

    Parameters
    ----------
    evt : NuRadioReco.event.Event
        Event to process
    det : NuRadioReco.detector.detector.Detector
        Detector object
    """

    # loop over all stations in the event. Typically we only have a single station per event.
    for station in evt.get_stations():
        # The RNO-G detector changed over time (e.g. because certain hardware components were replaced).
        # The time-dependent detector description has this information but needs to be updated with the
        # current time of the event.
        det.update(station.get_station_time())

        # The first step is to upsample the data to a higher sampling rate. This will e.g. allow to
        # determine the maximum amplitude and time of the signal more accurately. Studies showed that
        # upsampling to 5 GHz is good enough for most task.
        # In general, we need to find a good compromise between the data size (and thus processing time)
        # and the required accuracy.
        # Also remember to always downsample the data again before saving it to disk to avoid unnecessary
        # large files.
        channelResampler.run(evt, station, det, sampling_rate=5 * units.GHz)


        # Our antennas are only sensitive in a certain frequency range. Hence, we should apply a bandpass filter
        # in the range where the antennas are sensitive. This will reduce the noise in the data and make the
        # signal more visible. The optimal frequency range and filter type depends on the antenna type and
        # the expected signal. For the RNO-G antennas, a bandpass filter between 100 MHz and 600 MHz is a good
        # general choice.
        channelBandPassFilter.run(
            evt, station, det,
            passband=[0.1 * units.GHz, 0.6 * units.GHz],
            filter_type='butter', order=10)

        # The signal chain amplifies and disperses the signal. This module will correct for the effects of the
        # analog signal chain, i.e., everything between the antenna and the ADC. This will typically increase the 
        # signal-to-noise ratio and make the signal more visible.
        hardwareResponseIncorporator.run(evt, station, det, sim_to_data=False, mode='phase_only')

        # The antennas often pick up continuous wave (CW) signals from noise various sources. These signals can be
        # very strong and can make it difficult to see other signals. This module will remove the CW signals from
        # the data by dynamically identifying and removing the contaminated frequency bins.
        # An alternative module is the channelSineWaveFilter, which only removes the noise contribution from the CW
        # but not the thermal noise of that frequency. However, this is more computationally expensive.
        channelCWNotchFilter.run(evt, station, det)

        # The data is now preprocessed and ready for further analysis. The next steps depend on the analysis
        # you want to perform. For example, you can now search for signals in the data, determine the arrival
        # direction of the signal, or reconstruct the energy of the signal.
        # The channelSignalReconstructor module is a good starting point for the signal reconstruction.
        channelSignalReconstructor.run(evt, station, det)