import NuRadioReco.modules.RNO_G.dataProviderRNOG
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.channelResampler
import NuRadioReco.detector.RNO_G.rnog_detector
import NuRadioReco.modules.RNO_G.stationHitFilter
from NuRadioReco.utilities import units, logging as nulogging

from NuRadioReco.examples.RNOG.processing import process_event


from matplotlib import pyplot as plt
import numpy as np
import argparse
import logging
import time

logger = logging.getLogger("NuRadioReco.example.RNOG.rnog_standard_data_processing")
logger.setLevel(nulogging.LOGGING_STATUS)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run standard RNO-G data processing')

    parser.add_argument('filenames', type=str, nargs="*",
                        help='Specify root data files if not specified in the config file')
    parser.add_argument('--outputfile', type=str, required=True, help='Specify the output file')
    parser.add_argument('--detectorfile', type=str, nargs=1, default=None,
                        help="Specify detector file. If you do not specified a file. "
                        "the description is queried from the database.")

    args = parser.parse_args()
    args.outputfile = args.outputfile
    nulogging.set_general_log_level(logging.INFO)
    logger.status(f"writing output to {args.outputfile}")

    # Initialize detector class
    det = NuRadioReco.detector.RNO_G.rnog_detector.Detector(detector_file=args.detectorfile)

    # Initialize io modules
    dataProviderRNOG = NuRadioReco.modules.RNO_G.dataProviderRNOG.dataProviderRNOG()
    dataProviderRNOG.begin(files=args.filenames, det=det)

    eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
    eventWriter.begin(filename=args.outputfile)

    # initialize additional modules
    channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
    channelResampler.begin()

    # Initialize Hit Filter
    stationHitFilter = NuRadioReco.modules.RNO_G.stationHitFilter.stationHitFilter()
    stationHitFilter.begin()

    # For time logging
    t_total = 0

    # Loop over all events (the reader module has options to select events -
    # see class documentation or module arguements in config file)
    for idx, evt in enumerate(dataProviderRNOG.run()):

        if (idx + 1) % 50 == 0:
            print(f'Processing events: {idx + 1}', end='\r')

        t0 = time.time()
        # perform standard RNO-G data processing
        process_event(evt, det)

        ########################
        ########################
        #
        # add more usercode here
        #
        ########################
        ########################

        # the following code is just an example of how to access station and channel information:

        # this assumes that there is only one station per event. If multiple stations are present,
        # this will throw an error. Use evt.get_stations() to iterate over several stations.
        station = evt.get_station()

        # the following code is just an example of how to access reconstructed parameters
        # (the ones that were calculated in the processing steps by the channelSignalReconstructor)
        from NuRadioReco.framework.parameters import channelParameters as chp
        max_SNR = 0
        for channel in station.iter_channels():
            SNR = channel[chp.SNR]['peak_2_peak_amplitude']  # alternatively, channel.get_parameter(chp.SNR)
            max_SNR = max(max_SNR, SNR)
            signal_amplitude = channel[chp.maximum_amplitude_envelope]
            signal_time = channel[chp.signal_time]

            #print(f"Channel {channel.get_id()}: SNR={SNR:.1f}, signal amplitude={signal_amplitude / units.mV:.2f}mV, "
                  #f"signal time={signal_time / units.ns:.2f}ns")


        # the following code is just an example of how to access the channel waveforms and plot them
        # we do it only for high-SNR events
        if max_SNR > 5:
            # iterate over some channels in station and plot them
            fig, ax = plt.subplots(4, 2)
            for channel_id in range(4): # iterate over the first 4 channels
                channel = station.get_channel(channel_id)
                ax[channel_id, 0].plot(channel.get_times()/units.ns, channel.get_trace() / units.mV)  # plot the timetrace in nanoseconds vs. microvolts
                ax[channel_id, 1].plot(channel.get_frequencies() / units.MHz, np.abs(channel.get_frequency_spectrum()) / units.mV)  # plot the frequency spectrum in MHz vs. microvolts
                ax[channel_id, 0].set_xlabel('Time [ns]')
                ax[channel_id, 0].set_ylabel('Voltage [mV]')
                ax[channel_id, 1].set_xlabel('Frequency [MHz]')
                ax[channel_id, 1].set_ylabel('Voltage [mV]')
            plt.tight_layout()
            fig.savefig(f'channel_traces_{idx}.png')
            plt.close(fig)


        # Apply the Hit Filter
        is_passed_HF = stationHitFilter.run(evt, station, det)

        # before saving events to disk, it is advisable to downsample back to the two-times the maximum frequency available in the data, i.e., back to the Nquist frequency
        # this will save disk space and make the data processing faster. The preprocessing applied a 600MHz low-pass filter, so we can downsample to 2GHz without losing information
        channelResampler.run(evt, station, det, sampling_rate=2 * units.GHz)


        # it is advisable to only save the full waveform information for events that pass certain analysis cuts
        # this will save disk space and make the data processing faster
        # Here, we save events that passed the Hit Filter and exclude forced trigger events and RADIANT trigger events
        # Write event - the RNO-G detector class is not stored within the nur files.
        if is_passed_HF and stationHitFilter.is_wanted_trigger_type():
            # save full waveform information
            #print("saving full waveform information")
            eventWriter.run(evt, det=None, mode={'Channels':True, "ElectricFields":True})
        else:
            # only save meta information but no traces to save disk space
            #print("saving only meta information")
            eventWriter.run(evt, det=None, mode={'Channels':False, "ElectricFields":False})


        logger.debug("Time for event: %f", time.time() - t0)
        t_total += time.time() - t0

    dataProviderRNOG.end()
    eventWriter.end()
    stationHitFilter.end()

    logger.status(
        f"\nProcessed {idx + 1} events:"
        f"\n\tTotal time: {t_total:.2f}s"
        f"\n\tTime per event: {t_total / (idx + 1):.2f}s")
