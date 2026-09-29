import logging
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelCWNotchFilter
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.channelSinewaveSubtraction

import NuRadioReco.modules.RNO_G.dataProviderRNOG
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.io.eventWriter
import numpy as np 

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

channelSinewaveSubtraction = NuRadioReco.modules.channelSinewaveSubtraction.channelSinewaveSubtraction()
channelSinewaveSubtraction.begin(save_filtered_freqs=False, freq_band= (0.05, 0.6))
LAB4D_SAMPLING_BLOCK_SIZE = 64


def process_event(evt, det, run_no = None, us_channels = None):
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
        # det.update(station.get_station_time())

        # The first step is to upsample the data to a higher sampling rate. This will e.g. allow to
        # determine the maximum amplitude and time of the signal more accurately. Studies showed that
        # upsampling to 5 GHz is good enough for most task.
        # In general, we need to find a good compromise between the data size (and thus processing time)
        # and the required accuracy.
        # Also remember to always downsample the data again before saving it to disk to avoid unnecessary
        # large files.

        if (us_channels != None):
            if (evt.get_id() == 1796):
                print("us channels processing")
            for ch in station.iter_channels():
                if (True == True):
                    trace = ch.get_trace()
                    sampling_rate = ch.get_sampling_rate()

                    trimmed_trace = trace[LAB4D_SAMPLING_BLOCK_SIZE:-LAB4D_SAMPLING_BLOCK_SIZE]

                    start_time = ch.get_trace_start_time()
                    delta_t = 1 / sampling_rate
                    new_start_time = start_time + LAB4D_SAMPLING_BLOCK_SIZE * delta_t

                    ch.set_trace(trimmed_trace, sampling_rate)
                    ch.set_trace_start_time(new_start_time)

        channelResampler.run(evt, station, det, sampling_rate=5 * units.GHz)
        """ 
        snr_before = {}
        power_before = {}

        for ch in station.iter_channels():
            trace = ch.get_trace()
            times = ch.get_times()
            power = np.sum(trace**2) / len(trace)
            vMax = max(trace)
            vMin = min(trace)
            vp2p = vMax - vMin
            rms = 1e100
            traceLen = len(trace)
            segLen = traceLen // 8
            if(segLen < 2):
                raise Exception("Number of segments cannot be more than number of points in trace. Abort.")

            segRem = traceLen % 8

            for i in range(8):
                start = i*segLen
                if(i < segRem):
                    start += i
                end = start + segLen
                if(i < segRem):
                    end += 1

                thisRms = np.sqrt(np.mean(trace[start:end+1]**2))

                if(thisRms < rms):
                    rms = thisRms

            snr = vp2p/rms/2.0
            snr_before[ch.get_id()] = snr 
            power_before[ch.get_id()] = power

        """
        channelSinewaveSubtraction.run(evt, station, det, algorithm="simple", peak_prominence=3.0)
        """
        snr_after = {}
        power_after = {}

        for ch in station.iter_channels():
            trace = ch.get_trace()
            times = ch.get_times()
            power = np.sum(trace**2) / len(trace)
            vMax = max(trace)
            vMin = min(trace)
            vp2p = vMax - vMin
            rms = 1e100
            traceLen = len(trace)
            segLen = traceLen // 8
            if(segLen < 2):
                raise Exception("Number of segments cannot be more than number of points in trace. Abort.")

            segRem = traceLen % 8

            for i in range(8):
                start = i*segLen
                if(i < segRem):
                    start += i
                end = start + segLen
                if(i < segRem):
                    end += 1

                thisRms = np.sqrt(np.mean(trace[start:end+1]**2))

                if(thisRms < rms):
                    rms = thisRms
            
            snr = vp2p/rms/2.0
            snr_after[ch.get_id()] = snr 
            power_after[ch.get_id()] = power
        
        np.save(f"/users/PAS2608/avijai/rno-g/NuRadioMC/NuRadioReco/examples/RNOG/interferometric_reco_nu/snr_pwr_diff_0721_3/{run_no}_{evt.get_id()}_snr_before.npy", snr_before)
        np.save(f"/users/PAS2608/avijai/rno-g/NuRadioMC/NuRadioReco/examples/RNOG/interferometric_reco_nu/snr_pwr_diff_0721_3/{run_no}_{evt.get_id()}_snr_after.npy", snr_after)
        np.save(f"/users/PAS2608/avijai/rno-g/NuRadioMC/NuRadioReco/examples/RNOG/interferometric_reco_nu/snr_pwr_diff_0721_3/{run_no}_{evt.get_id()}_pwr_before.npy", power_before)
        np.save(f"/users/PAS2608/avijai/rno-g/NuRadioMC/NuRadioReco/examples/RNOG/interferometric_reco_nu/snr_pwr_diff_0721_3/{run_no}_{evt.get_id()}_pwr_after.npy", power_after)
        """
         
        """
        if (us_channels != None):
            if (evt.get_id() == 1796):
                print("us channels processing")
            for ch in station.iter_channels():
                if (True == True):
                    trace = ch.get_trace()
                    sampling_rate = ch.get_sampling_rate()

                    trimmed_trace = trace[LAB4D_SAMPLING_BLOCK_SIZE:-LAB4D_SAMPLING_BLOCK_SIZE]

                    start_time = ch.get_trace_start_time()
                    delta_t = 1 / sampling_rate
                    new_start_time = start_time + LAB4D_SAMPLING_BLOCK_SIZE * delta_t

                    ch.set_trace(trimmed_trace, sampling_rate)
                    ch.set_trace_start_time(new_start_time)
        """
        pad_info = {}
        for ch in station.iter_channels():
            trace = ch.get_trace()
            times = ch.get_times()
            #print(trace, times)
            start_time = ch.get_trace_start_time()
            dt = times[1] - times[0]
            sampling_rate = ch.get_sampling_rate()
            delta_t = 1 / sampling_rate
            pad = int(50 / delta_t)
            padded_trace = np.pad(trace, (pad, pad), mode='constant')
            start_time = ch.get_trace_start_time()
            pad_info[ch.get_id()] = {
                    "pad": pad,
                    "orig_len": len(trace),
                    "start_time": ch.get_trace_start_time()}
            ch.set_trace(padded_trace, sampling_rate)
            ch.add_trace_start_time(-pad * delta_t)

        #channelSinewaveSubtraction.run(evt, station, det, algorithm="sliding", peak_prominence=1.5)
        

        # Our antennas are only sensitive in a certain frequency range. Hence, we should apply a bandpass filter
        # in the range where the antennas are sensitive. This will reduce the noise in the data and make the
        # signal more visible. The optimal frequency range and filter type depends on the antenna type and
        # the expected signal. For the RNO-G antennas, a bandpass filter between 100 MHz and 600 MHz is a good
        # general choice.
        channelBandPassFilter.run(
            evt, station, det,
            passband=[0.1 * units.GHz, 0.6 * units.GHz],
            filter_type='butter', order=10)

        for ch in station.iter_channels():
            trace = ch.get_trace()
            times = ch.get_times()
            dt = times[1] - times[0]
            sampling_rate = ch.get_sampling_rate()
            delta_t = 1 / sampling_rate
            pad = int(50 / delta_t)
            info = pad_info[ch.get_id()]
            pad = info["pad"]
            orig_len = info["orig_len"]
            start_time = info["start_time"]
            trimmed_trace = trace[pad : pad + orig_len]
            ch.set_trace(trimmed_trace, sampling_rate)
            ch.set_trace_start_time(start_time)


        # The signal chain amplifies and disperses the signal. This module will correct for the effects of the
        # analog signal chain, i.e., everything between the antenna and the ADC. This will typically increase the 
        # signal-to-noise ratio and make the signal more visible.
        hardwareResponseIncorporator.run(evt, station, det, sim_to_data=False, mode='phase_only')

        # The antennas often pick up continuous wave (CW) signals from noise various sources. These signals can be
        # very strong and can make it difficult to see other signals. This module will remove the CW signals from
        # the data by dynamically identifying and removing the contaminated frequency bins.
        # An alternative module is the channelSineWaveFilter, which only removes the noise contribution from the CW
        # but not the thermal noise of that frequency. However, this is more computationally expensive.
        # channelCWNotchFilter.run(evt, station, det)

        # The data is now preprocessed and ready for further analysis. The next steps depend on the analysis
        # you want to perform. For example, you can now search for signals in the data, determine the arrival
        # direction of the signal, or reconstruct the energy of the signal.
        # The channelSignalReconstructor module is a good starting point for the signal reconstruction.
        channelSignalReconstructor.run(evt, station, det)
        
