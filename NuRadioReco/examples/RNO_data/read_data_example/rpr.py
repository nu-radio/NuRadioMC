import numpy as np
from scipy.ndimage import uniform_filter1d
from snr import SNR

class RPR:

    def __init__(self, nsegs=8):
        self.nsegs = nsegs

    def get_dt_and_sampling_rate(self, times):
        
        dt = times[1] - times[0]
        sampling_rate = 1/dt
        return dt, sampling_rate

    def get_single_rpr(self, times, trace):
        wf_len = len(trace)
        channel_wf = trace ** 2
        
        # Calculate the smoothing window size based on sampling rate
        dt =  self.get_dt_and_sampling_rate(times)[0]
        sum_win = 25  # Smoothing window in ns
        sum_win_idx = int(np.round(sum_win / dt))  # Convert window size to sample points

        channel_wf = np.sqrt(uniform_filter1d(channel_wf, size=sum_win_idx, mode='constant'))

        # Find the maximum value of the smoothed waveform
        max_bin = np.argmax(channel_wf)
        max_val = channel_wf[max_bin]

        rms = 1e100

        traceLen = len(trace)
        segLen = traceLen // self.nsegs
        if(segLen < 2):
            raise Exception("Number of segments cannot be more than number of points in trace. Abort.")

        segRem = traceLen % self.nsegs
        
        for i in range(self.nsegs):
            start = i*segLen
            if(i < segRem):
                start += i
            end = start + segLen
            if(i < segRem):
                end += 1

            thisRms = np.sqrt(np.mean(trace[start:end+1]**2))

            if(thisRms < rms):
                rms = thisRms

        rpr_val = max_val / rms
        
        return rpr_val


    def get_rpr(self, event, station):
        rpr = {}
        for ch in station.iter_channels():
            trace = ch.get_trace()
            times = ch.get_times()

            wf_len = len(trace)

            # Square the waveform data for further processing
            channel_wf = trace ** 2

            # Calculate the smoothing window size based on sampling rate
            dt =  self.get_dt_and_sampling_rate(times)[0]
            sum_win = 25  # Smoothing window in ns
            sum_win_idx = int(np.round(sum_win / dt))  # Convert window size to sample points

            channel_wf = np.sqrt(uniform_filter1d(channel_wf, size=sum_win_idx, mode='constant'))

            # Find the maximum value of the smoothed waveform
            max_bin = np.argmax(channel_wf)
            max_val = channel_wf[max_bin]

            # Get noise rms from snr module
            noise_sigma = SNR.get_min_segmented_rms(self, event, station)[ch.get_id()]

            # Calculate and return the RPR value
            rpr_val = max_val / noise_sigma
            rpr[ch.get_id()] = rpr_val

        return rpr

    def run(self, event, station, excluded_channels=[]):

        rpr_all = self.get_rpr(event, station)

        chans = list(rpr_all.keys())

        avg_rpr = []

        for chan in chans:
            if chan in excluded_channels:
                continue
            rpr = rpr_all[chan]  # Calculate RPR for the channel
            avg_rpr.append(rpr)

        # Return the average RPR value across all selected channels
        return np.mean(avg_rpr)



