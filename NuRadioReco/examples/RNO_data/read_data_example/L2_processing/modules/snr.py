import numpy as np

class SNR:

    def __init__(self, nsegs=8):
        self.nsegs = nsegs 
    
    def get_snr_single(self, channel_times, channel_signals):
        vMax = max(channel_signals)
        vMin = min(channel_signals)
        vpp = vMax - vMin
        rms = 1e100
        traceLen = len(channel_signals)
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

            thisRms = np.sqrt(np.mean(channel_signals[start:end+1]**2))

            if(thisRms < rms):
                rms = thisRms

        snr = vpp/rms/2.0

        return snr

    def vpp(self, event, station):

        vp2p = {}

        for ch in station.iter_channels():
            trace = ch.get_trace()
            times = ch.get_times()
            vMax = max(trace)
            vMin = min(trace)

            vp2p[ch.get_id()] = vMax - vMin

        return vp2p 


    def get_min_segmented_rms(self, event, station):
        all_rms = {}

        for ch in station.iter_channels():
            trace = ch.get_trace()
            times = ch.get_times()
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
            all_rms[ch.get_id()] = rms

        return all_rms

    def get_snr(self, event, station):
        vp2p = self.vpp(event, station)
        rms = self.get_min_segmented_rms(event, station)
        snr_all = {}

        for ch in vp2p:
            if(rms[ch] == 0.0):
                snr_all[ch] = 0
            else:
                snr = vp2p[ch]/rms[ch]/2.0
                snr_all[ch] = snr

        return snr_all

    def run(self, event, station, excluded_channels=[]):
        snr = self.get_snr(event, station)
        
        snr_all = []
        for chan in snr:
            if(chan in excluded_channels):
                continue
            else:
                snr_all.append(snr[chan])

        avg_snr = np.mean(snr_all)

        return avg_snr









