import numpy as np
from scipy.signal import hilbert


class Hilbert:
    def __init__(self, nsegs=8):
        self.nsegs = nsegs

    def hilbert_snr(self, trace):
        hill = np.abs(hilbert(trace))
        hill_max_idx = np.argmax(hill)
        hill_max = hill[hill_max_idx]
        
        traceLen = len(trace)
        rms = 1e100

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

        if(rms == 0.0):
            return 0
        else:
            return hill_max/rms




