from NuRadioReco.modules.base.module import register_run
import ROOT
import random
from NuRadioReco.utilities import units
import numpy as np
import logging
logger = logging.getLogger('noiseImporter')


class noiseImporter:
    """
    Imports recorded noise from ARIANNA station. The recorded noise needs to match the station geometry and sampling
    as chosen with channelResampler and channelLengthAdjuster
    
    For different stations, new noise files need to be used. 
    Collect forced triggers from any type of station to use for analysis.
    A seizable fraction of data is recommended for accuracy. 
    
    The noise will be random. This module therefore might produce non-reproducible results on a single event basis,
    if run several times.  
    """

    def begin(self, noise_files):

        self.nt = ROOT.TChain("CalibTree")
        if len(noise_files) > 1:
            logger.warning("Only using the first noise file, more is not implemented yet")
        self.nt.Add(noise_files[0])

        self.data = ROOT.TSnCalWvData()
        self.nt.SetBranchAddress("AmpOutData.", self.data)

        self.nevts = self.nt.GetEntries()

    @register_run()
    def run(self, evt, station, det):
        # loop over stations in simulation
        for channel in station.iter_channels():
            channel_id = channel.get_id()

            # pick a noise waveform
            noise_event = random.randint(0, self.nevts)
            self.nt.GetEntry(noise_event)
#                 nchans = self.data.GetNumChans()
            noise_samples = self.data.GetNumSamplesOn(0)

            trace = channel.get_trace()
            if (noise_samples != trace.shape[0]):
                logger.warning("Mismatch: Noise has {0} and simulation {1} samples\n Not adding noise!".format(noise_samples, trace.shape[0]))
            else:
                noise_trace = np.array(self.data.GetDataOnCh(channel_id)) * units.mV
                noise_trace += trace

                channel.set_trace(noise_trace, channel.get_sampling_rate())

    def end(self):
        pass
