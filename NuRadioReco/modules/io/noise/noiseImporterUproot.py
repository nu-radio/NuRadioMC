from NuRadioReco.modules.base.module import register_run
import uproot
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
        # TODO: maybe better use uproot.concatenate(list-of-trees), but need to get interpretations right first to also read trigger info
        data = []
        for noise_file in noise_files:
            with uproot.open(noise_file) as nf:
                nt = nf["CalibTree"]
                data.append(np.array(nt["AmpOutData."]["AmpOutData.fData"].array()))
        self.data = np.concatenate(data)
        self.nevts = len(self.data)

    @register_run()
    def run(self, evt, station, det):
        # loop over stations in simulation
        for channel in station.iter_channels():
            channel_id = channel.get_id()

            # pick a noise waveform
            noise_event = random.randint(0, self.nevts)
            nchans = np.shape(self.data)[1]

            # check if simulated channel exists in input file
            # otherwise pick a smaller channel number
            if (channel_id > nchans-1):
                orig_channel_id = channel_id
                channel_id %= nchans
                logger.warning("Channel {0} not in noise file ({1} channels): Using channel {2}".format(origi_channel_id, nchans, channel_id))

            noise_samples = np.shape(self.data)[-1]

            trace = channel.get_trace()
            if (noise_samples != trace.shape[0]):
                logger.warning("Mismatch: Noise has {0} and simulation {1} samples\n Not adding noise!".format(noise_samples, trace.shape[0]))
            else:
                noise_trace = np.array(self.data[noise_event][channel_id]) * units.mV
                noise_trace += trace

                channel.set_trace(noise_trace, channel.get_sampling_rate())

    def end(self):
        pass
