from NuRadioReco.modules.base.module import register_run
from NuRadioReco.framework.trigger import SimpleThresholdTrigger
from NuRadioReco.utilities import units

import logging
import numpy as np
import time
import torch


def CalculateTimeOverThresholdIntervals(threshold, tot_bins, amplitudes):
    """
    Calculates the bin-intervals for which the amplitudes are above `threshold`
    for at least `tot_bins`. Returns a list of pairs corresponding to the first
    bin that is above the threshold and the first bin which drops below the threshold

    Parameters
    ----------
    threshold: float
        threshold above which the TOT is considered
    tot_bins: int
        the number of consecutive bins for which the `amplitudes` must be above `threshold`
    amplitudes: np.ndarray or torch.Tensor of floats
        values on which the algorithm will be run

    Returns
    -------
    passing bin intervals: list of pairs of ints
        pairs corresponding to the first bin that is above `threshold` and the first bin
        when the `amplitudes` drop below the `threshold`

    """

    logger = logging.getLogger("CalculateTimeOverThresholdIntervals")

    if isinstance(amplitudes, np.ndarray):
        # Calculate intervals over which the values are above threshold
        intervals = np.where(np.diff(amplitudes > threshold, prepend=0, append=0))[0].reshape(-1, 2)
        # Select only those which are long enough
        intervals = intervals[np.subtract(*intervals.T) <= -tot_bins]
    elif isinstance(amplitudes, torch.Tensor):
        # Calculate intervals over which the values are above threshold
        intervals = torch.where(
            torch.diff(amplitudes.squeeze() > threshold, prepend=torch.Tensor([0]), append=torch.Tensor([0]))
        )[0].reshape(-1, 2)
        # Select only those which are long enough
        intervals = intervals[torch.subtract(*intervals.T) <= -tot_bins].cpu().numpy()
    else:
        msg = f"Supplied amplitudes of type {type(amplitudes)}. Must be np.ndarray or torch.Tensor"
        logger.error(msg)
        raise TypeError(msg)

    return intervals


class triggerSimulator:
    """
    Calculates a GRU-based triggering algorithm using a time-over-threshold algorithm on the netowrk output
    """

    def __init__(self):
        self.__t = 0
        self.logger = logging.getLogger("ChunkedTriggerSimulator")
        self.logger.setLevel(logging.WARNING)

        self.__model = None
        self._device = None
        self.__h = None

    def begin(self, model, device=None, h=None, log_level=None):
        """
        Parameters
        ----------
        model: torch.nn.Module
            initialized Pytorch model that is applied on chunked batches.

        device: torch.device
            hardware indicator to run the model on
        """
        self.logger.setLevel(log_level)

        self._model = model
        self._device = device
        self.__h = h
        
        ## TODO: remove hardcoding here
        self.trigger_waveform_length=256
        self.skip_step_size=200

        # Find if a GPU is found by the PyTorch
        if self._device is None:
            is_cuda = torch.cuda.is_available()
            print(f"CUDA is available: {('no', 'yes')[is_cuda]}")
            if is_cuda:
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")

        # Put the model on the CPU/GPU depending on what is available
        self._model.to(self._device)
        self._model.eval()

    @register_run()
    def run(
        self, evt, station, det, threshold, triggered_channels, vrms_per_channel, trigger_name="default_chunked_trigger"
    ):
        """
        Applies the GRU to the waveforms and calculates TOT condition on the output

        Parameters
        ----------
        evt: None
            argument needs to be included for the `run` decorator, but is not used
        station: Station
            station to run the module on
        det: Detector
            detector instance (to follow normal module format, not used)
        threshold: float
            threshold value for the TOT calculation on the GRU output values

        triggered_channels: list of ints
            list of channel ids that specify which waveforms will be given to the GRU
        vrms_per_channel: list of floats
            rms voltage values that are used to normalize the waveforms into SNR amplitudes
        trigger_name: string
            a unique name of this particular trigger
        """

        if triggered_channels is None or not len(triggered_channels):
            msg = f"Expected to get a list of channels for triggered_channels, instead got {triggered_channels}"
            self.logger.error(msg)
            raise TypeError(msg)

        if not len(vrms_per_channel) or not np.all([vrms_per_channel[station.get_id()][ich] > 0.0 for ich in triggered_channels]):
            msg = f"Argument `vrms_per_channel` is supposed to be a list of (positive-valued) channel RMSs, was given {vrms_per_channel}"
            self.logger.error(msg)
            raise TypeError(msg)

        t_profile = time.time()
        channels_that_passed_trigger = []

        prediction_traces = None
        start_time = None

        # Iterate through the channels, get the waveforms, scale them into units of SNR
        for channel in station.iter_channels():
            channel_id = channel.get_id()
            if channel_id not in triggered_channels:
                continue

            # Get traces normalized by V_RMS
            vrms = vrms_per_channel[station.get_id()][channel_id]
            if prediction_traces is None:
                # Pytorch is slow (by its own admission) in converting lists into Tensor, so we initialize an array here
                ipred = 0
                prediction_traces = np.zeros((len(triggered_channels), len(channel.get_trace())))

            prediction_traces[ipred] = channel.get_trace() / vrms
            ipred += 1

            # Small safety loop to make sure that the channels start at the same t0
            ch_start_time = channel.get_trace_start_time()
            if start_time is not None and ch_start_time != start_time:
                self.logger.warning(
                    f"Channel {channel_id} has a trace_start_time that differs from the other channels. The trigger simulator may not work properly"
                )
            start_time = ch_start_time

        if ipred != len(triggered_channels):
            msg = f"[{trigger_name}] Expected the channels: {triggered_channels} in the file, but found {ipred} waveforms. Maybe some of these channels do not exist in the file"
            self.logger.error(msg)
            raise ValueError(msg)

        # Need to do some shape manipulation based on torch requirements
        torch_tensor = torch.Tensor(prediction_traces).unsqueeze(0)
        assert torch_tensor.ndim == 3  # Ensure a 3D tensor
        assert torch_tensor.shape[0] == 1  # First dim is the unsqueezed part
        assert torch_tensor.shape[2] == len(prediction_traces[0])  # Waveform length
        assert torch_tensor.shape[1] == len(triggered_channels)  # The N channels are the features
        
        print("tensor shape .... !", torch_tensor.shape)
        torch_tensor=torch_tensor.to(self._device)
        ## maximum number of applies for the chunked network

        all_preds=[]
        has_triggered=False
        sampling_rate = station.get_channel(triggered_channels[0]).get_sampling_rate()
      
        trigger_times=[]
        first_index=np.random.randint(self.skip_step_size)
        print("first index ", first_index)
        next_to_last_index=first_index+self.trigger_waveform_length

        with torch.no_grad():
            
            pred_ind=0
            while(next_to_last_index<=len(prediction_traces[0])):

                pred = self._model(torch_tensor[:, :, first_index:next_to_last_index])[0].cpu().item()
                print(pred_ind,"PRED ", pred)
                #all_preds.append(pred.cpu().item())
                if(pred>threshold):
                    ## we trigger
                    has_triggered=True

                    ## setting trigger time in center
                    trigger_times.append(start_time+1.0/sampling_rate*(self.skip_step_size*pred_ind+int(0.5*self.trigger_waveform_length)))
                    

                first_index+=self.skip_step_size
                next_to_last_index=first_index+self.trigger_waveform_length

                pred_ind+=1
                
            #tot_intervals = CalculateTimeOverThresholdIntervals(threshold, tot_bins, yhat.cpu())
            #has_triggered = len(tot_intervals) > 0  # Any number of tot passings constitute a global success for this waveform
            #self.logger.debug(f"[{trigger_name}] Found {len(tot_intervals)} instances of the trigger being passed")
        print("FIRST CHANNEL")
        print(prediction_traces[0,:50])
        # Set the trigger results and times
        #trigger = TimeOverThresholdTrigger(trigger_name, threshold, tot_bins, channels=triggered_channels)
        trigger = SimpleThresholdTrigger(trigger_name, threshold, channels=triggered_channels)
        trigger.set_triggered_channels(triggered_channels)
        trigger.set_triggered(has_triggered)

        if has_triggered:
        
            trigger.set_trigger_time(trigger_times[0])
            trigger.set_trigger_times(trigger_times)
            #self.logger.debug(f"[{trigger_name}] Station has passed trigger, trigger times are {trigger_times / units.ns} ns")
            #self.logger.debug(f"[{trigger_name}] \t--> Time since waveform start: {trigger_times - start_time / units.ns} ns")
            #self.logger.debug(f"[{trigger_name}] \t--> trigger bins: {tot_intervals[:, 0]}, of {yhat.shape[1]} bins")
        else:
            #self.logger.debug(f"[{trigger_name}] Station has NOT passed trigger")

        station.set_trigger(trigger)

        self.__t += time.time() - t_profile

    def end(self):
        from datetime import timedelta

        self.logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        self.logger.info(f"Total time used by this module is: {dt}")
        return dt
