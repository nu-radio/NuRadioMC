from NuRadioReco.modules.base.module import register_run
from NuRadioReco.framework.trigger import TimeOverThresholdTrigger
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
        self.logger = logging.getLogger("GRUTriggerSimulator")
        self.logger.setLevel(logging.WARNING)

        self.__model = None
        self._device = None
        self.__h = None

    def begin(self, model, device=None, h=None, log_level=None):
        """
        Parameters
        ----------
        model: torch.nn.Module
            initialized Pytorch GRU model that takes N-input waveforms and outputs a single output value
            for each time bin. Must include a `init_hidden` function that returns the initialized hidden state
        device: torch.device
            hardware indicator to run the model on
        """
        self.logger.setLevel(log_level)

        self._model = model
        self._device = device
        self.__h = h

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
        self, evt, station, det, threshold, tot_bins, triggered_channels, vrms_per_channel, trigger_name="default_gru_trigger"
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
        tot_bins: int
            the number of consecutive bins for which the GRU output must be above `threshold`
        triggered_channels: list of ints
            list of channel ids that specify which waveforms will be given to the GRU
        vrms_per_channel: list of floats
            rms voltage values that are used to normalize the waveforms into SNR amplitudes
        trigger_name: string
            a unique name of this particular trigger
        """

        if not callable(getattr(self._model, "init_hidden", None)):
            msg = f"Argument `model` is expected to have `init_hidden` implemented which returns the initial hidden state"
            self.logger.error(msg)
            raise NotImplementedError(msg)

        if int(tot_bins) != tot_bins:
            msg = f"Argument `tot_bins` is supposed to be an int, was given {tot_bins}"
            self.logger.error(msg)
            raise TypeError(msg)

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
        torch_tensor = torch.Tensor(prediction_traces).transpose(0, 1).unsqueeze(0)
        assert torch_tensor.ndim == 3  # Ensure a 3D tensor
        assert torch_tensor.shape[0] == 1  # First dim is the unsqueezed part
        assert torch_tensor.shape[1] == len(prediction_traces[0])  # Waveform length
        assert torch_tensor.shape[2] == len(triggered_channels)  # The N channels are the features

        # Run the network a few times on this data to build up the hidden state
        if self.__h is None:
            with torch.no_grad():
                n_warmup = 10
                self.logger.info(
                    f"[{trigger_name}] Running one time warmup of the GRU on the passed in waveforms. Will process this waveform {n_warmup} times"
                )
                self.__h = self._model.init_hidden()
                for i in range(n_warmup):  # Do a little warm up to get the hidden state built up
                    yhat, self.__h = self._model(torch_tensor.to(self._device), self.__h)

        # Apply the network to this event's data and calculate TOT on the output
        with torch.no_grad():
            yhat, self.__h = self._model(torch_tensor.to(self._device), self.__h)
            tot_intervals = CalculateTimeOverThresholdIntervals(threshold, tot_bins, yhat.cpu())
            has_triggered = len(tot_intervals) > 0  # Any number of tot passings constitute a global success for this waveform
            self.logger.debug(f"[{trigger_name}] Found {len(tot_intervals)} instances of the trigger being passed")

        # Set the trigger results and times
        trigger = TimeOverThresholdTrigger(trigger_name, threshold, tot_bins, channels=triggered_channels)
        trigger.set_triggered_channels(triggered_channels)
        trigger.set_triggered(has_triggered)
        if has_triggered:
            sampling_rate = station.get_channel(triggered_channels[0]).get_sampling_rate()
            dt = 1.0 / sampling_rate
            trigger_times = start_time + dt * tot_intervals[:, 0]
            trigger.set_trigger_time(trigger_times[0])
            trigger.set_trigger_times(trigger_times)
            self.logger.debug(f"[{trigger_name}] Station has passed trigger, trigger times are {trigger_times / units.ns} ns")
            self.logger.debug(f"[{trigger_name}] \t--> Time since waveform start: {trigger_times - start_time / units.ns} ns")
            self.logger.debug(f"[{trigger_name}] \t--> trigger bins: {tot_intervals[:, 0]}, of {yhat.shape[1]} bins")
        else:
            self.logger.debug(f"[{trigger_name}] Station has NOT passed trigger")

        station.set_trigger(trigger)

        self.__t += time.time() - t_profile

    def end(self):
        from datetime import timedelta

        self.logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        self.logger.info(f"Total time used by this module is: {dt}")
        return dt