from NuRadioReco.modules.base.module import register_run
from NuRadioReco.framework.trigger import SimpleThresholdTrigger

import logging
import numpy as np
import time
import torch


class triggerSimulator:
    """
    Calculates a trigger using a convolutional neutral network on a small chunk of data
    """

    def __init__(self):
        self.__t = 0
        self.logger = logging.getLogger("CNNTriggerSimulator")
        self.logger.setLevel(logging.WARNING)

        self.__model = None
        self._device = None

    def begin(self, model, wvf_length, device=None, log_level=logging.WARNING):
        """
        Parameters
        ----------
        model: torch.nn.Module
            initialized Pytorch CNN model that takes N-input waveforms and outputs a single output value
            for a given waveform
        wvf_length: int
            number of bins that the CNN expects as an input
        device: torch.device (optional)
            hardware indicator to run the model on. If not set, will use whatever is available
        log_level: logging level (optional)
            level of the output to print
        """
        self.logger.setLevel(log_level)

        self._model = model
        self._device = device

        self._wvf_length = int(wvf_length)
        if self._wvf_length != wvf_length:
            msg = f"The waveform length must be an int, was given {wvf_length}"
            self.logger.error(msg)
            raise TypeError(msg)

        # Find if a GPU is found by the PyTorch
        if self._device is None:
            is_cuda = torch.cuda.is_available()
            self.logger.debug(f"CUDA is available: {('no', 'yes')[is_cuda]}")
            if is_cuda:
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")

        # Put the model on the CPU/GPU depending on what is available
        self._model.to(self._device)
        self._model.eval()

    @register_run()
    def run(
        self,
        evt,
        station,
        det,
        threshold,
        triggered_channels,
        vrms_per_channel,
        search_bin_location,
        trigger_time=None,
        pre_trigger_name=None,
        trigger_name="default_cnn_trigger",
    ):
        """
        Applies the CNN to the waveforms and calculates if trigger is satisfied

        Parameters
        ----------
        evt: None
            argument needs to be included for the `run` decorator, but is not used
        station: Station
            station to run the module on
        det: Detector
            detector instance (to follow normal module format, not used)
        threshold: float
            threshold value for the CNN output values
        triggered_channels: list of ints
            list of channel ids that specify which waveforms will be given to the CNN
        vrms_per_channel: list of floats
            rms voltage values that are used to normalize the waveforms into SNR amplitudes
        search_bin_location: float
            location of where the `trigger_time` will be placed in the waveform that is
            given to the CNN. Given as a fraction into the waveform. i.e. 0.5 is in the center
        trigger_time: float (conditionally optional)
            time `since the beginning of the waveform` indicating where to cut the waveform if the
            waveform length is lonver than the maximum waveform size. Must either set this option
            or set `trigger_name`
        pre_trigger_name: str (conditionally optional)
            name of the trigger to use as the `trigger_time`. If the given trigger has not triggered, this
            trigger will automatically fail. If `trigger_time` is not set, then a trigger name must be supplied
        trigger_name: string (optional)
            a unique name of this particular trigger
        """

        t_profile = time.time()

        if triggered_channels is None or not len(triggered_channels):
            msg = f"[{trigger_name}] Expected to get a list of channels for triggered_channels, instead got {triggered_channels}"
            self.logger.error(msg)
            raise TypeError(msg)

        if (trigger_time is None and pre_trigger_name is None) or (trigger_time is not None and pre_trigger_name is not None):
            msg = f"[{trigger_name}] Must set exactly one of options `trigger_time` and `pre_trigger_name`. Given values {trigger_time} and {pre_trigger_name}"
            self.logger.error(msg)
            raise ValueError(msg)

        output_trigger = SimpleThresholdTrigger(trigger_name, threshold, triggered_channels)
        if trigger_name is not None:
            input_trigger = station.get_trigger(pre_trigger_name)

            self.logger.debug(f'[{trigger_name}] Will use "{pre_trigger_name}" as a pre-trigger')

            # Quick check if can just skip this event
            if not input_trigger.has_triggered():
                output_trigger.set_triggered(False)
                self.logger.debug(f'[{trigger_name}] Pre-trigger "{pre_trigger_name}" did not trigger, automatically failing')
                station.set_trigger(output_trigger)
                self.__t += time.time() - t_profile
                return

            search_time = input_trigger.get_trigger_time()
        else:
            search_time = trigger_time

        self.logger.debug(f"[{trigger_name}] Running CNN on {len(triggered_channels)} channels {triggered_channels}")

        prediction_traces = None
        search_bin = None
        cut_low_bin = None
        cut_high_bin = None

        # Iterate through the channels, get the waveforms, scale them into units of SNR
        for channel in station.iter_channels():
            channel_id = channel.get_id()
            if channel_id not in triggered_channels:
                continue

            vrms = vrms_per_channel[station.get_id()][channel_id]

            # First time through the loop, calculate the trigger bins and create the waveform container
            if prediction_traces is None:
                ipred = 0
                prediction_traces = np.zeros((len(triggered_channels), self._wvf_length))

                # Find which bin is the trigger-bin and which bins to keep
                dt = 1 / channel.get_sampling_rate()
                start_time = channel.get_times()[0]
                search_bin = int((search_time - start_time) / dt)
                cut_low_bin = max(0, int(search_bin - self._wvf_length * search_bin_location))
                cut_high_bin = cut_low_bin + self._wvf_length

                self.logger.debug(
                    f"[{trigger_name}] Trigger will be searched for in bin {search_bin} ({search_time - start_time:0.2f} ns into wvf)"
                )
                self.logger.debug(f"[{trigger_name}] Will give bins {cut_low_bin} to {cut_high_bin} to the CNN")

            if cut_high_bin > len(channel.get_trace()):
                msg = f"[{trigger_name}] Need bins up to {cut_high_bin}, but waveform only has {len(channel.get_trace())} bins."
                cut_high_bin = len(channel.get_trace()) - 1
                cut_low_bin = cut_high_bin - self._wvf_length
                msg += f" Instead using bins {cut_low_bin} to {cut_high_bin}!"
                self.logger.warning(msg)

            subset = channel.get_trace()[cut_low_bin:cut_high_bin]
            if len(subset) != self._wvf_length:
                msg = f"[{trigger_name}] selected bins from channel {channel_id} are only {len(subset)} long but should be {self._wvf_length}"
                self.logger.error(msg)
                raise Exception(msg)

            prediction_traces[ipred] = subset / vrms
            ipred += 1

        if ipred != len(triggered_channels):
            msg = f"[{trigger_name}] The specified channels to trigger on ({triggered_channels}) where not in this event, only found {ipred} of these. Something has likely gone wrong!"
            self.logger.warning(msg)

        if search_bin is None:
            msg = f"[{trigger_name}] Trigger times were not properly determined in this event. Requested channels: {triggered_channels}"
            self.logger.error(msg)

        # Convert to pytorch data class and manipulate shape
        torch_tensor = torch.Tensor(prediction_traces).unsqueeze(0)
        assert torch_tensor.ndim == 3  # Ensure a 3D tensor
        assert torch_tensor.shape[0] == 1  # First dim is the unsqueezed part
        assert torch_tensor.shape[1] == len(triggered_channels)  # The N channels are the features
        assert torch_tensor.shape[2] == len(prediction_traces[0])  # Waveform length

        with torch.no_grad():
            yhat = self._model(torch_tensor.to(self._device).float()).squeeze().cpu()
            has_triggered = bool(yhat > threshold)
            self.logger.debug(f"[{trigger_name}] CNN score {yhat}, triggered: {has_triggered}")

        output_trigger.set_triggered(has_triggered)

        if has_triggered:
            output_trigger.set_trigger_time(search_time)
            
        station.set_trigger(output_trigger)

        self.__t += time.time() - t_profile

    def end(self):
        from datetime import timedelta

        self.logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        self.logger.info(f"CNN trigger: Total time used by this module is: {dt}")
        return dt
