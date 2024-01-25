import numpy as np
import logging
import glob

from NuRadioReco.modules.base.module import register_run
import NuRadioReco.modules.io.NuRadioRecoio
from NuRadioReco.utilities import units


class channelMeasuredNoiseAdder:
    """
    Module that adds measured noise to channel traces
    It does so by reading in a set of .nur files, randomly selecting a forced trigger event
    and adding the noise from it to the channel waveforms.
    The waveforms from the channels in the noise files need to be at least as long as the
    waveforms to which the noise is added, so it is recommended to cut them to the right size
    first, for example using the channelLengthAdjuster.
    """
    def __init__(self):
        self.__filenames = None
        self.__io = None
        self.__random_state = None
        self.__max_iterations = None
        self.logger = logging.getLogger('NuRadioReco.channelMeasuredNoiseAdder')
        self.__noise_data = None


    def begin(self, filenames=None, folder=None, file_pattern="*", 
              random_seed=None, max_iterations=100, debug=False, 
              draw_noise_statistics=False, channel_mapping=None, log_level=logging.WARNING, 
              restrict_station_id=True, station_id=None, allow_noise_resampling=False, 
              baseline_substraction=True, allowed_triggers=["FORCE"]):
        """
        Set up module parameters

        Parameters
        ----------
        filenames: list of strings
            List of .nur files containing the measured noise. If None, look for .nur files in "folder".
            (Default: None)
            
        folder: str
            Only used when "filenames" is None. Directory to search for .nur files matching the "file_pattern"
            including subdirectories. (Default: None)
            
        file_pattern: str
            Use ``glob.glob(f"{folder}/**/{file_pattern}.nur", recursive=True)`` to search for files. (Default: "*")
        
        random_seed: int, default: None
            Seed for the random number generator. By default, no seed is set.
        
        max_iterations: int, default: 100
            The module will pick a random event from the noise files, until a suitable event is found
            or until the number of iterations exceeds max_iterations. In that case, an error is thrown.
        
        debug: bool, default: False
            Set True to get debug output
        
        draw_noise_statistics: boolean, default: False
            If true, the values of all samples is stored and a histogram with noise statistics is drawn
            be the end() method
        
        channel_mapping: dict or None
            option relevant for MC studies of new station designs where we do not
            have forced triggers for. The channel_mapping dictionary maps the channel
            ids of the MC station to the channel ids of the noise data
            Default is None which is 1-to-1 mapping
        
        log_level: loggging log level
            the log level, default logging.WARNING
        
        baseline_substraction: boolean
            Option to subtract mean from trace. Set mean_opt=False to remove mean subtraction from trace.
            
        restrict_station_id: bool
            Require the station in the noise event to be the same as in the simulated event. (Default: True)
            
        station_id: int
            If restrict_station_id is False specify the station id to be used for the noise. If None take first station
            in the noise event. (Default: None)
            
        allow_noise_resampling: bool
            Allow resampling the noise trace to match the simulated trace. (Default: False)
            
        allowed_triggers: list(str)
            List of trigger names which should be used, events with other triggers are not used. (Default: ["FORCE"])
        """
        
        self.logger.setLevel(log_level)
        
        self.__filenames = filenames
        if self.__filenames is None:
            if folder is None:
                err = "Both, \"filenames\" and \"folder\" are None, you have to specify at least one ..."
                self.logger.error(err)
                raise ValueError(err)
                
            self.__filenames = glob.glob(f"{folder}/**/{file_pattern}.nur", recursive=True)
            
        self.logger.info(f"Found {len(self.__filenames)} noise file(s) ...")
        
        self.__io = NuRadioReco.modules.io.NuRadioRecoio.NuRadioRecoio(self.__filenames)
        self.__random_state = np.random.Generator(np.random.Philox(random_seed))
        self.__max_iterations = max_iterations
        
        self.__channel_mapping = channel_mapping
        self.__baseline_substraction = baseline_substraction
        
        self.__restrict_station_id = restrict_station_id
        self.__noise_station_id = station_id
        self.__allow_noise_resampling = allow_noise_resampling
        
        self._allowed_triggers = allowed_triggers

        if debug:
            self.logger.setLevel(logging.DEBUG)
            self.logger.debug('Reading noise from {} files containing {} events'.format(len(filenames), self.__io.get_n_events()))
        
        if draw_noise_statistics:
            self.__noise_data = []
            

    def __get_noise_channel(self, channel_id):
        if self.__channel_mapping is None:
            return channel_id
        else:
            return self.__channel_mapping[channel_id]


    @register_run()
    def run(self, event, station, det):
        """
        Add measured noise to station channels

        Parameters
        ----------
        event: event object
        station: station object
        det: detector description
        """
        noise_station = None
        
        for _ in range(self.__max_iterations):
            noise_station = self.get_noise_station(station)
            # Get random station from noise file. If we got a suitable station, we continue,
            # otherwise we try again
            if noise_station is not None:
                break
        
        # To avoid infinite loops, if no suitable noise station was found after a number of iterations we raise an error
        if noise_station is None:
            raise ValueError('Could not find suitable noise event in noise files after {} iterations'.format(self.__max_iterations))
        
        for channel in station.iter_channels():
            noise_channel = noise_station.get_channel(self.__get_noise_channel(channel.get_id()))
            channel_trace = channel.get_trace()
            
            # if resampling is not desired no channel with wrong sampling rate is selected in get_noise_station()
            resampled = False
            if noise_channel.get_sampling_rate() != channel.get_sampling_rate():
                noise_channel.resample(channel.get_sampling_rate())
                resampled = True
            
            noise_trace = noise_channel.get_trace()
            
            if self.__baseline_substraction:
                mean = noise_trace.mean()
                std = noise_trace.std()
                if mean > 0.05 * std:
                    self.logger.warning((
                        "The noise trace has an offset/baseline of {:.3f}mV which is more than 5% of the STD of {:.3f}mV. "
                        "The module corrects for the offset but it might points to an error in the FPN subtraction.").format(mean / units.mV, std / units.mV))
                               
                noise_trace -= mean
                
            if len(channel_trace) > len(noise_trace):
                err = "{} is shorter than simulated trace. Stop".format("Resampled noise trace" if resampled else "Noise trace")
                self.logger.error(err)
                raise ValueError(err)
            elif len(channel_trace) < len(noise_trace):
                self.logger.warn("Noise trace has more samples than the simulated one, clip noise trace at the end ...")
                noise_trace = noise_trace[:channel.get_number_of_samples()]  # if to long, clip the end
            else:
                pass

            channel_trace += noise_trace

            # if draw_noise_statistics == True
            if self.__noise_data is not None:
                self.__noise_data.append(noise_trace)


    def get_noise_station(self, station):
        """
        Returns a random station from the noise files that can be used as a noise sample.
        The function selects a random event from the noise files and checks if it is suitable.
        If it is, the station is returned, otherwise None is returned. The event is suitable if it
        fulfills these criteria:
    
        * It contains a station with the same station ID as the one to which the noise shall be added
        * The station does not have a trigger that has triggered.
        * The every channel in the station to which the noise shall be added is also present in the station

        Parameters
        ----------
        station: Station class
            The station to which the noise shall be added
        """
        event_i = self.__random_state.integers(0, self.__io.get_n_events())
        noise_event = self.__io.get_event_i(event_i)
        
        if self.__restrict_station_id and station.get_id() not in noise_event.get_station_ids():
            self.logger.debug('Event {}: No station with ID {} found.'.format(event_i, station.get_id()))
            return None
        
        if self.__restrict_station_id:
            noise_station = noise_event.get_station(station.get_id())
        else:
            # If __noise_station_id == None get first station stored in event
            noise_station = noise_event.get_station(self.__noise_station_id)

        for trigger_name in noise_station.get_triggers():
            if trigger_name in self._allowed_triggers:
                continue
            
            trigger = noise_station.get_trigger(trigger_name)
            if trigger.has_triggered():
                self.logger.debug(f'Noise station has triggered ({trigger_name}), reject noise event.')
                return None
        
        for channel_id in station.get_channel_ids():
            noise_channel_id = self.__get_noise_channel(channel_id)
            if noise_channel_id not in noise_station.get_channel_ids():
                if channel_id == noise_channel_id:
                    self.logger.debug('Event {}: Requested channel {} not found'.format(event_i, noise_channel_id))
                return None
            
            noise_channel = noise_station.get_channel(noise_channel_id)
            channel = station.get_channel(channel_id)
            
            if channel.get_sampling_rate() != noise_channel.get_sampling_rate() and not self.__allow_noise_resampling:
                self.logger.debug('Event {}, Channel {}: Different sampling rates, reject noise event.'
                                  .format(event_i, noise_channel_id))
                return None   
            
            if (noise_channel.get_number_of_samples() / noise_channel.get_sampling_rate() < 
                channel.get_number_of_samples() / channel.get_sampling_rate()):
                # Just add debug message...
                self.logger.debug('Event {}, Channel {}: Different sampling rate / trace lenght ratios'
                                  .format(event_i, noise_channel_id))        
        
        return noise_station


    def end(self):
        """
        End method. Draws a histogram of the noise statistics and fits a
        Gaussian distribution to it.
        """
        if self.__noise_data is not None:
            import matplotlib.pyplot as plt
            plt.close('all')
            
            noise_entries = np.array(self.__noise_data)
            noise_bins = np.arange(-150, 150, 5.) * units.mV
            noise_entries = noise_entries.flatten()
            mean = noise_entries.mean()
            sigma = noise_entries.std()
        
            fig1, ax1_1 = plt.subplots()
            n, bins, pathes = ax1_1.hist(noise_entries / units.mV, bins=noise_bins / units.mV)
            ax1_1.plot(noise_bins / units.mV, np.max(n) * np.exp(-0.5 * (noise_bins - mean) ** 2 / sigma ** 2))
            ax1_1.grid()
            ax1_1.set_xlabel('sample value [mV]')
            ax1_1.set_ylabel('entries')
            plt.show()
