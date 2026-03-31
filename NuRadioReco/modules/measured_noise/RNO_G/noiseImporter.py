import numpy as np
import glob
import os
import collections
import time

from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units

import logging


class noiseImporter:
    """
    Imports recorded traces from RNO-G stations. Uses forced/software triggers.

    Adds measured noise to both readout channels and (optionally) trigger
    channel copies. The trigger copy injection handles the readout-to-trigger
    signal chain conversion automatically using the detector description.

    The trigger copy feature is needed for any simulation that evaluates a
    realistic hardware trigger (e.g., FLOWER board) on measured noise. The
    FLOWER board sees the signal through a different signal chain than the
    RADIANT readout (see arXiv:2411.12922, Sec. 3.2), so the noise must
    be transformed from the readout domain to the trigger domain before
    injection into trigger copies.
    """

    def begin(
            self, noise_folders=None, noise_files=None, file_pattern="*",
            match_station_id=False, station_ids=None,
            channel_mapping=None, scramble_noise_file_order=True,
            log_level=logging.NOTSET, random_seed=None, reader_kwargs={},
            inject_trigger_copies=False, trigger_channels=None,
            hardware_response_incorporator=None):
        """

        Parameters
        ----------
        noise_folders: str or list(str) or None
            Folder(s) containing noise file(s). Search in any subfolder as well.
            Either noise_folders or noise_files must be provided.

        noise_files: list(str) or None
            Explicit list of ROOT file paths or run directories to use.
            Skips recursive glob discovery. Takes precedence over noise_folders.

        file_pattern: str
            File pattern used to search for directories, (Default: "*", other examples might be "combined")

        match_station_id: bool
            If True, add only noise from stations with the same id. (Default: False)

        station_ids: list(int)
            Only add noise from those station ids. If None, use any station. (Default: None)

        channel_mapping: dict or None
            option relevant for MC studies of new station designs where we do not
            have forced triggers for. The channel_mapping dictionary maps the channel
            ids of the MC station to the channel ids of the noise data
            Default is None which is 1-to-1 mapping

        scramble_noise_file_order: bool
            If True, randomize the order of noise files before reading them. (Default: True)

        log_level: logging log level
            Override he log level to control verbosity. (Default: logging.NOTSET, ie follow general log level)

        random_seed: int
            Seed for the random number generator. (Default: None, no fixed seed).

        reader_kwargs: dict
            Optional arguements passed to readRNOGDataMattak

        inject_trigger_copies: bool
            If True, also inject noise into trigger channel copies with
            the readout-to-trigger transfer function applied. Required
            for realistic FLOWER trigger evaluation. (Default: False)

        trigger_channels: list(int) or None
            Channel IDs that have trigger copies (e.g., [0, 1, 2, 3] for
            the phased array). Required when inject_trigger_copies=True.
            (Default: None)

        hardware_response_incorporator: hardwareResponseIncorporator or None
            An initialized hardwareResponseIncorporator instance, used to
            compute the readout-to-trigger transfer function. Required
            when inject_trigger_copies=True. (Default: None)
        """

        self.logger = logging.getLogger('NuRadioReco.RNOG.noiseImporter')
        self.logger.setLevel(log_level)
        self.__random_gen = np.random.Generator(np.random.Philox(random_seed))

        self._match_station_id = match_station_id
        self.__station_ids = station_ids

        self.__channel_mapping = channel_mapping

        self._inject_trigger_copies = inject_trigger_copies
        self._trigger_channels = trigger_channels or []
        self._hw_resp = hardware_response_incorporator
        self._readout_to_trigger_transfer = {}

        if inject_trigger_copies:
            if not self._trigger_channels:
                raise ValueError(
                    "trigger_channels must be specified when "
                    "inject_trigger_copies=True")
            if self._hw_resp is None:
                raise ValueError(
                    "hardware_response_incorporator must be provided when "
                    "inject_trigger_copies=True")

        self.logger.info(f"\n\tMatch station id: {match_station_id}"
                    f"\n\tUse noise from only those stations: {station_ids}"
                    f"\n\tUse the following channel mapping: {channel_mapping}"
                    f"\n\tRandomize sequence of noise files: {scramble_noise_file_order}")

        if noise_files is not None:
            if isinstance(noise_files, str):
                noise_files = [noise_files]
            self.__noise_folders = np.array(noise_files)
        elif noise_folders is not None:
            if not isinstance(noise_folders, list):
                noise_folders = [noise_folders]

            discovered = []
            for noise_folder in noise_folders:
                if noise_folder == "":
                    continue
                discovered += glob.glob(f"{noise_folder}/**/{file_pattern}root", recursive=True)
            self.__noise_folders = np.unique([os.path.dirname(e) for e in discovered])
        else:
            raise ValueError("Either noise_folders or noise_files must be provided")

        self.logger.info(f"Found {len(self.__noise_folders)}")
        if not len(self.__noise_folders):
            self.logger.error("No folders found")
            raise FileNotFoundError("No folders found")

        if scramble_noise_file_order:
            self.__random_gen.shuffle(self.__noise_folders)

        self._noise_reader = readRNOGData()

        default_reader_kwargs = {
            "selectors": [lambda einfo: einfo.triggerType == "FORCE"],
            "select_runs": True, "max_trigger_rate": 2 * units.Hz,
            "run_types": ["physics"]
        }
        default_reader_kwargs.update(reader_kwargs)

        self._noise_reader.begin(self.__noise_folders, **default_reader_kwargs)

        # instead of reading all noise events into memory we only get certain information here and read all data in run()
        self.logger.info("Get event informations ...")
        t0 = time.time()
        noise_information = self._noise_reader.get_events_information(keys=["station"])
        self.logger.info(f"... of {len(noise_information)} (selected) events in {time.time() - t0:.2f}s")

        self.__event_index_list = np.array(list(noise_information.keys()))
        self.__station_id_list = np.array([ele["station"] for ele in noise_information.values()])

        self._n_use_event = collections.defaultdict(int)

    @property
    def n_events_available(self):
        """Number of noise events in the pool."""
        return len(self.__event_index_list)

    def __get_noise_channel(self, channel_id):
        if self.__channel_mapping is None:
            return channel_id
        else:
            return self.__channel_mapping[channel_id]


    def __draw_noise_event(self, mask):
        """
        reader.get_event_by_index can return None when, e.g., the trigger time is inf or the sampling rate 0.
        Hence, try again if that happens (should only occur rearly).

        Parameters
        ----------
        mask: np.array(bool)
            Mask of which noise events are allowed (e.g. because of matching station ids, ...)

        Returns
        -------
        noise_event: NuRadioReco.framework.event
            A event containing noise traces

        i_noise: int
            The index of the drawn event
        """
        tries = 0
        while tries < 100:
            # int(..) necessary because pyroot can not handle np.int64
            i_noise = int(self.__random_gen.choice(self.__event_index_list[mask]))
            noise_event = self._noise_reader.get_event_by_index(i_noise)
            tries += 1
            if noise_event is not None:
                break

        if noise_event is None:
            err = "Could not draw a random station which is not None after 100 tries. Stop."
            self.logger.error(err)
            raise ValueError(err)

        self._n_use_event[i_noise] += 1
        return noise_event, i_noise


    def _draw_and_cache_noise(self, station):
        """Draw a noise event and cache it for two-stage injection."""
        if self._match_station_id:
            station_mask = self.__station_id_list == station.get_id()
            if not np.any(station_mask):
                raise ValueError(f"No station with id {station.get_id()} in noise data.")
        else:
            station_mask = np.full_like(self.__event_index_list, True)

        noise_event, i_noise = self.__draw_noise_event(station_mask)
        station_id = noise_event.get_station_ids()[0]
        noise_station = noise_event.get_station(station_id)

        if self.__station_ids is not None and station_id not in self.__station_ids:
            raise ValueError(f"Station id {station_id} not in list: {self.__station_ids}")

        self.logger.debug("Selected noise event {} ({}, run {}, event {})".format(
            i_noise, noise_station.get_station_time(), noise_event.get_run_number(),
            noise_event.get_id()))

        self._cached_noise_station = noise_station
        return noise_station

    @register_run()
    def run(self, evt, station, det, trigger_copies_only=False):
        """Add measured noise to station channels.

        Parameters
        ----------
        evt, station, det : standard NuRadioReco objects
        trigger_copies_only : bool
            If True, only inject into trigger channel copies (for use
            at the internal simulation rate before trigger evaluation).
            If False, inject into readout channels (normal mode).
            When True, a new noise event is drawn and cached. When
            False with a cached event, the cached event is reused to
            ensure trigger and readout see the same noise realization.
        """
        if trigger_copies_only:
            noise_station = self._draw_and_cache_noise(station)
        elif hasattr(self, '_cached_noise_station') and self._cached_noise_station is not None:
            noise_station = self._cached_noise_station
            self._cached_noise_station = None
        else:
            noise_station = self._draw_and_cache_noise(station)

        for channel in station.iter_channels():
            channel_id = channel.get_id()
            noise_channel = noise_station.get_channel(self.__get_noise_channel(channel_id))
            noise_trace = noise_channel.get_trace()

            if not trigger_copies_only:
                trace = channel.get_trace()

                if len(trace) > 2048:
                    self.logger.warning("Simulated trace longer than 2048, trimming")
                    trace = trace[:2048]

                if len(trace) != len(noise_trace):
                    erg_msg = (f"Mismatch in trace length: Noise has {len(noise_trace)} "
                               f"and simulation has {len(trace)} samples")
                    self.logger.error(erg_msg)
                    raise ValueError(erg_msg)

                if channel.get_sampling_rate() != noise_channel.get_sampling_rate():
                    erg_msg = (f"Mismatch in sampling rate: Noise has "
                               f"{noise_channel.get_sampling_rate() / units.GHz} and "
                               f"simulation has {channel.get_sampling_rate() / units.GHz} GHz")
                    self.logger.error(erg_msg)
                    raise ValueError(erg_msg)

                trace = trace + noise_trace
                channel.set_trace(trace, channel.get_sampling_rate())

            if (trigger_copies_only
                    and self._inject_trigger_copies
                    and channel_id in self._trigger_channels
                    and channel.has_extra_trigger_channel()):
                trig_ch = channel.get_trigger_channel()
                trig_trace = trig_ch.get_trace()
                n_trig = len(trig_trace)
                trig_sr = trig_ch.get_sampling_rate()

                if trig_sr == 0:
                    trig_sr = channel.get_sampling_rate()
                    self.logger.debug(
                        f"ch{channel_id}: trigger copy sr=0, using channel sr={trig_sr/units.GHz:.1f} GHz")

                n_up = int(round(len(noise_trace) * trig_sr / noise_channel.get_sampling_rate()))
                noise_up = self._upsample(noise_trace, n_up)

                transfer = self._get_readout_to_trigger_transfer(
                    channel_id, n_up, det, station.get_id(), trig_sr)
                noise_fft = np.fft.rfft(noise_up)
                trig_noise = np.fft.irfft(noise_fft * transfer, n=n_up)

                # Noise covers first n_up samples; rest is signal-only from convolution
                trig_trace[:n_up] += trig_noise
                trig_ch.set_trace(trig_trace, trig_sr)

    @staticmethod
    def _upsample(trace, target_n_samples):
        """Upsample via FFT zero-padding above Nyquist."""
        n_orig = len(trace)
        spec = np.fft.rfft(trace)
        new_spec = np.zeros(target_n_samples // 2 + 1, dtype=complex)
        new_spec[:len(spec)] = spec
        return np.fft.irfft(new_spec, n=target_n_samples) * (target_n_samples / n_orig)

    def _get_readout_to_trigger_transfer(self, ch_id, n_samples, det, station_id, sampling_rate):
        """Compute and cache the readout-to-trigger transfer function.

        Returns trigger_response / readout_response in the frequency domain.
        Regularizes near-zero readout values to prevent division artifacts.
        """
        key = (ch_id, n_samples)
        if key not in self._readout_to_trigger_transfer:
            ff = np.fft.rfftfreq(n_samples, d=1.0 / sampling_rate)
            readout = self._hw_resp.get_filter(
                ff, station_id, ch_id, det,
                sim_to_data=True, is_trigger=False)
            trigger = self._hw_resp.get_filter(
                ff, station_id, ch_id, det,
                sim_to_data=True, is_trigger=True)
            readout_abs = np.abs(readout)
            max_r = np.max(readout_abs)
            safe_readout = np.where(
                readout_abs > 1e-3 * max_r, readout, max_r)
            self._readout_to_trigger_transfer[key] = trigger / safe_readout
        return self._readout_to_trigger_transfer[key]

    def end(self):
        self._noise_reader.end()
        n_use = np.array(list(self._n_use_event.values()))
        sort = np.flip(np.argsort(n_use))
        self.logger.info(
            "\n\tThe five most used noise events have been used: {}"
            .format(", ".join([str(ele) for ele in n_use[sort][:5]])))
