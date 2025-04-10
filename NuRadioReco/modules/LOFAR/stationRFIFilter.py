import logging

import numpy as np

from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities.signal_processing import half_hann_window
from NuRadioReco.modules.io.LOFAR._rawTBBio import MultiFile_Dal1
from NuRadioReco.modules.io.LOFAR.readLOFARData import tbbID_to_nrrID, nrrID_to_tbbID

from NuRadioReco.framework.parameters import stationParameters
from NuRadioReco.utilities import units

logger = logging.getLogger('NuRadioReco.LOFAR.stationRFIFilter')

def num_double_zeros(data, threshold=None, ave_shift=False):
    """if data is a numpy array, give number of points that have  zero preceded by a zero"""

    if ave_shift:
        data = data - np.average(data)

    if threshold is None:
        is_zero = data == 0
    else:
        is_zero = np.abs(data) < threshold

    bad = np.logical_and(is_zero[:-1], is_zero[1:])
    return np.sum(bad)


def median_sorted_by_power(psort):
    """
    This function takes a list `psort`, which is assumed to be a sorted list (of indices). This function returns a list
    beginning with the median (i.e. the middle element) and subsequently the elements right and left of it, in
    increasing distance from the median.

    Parameters
    ----------
    psort: 1D array_like
        The list of elements to reoder.

    Examples
    --------
    >>> median_sorted_by_power([1,2,3,4,5])
    [3, 4, 2, 5, 1]

    In the example above, the middle element is 3. Therefore, the resulting array starts with 3. From there the
    algorithm takes the first element right (4) and left (2) of it. Then it takes the second element right (5) and
    left (1). This continues until all elements have been added to the output list.
    """
    lpsort = len(psort)
    if lpsort % 2 == 0:
        index = int(lpsort / 2) - 1
    else:
        index = int(lpsort / 2)

    modifier = 0
    out_psort = []
    start_index = index
    for i in range(0, lpsort):
        out_psort.append(psort[index])
        if modifier == 0:
            modifier = 1
        elif modifier > 0:
            modifier = -modifier
        elif modifier < 0:
            modifier = -(modifier - 1)
        index = start_index + modifier

    return out_psort


def FindRFI_LOFAR(
        tbb_filename,
        metadata_dir,
        target_trace_length=65536,
        rfi_cleaning_trace_length=8192,
        flagged_antenna_ids=None,
        num_dbl_z=1000,
        pol=None
):
    """
    A code that basically reads given LOFAR TBB H5 file and returns an array of dirty channels.
    Some values are hard coded based on LOFAR Experience.

    Parameters
    ----------
    tbb_filename : list[Path-like str]
        A list of paths to the TBB file to be analysed.
    metadata_dir : Path-like str
        The path to the directory containing the LOFAR metadata (required to open TBB file correctly).
    target_trace_length : int
        Size of total block of trace to be evaluated for finding dirty rfi channels.
    rfi_cleaning_trace_length : int
        Size of one chunk of trace to be evaluated at a time for calculating spectrum from.
    flagged_antenna_ids : list[str] or set[str], default=[]
        List of antennas which are already flagged. These will not be considered for the RFI detection process.
    num_dbl_z : int, default=100
        The number of double zeros allowed in a block, if there are too many, then there could be data loss.

    Raises
    ------
    ValueError
        If the `target_trace_length` is not a multiple of the `rfi_cleaning_trace_length`.

    Returns
    -------
    output_dict : dict
        A dictionary with the following key-value pairs:

           * "avg_power_spectrum": array that contains the average of the magnitude for each frequency channel over all
             antennas.

           * "avg_antenna_power": array that contains the average power in frequency domain for each antenna.

           * "antenna_names": the antenna IDs of which the data was used.

           * "cleaned_power": array containing the power in each antenna after contaminated channels have been removed.
             Note that the result is multiplied by a factor of 2 to account for the negative frequencies.

           * "phase_stability": 2-dimensional array with the average phase for every frequency channel in each antenna.

           * "dirty_channels": array of indices indicating the channels that are contaminated with RFI.

           * "dirty_channels_block_size": the number of samples per block used to detect phase stability.
    """
    if flagged_antenna_ids is None:
        flagged_antenna_ids = []

    # Hardcoded values for LOFAR
    lower_frequency_bound = 0.0 * units.Hz
    upper_frequency_bound = 100e6 * units.Hz
    logger.info('Upper frequency bound = %1.4e' % upper_frequency_bound)

    # Some checks to make sure variables are correctly matched with each other
    if target_trace_length % rfi_cleaning_trace_length != 0:
        logger.error(
            "The target total block size has to be multiple of single RFI block size."
        )
        raise ValueError

    if (rfi_cleaning_trace_length < 4096) or (rfi_cleaning_trace_length > 16384):
        logger.warning(  # WHY 6 times this warning? No self.logger available here
            "RFI cleaning may not work optimally at block sizes < 4096 or > 16384; %s" % tbb_filename
        )

    # Open the TBB file and pass it to FindRFI()
    tbb_file = MultiFile_Dal1(tbb_filename, metadata_dir=metadata_dir)

    num_blocks = np.median(tbb_file.get_nominal_data_lengths() // rfi_cleaning_trace_length)  # should all be the same

    # FIXME: -- should be fixed, needs explicit testing -- what if one bad antenna in Station? Currently FindRFI crashes

    logger.info(f"Running find RFI with {num_blocks} blocks")

    ############

    initial_block = 0
    num_blocks = int(num_blocks)  # make sure the number of blocks in an integer (as its used as a shape parameter)
    max_blocks = num_blocks

    window_function = half_hann_window(rfi_cleaning_trace_length, 0.1)

    antenna_ids = tbb_file.get_antenna_names()
    antenna_ids = [id for id in antenna_ids if id not in flagged_antenna_ids]
    if pol is not None:
        antenna_ids = [id for id in antenna_ids if int(id) % 2 == pol] # do one polarization
    num_antennas = len(antenna_ids)
    #logger.info(antenna_ids)

    # step one: find which blocks are good, and find average power
    oneAnt_data = np.zeros(rfi_cleaning_trace_length, dtype=np.double)  # initialize at zero

    logger.info("finding good blocks")
    blocks_good = np.zeros((num_antennas, max_blocks), dtype=bool)
    num_good_blocks = np.zeros(num_antennas, dtype=int)
    average_power = np.zeros(num_antennas, dtype=np.double)
    for block_i in range(max_blocks):
        block = block_i + initial_block

        for ant_i in range(num_antennas):
            try:
                oneAnt_data[:] = tbb_file.get_data(
                    rfi_cleaning_trace_length * block, rfi_cleaning_trace_length, antenna_ID=antenna_ids[ant_i]
                )
            except IndexError:
                logger.warning('Could not read data for antenna %s block %d' % (antenna_ids[ant_i], block_i))
                # proceed with zeros in the block
                blocks_good[ant_i, block_i] = False
                continue
            if (
                    num_double_zeros(oneAnt_data) < num_dbl_z
            ):  # this antenna on this block is good
                blocks_good[ant_i, block_i] = True
                num_good_blocks[ant_i] += 1

                oneAnt_data *= window_function

                FFT_data = np.fft.fft(oneAnt_data)
                np.abs(FFT_data, out=FFT_data)
                magnitude = FFT_data
                magnitude *= magnitude
                average_power[ant_i] += np.real(np.sum(magnitude))

    average_power[num_good_blocks != 0] /= num_good_blocks[num_good_blocks != 0]

    # Now we try to find the best reference antenna, require that antenna allows for maximum number of good antennas,
    # and has **best** average received power
    allowed_num_antennas = np.empty(
        num_antennas, dtype=int
    )
    # If ant_i is chosen to be your reference antenna, then allowed_num_antennas[ ant_i ] is the number of
    # antennas with num_blocks good blocks
    for ant_i in range(num_antennas):  # fill allowed_num_antennas
        blocks_can_use = np.where(blocks_good[ant_i])[0]
        num_good_blocks_per_antenna = np.sum(blocks_good[:, blocks_can_use], axis=1)
        allowed_num_antennas[ant_i] = np.sum(num_good_blocks_per_antenna >= num_blocks)

    max_allowed_antennas = np.max(allowed_num_antennas)

    if max_allowed_antennas < 2:
        logger.error("ERROR: station", tbb_file.get_station_name(), "cannot find RFI")
        return

    # Pick a reference antenna that allows max number of antennas, and has most median amount of power
    can_be_ref_antenna = allowed_num_antennas == max_allowed_antennas

    sorted_by_power = np.argsort(average_power)
    mps = median_sorted_by_power(sorted_by_power)

    for ant_i in mps:
        if can_be_ref_antenna[ant_i]:
            ref_antenna = ant_i
            break

    logger.info("Taking channel %d as reference antenna" % ref_antenna)  # this will fail if ref_antenna not found

    # Define some helping variables
    good_blocks = np.where(blocks_good[ref_antenna])[0]

    num_good_blocks = np.sum(blocks_good[:, good_blocks], axis=1)
    antenna_is_good = num_good_blocks >= (num_blocks - 1)

    blocks_good[np.logical_not(antenna_is_good), :] = False

    # Process data
    num_processed_blocks = np.zeros(num_antennas, dtype=int)
    frequencies = np.fft.fftfreq(rfi_cleaning_trace_length, 1.0 / tbb_file.get_sample_frequency())
    frequencies *= units.Hz
    lower_frequency_index = np.searchsorted(
        frequencies[: int(len(frequencies) / 2)], lower_frequency_bound
    )
    upper_frequency_index = np.searchsorted(
        frequencies[: int(len(frequencies) / 2)], upper_frequency_bound
    )

    phase_mean = np.zeros(
        (num_antennas, upper_frequency_index - lower_frequency_index), dtype=complex
    )
    spectrum_mean = np.zeros(
        (num_antennas, upper_frequency_index - lower_frequency_index), dtype=np.double
    )

    data = np.empty((num_antennas, len(frequencies)), dtype=complex)
    temp_mag_spectrum = np.empty((num_antennas, len(frequencies)), dtype=np.double)
    temp_phase_spectrum = np.empty((num_antennas, len(frequencies)), dtype=complex)
    for block_i in good_blocks:
        logger.info("Doing block %d" % block_i)
        block = block_i + initial_block

        for ant_i in range(num_antennas):
            if (
                    num_processed_blocks[ant_i] == num_blocks
                    or not blocks_good[ant_i, block_i]
            ):
                continue
            oneAnt_data[:] = tbb_file.get_data(
                rfi_cleaning_trace_length * block, rfi_cleaning_trace_length, antenna_ID=antenna_ids[ant_i]
            )

            # Window the data
            # Note: No hanning window if we want to measure power accurately from spectrum in the same units
            # as power from timeseries. Applying a window gives (at least) a scale factor difference!
            # But no window makes the cleaning less effective... :(
            oneAnt_data *= window_function
            data[ant_i] = np.fft.fft(oneAnt_data)

        np.abs(data, out=temp_mag_spectrum)
        temp_phase_spectrum[:] = data
        temp_phase_spectrum /= temp_mag_spectrum + 1.0e-15
        temp_phase_spectrum[:, :] /= temp_phase_spectrum[ref_antenna, :]

        temp_mag_spectrum *= temp_mag_spectrum

        for ant_i in range(num_antennas):
            if (
                    num_processed_blocks[ant_i] == num_blocks
                    or not blocks_good[ant_i, block_i]
            ):
                continue

            phase_mean[ant_i, :] += temp_phase_spectrum[ant_i][
                                    lower_frequency_index:upper_frequency_index
                                    ]
            spectrum_mean[ant_i, :] += temp_mag_spectrum[ant_i][
                                       lower_frequency_index:upper_frequency_index
                                       ]

            num_processed_blocks[ant_i] += 1

        if np.min(num_processed_blocks[antenna_is_good]) == num_blocks:
            break

    logger.info(f"{num_blocks} analyzed blocks, {np.sum(antenna_is_good)} analyzed antennas out of {len(antenna_is_good)}")

    # Get only good antennas
    antenna_is_good[
        ref_antenna
    ] = False  # we don't want to analyze the phase stability of the reference antenna

    # Get mean and phase stability
    spectrum_mean /= num_blocks

    phase_stability = np.abs(phase_mean)
    phase_stability *= -1.0 / num_blocks
    phase_stability += 1.0

    # Get median of stability by channel, across each antenna
    median_phase_spread_byChannel = np.median(phase_stability[antenna_is_good], axis=0)
    # Get median across all channels
    median_spread = np.median(median_phase_spread_byChannel)
    # Create a noise cutoff
    sorted_phase_spreads = np.sort(median_phase_spread_byChannel)
    N = len(median_phase_spread_byChannel)
    noise = sorted_phase_spreads[int(N * 0.95)] - sorted_phase_spreads[int(N / 2)]
    # Get channels contaminated by RFI, where phase stability is smaller than noise
    dirty_channels = np.where(
        median_phase_spread_byChannel < (median_spread - 3 * noise)
    )[0]

    # Extend dirty channels by some size, in order to account for shoulders
    extend_dirty_channels = np.zeros(N, dtype=bool)
    half_flagwidth = int(rfi_cleaning_trace_length / 8192)
    for i in dirty_channels:
        flag_min = i - half_flagwidth
        flag_max = i + half_flagwidth
        if flag_min < 0:
            flag_min = 0
        if flag_max >= N:
            flag_max = N - 1
        extend_dirty_channels[flag_min:flag_max] = True

    dirty_channels = np.where(extend_dirty_channels)

    antenna_is_good[ref_antenna] = True  # cause'.... ya know.... it is

    ave_spectrum_magnitude = spectrum_mean

    tbb_file.close_file()

    # Calculate the required output variables
    avg_power_spectrum = np.sum(ave_spectrum_magnitude, axis=0) / ave_spectrum_magnitude.shape[0]
    avg_antenna_power = np.sum(ave_spectrum_magnitude, axis=1) / ave_spectrum_magnitude.shape[1]

    antenna_names = antenna_ids

    cleaned_spectrum = np.array(ave_spectrum_magnitude)
    cleaned_spectrum[:, dirty_channels] = 0.0
    cleaned_power = 2 * np.sum(cleaned_spectrum, axis=1)

    dirty_channels += lower_frequency_index
    dirty_channels = dirty_channels[0]
    multiplied_channels = []
    multiplied_blocks = target_trace_length // rfi_cleaning_trace_length
    for ch in dirty_channels:  # TODO: could be done more efficiently...?
        this_channel_list = np.arange(multiplied_blocks * ch, multiplied_blocks * ch + multiplied_blocks, 1)
        this_channel_list = list(this_channel_list)
        multiplied_channels.extend(this_channel_list)

    dirty_channels = np.sort(np.array(multiplied_channels))
    dirty_channels_block_size = target_trace_length

    # Use dictionary to avoid indexing mistakes
    output_dict = {
        "avg_power_spectrum": avg_power_spectrum,
        "avg_antenna_power": avg_antenna_power,
        "antenna_names": antenna_names,
        "cleaned_power": cleaned_power,
        "phase_stability": phase_stability,
        "dirty_channels": dirty_channels,
        "dirty_channels_block_size": dirty_channels_block_size,
    }

    return output_dict


# TODO: make stationRFIFilter take keyword to only process certain stations -> already implemented in reader?
class stationRFIFilter:
    """
    Remove the RFI from all stations in an Event, by using the phase-variance method described in
    the notes section. This algorithm returns the frequency channels which are contaminated, which are
    subsequently put to zero in the traces.

    **Note**: currently the class uses hardcoded values for LOFAR, this needs to be improved later.

    Notes
    -----
    The algorithm compares the phase stability of each frequency channel between a reference antenna and every
    other antenna in the station. If the phase is stable, this indicates a constant source contaminating the data.
    More information can be found in Section 3.2.2 of `this paper <https://arxiv.org/pdf/1311.1399.pdf>`_ .
    """

    def __init__(self):
        self.logger = logger  # logging.getLogger('NuRadioReco.channelRFIFilter')

        self.__rfi_trace_length = None
        self.__station_list = None
        self.__metadata_dir = None
        self.__median_spectrum = None
        self.__do_polarizations_apart = None

    @property
    def station_list(self):
        return self.__station_list

    @station_list.setter
    def station_list(self, new_list):
        self.__station_list = new_list

    @property
    def metadata_dir(self):
        return self.__metadata_dir

    @metadata_dir.setter
    def metadata_dir(self, new_dir):
        self.__metadata_dir = new_dir

    def begin(self, rfi_cleaning_trace_length=65536, reader=None, do_polarizations_apart=False,
              logger_level=logging.NOTSET):
        """
        Set the variables used for RFI detection. The `reader` object can be used to retrieve the filenames associated
        with the loaded stations, as well as the metadata directory.

        Parameters
        ----------
        rfi_cleaning_trace_length : int
            The number of samples to use per block to construct the frequency spectrum.
        reader : readLOFARData object, default=None
            If provided, the reader will be used to set the metadata directory and find the TBB files paths.
        do_polarizations_apart : bool, default=False
            If True, the X and Y polarisations will be processed separately.
        logger_level : int, default=logging.NOTSET
            Use this parameter to override the logging level for this module.

        Notes
        -----
        If no reader is provided here, the user should set the `self.station_list` and `self.metadata_dir` variables
        manually before attempting to execute the `stationRFIFilter.run()` function.
        """
        self.__rfi_trace_length = rfi_cleaning_trace_length
        self.__do_polarizations_apart = do_polarizations_apart
        if reader is not None:
            self.station_list = reader.get_stations()
            self.metadata_dir = reader.meta_dir

        self.logger.setLevel(logger_level)

    @register_run()
    def run(self, event):
        """
        Run the filter on the `event`. The method currently uses :py:func:`FindRFI_LOFAR` to find the contaminated
        channels and then puts the corresponding frequency bands to zero in every channel (in place).

        Parameters
        ----------
        event : Event object
            The event on which to run the filter.
        """
        stations_dict = self.station_list
        self.__median_spectrum = {}

        for station in event.get_stations():
            station_name = f'CS{station.get_id():03}'
            station_files = stations_dict[station_name]['files']
            antenna_set = stations_dict[station_name]['metadata'][1]
            flagged_channel_ids: dict[int, list[str]] = station.get_parameter(stationParameters.flagged_channels)  # this is a defaultdict

            # Find the length of a trace in the station (assume all channels have been loaded with same length)
            station_trace_length = station.get_channel(station.get_channel_ids()[0]).get_number_of_samples()

            # Do some checks
            if len(station_files) < 1:
                self.logger.warning(f'No files in reader dict for station {station_name}, skipping...')
                continue
            if (station_trace_length < self.__rfi_trace_length) or \
                    (station_trace_length % self.__rfi_trace_length != 0):
                self.logger.error(f'Station trace length ({station_trace_length}) has to be greater than RFI trace '
                                  f'length ({self.__rfi_trace_length}) as well as a multiple of it.')
                raise ValueError

            flagged_tbb_channel_ids = set()
            for ind in flagged_channel_ids:
                flagged_tbb_channel_ids.add(nrrID_to_tbbID(ind))  # in rawTBBio, antenna IDs are str

            if not self.__do_polarizations_apart:
                packet = FindRFI_LOFAR(station_files,
                                    self.metadata_dir,
                                    station_trace_length,
                                    self.__rfi_trace_length,
                                    flagged_antenna_ids=flagged_tbb_channel_ids
                                    )
                dirty_channels = packet['dirty_channels']

            else:
                packet0 = FindRFI_LOFAR(station_files,
                                    self.metadata_dir,
                                    station_trace_length,
                                    self.__rfi_trace_length,
                                    flagged_antenna_ids=flagged_tbb_channel_ids, pol=0
                                    )

                packet1 = FindRFI_LOFAR(station_files,
                                    self.metadata_dir,
                                    station_trace_length,
                                    self.__rfi_trace_length,
                                    flagged_antenna_ids=flagged_tbb_channel_ids, pol=1
                                    )


                # Extract the necessary information from FindRFI
                dirty_channels_0 = packet0['dirty_channels']
                dirty_channels_1 = packet1['dirty_channels']

                dirty_channels = list(set(dirty_channels_0) | set(dirty_channels_1))
                packet = packet0
                packet['antenna_names'].extend(packet1['antenna_names'])
                packet['cleaned_power'] = np.concatenate( (packet['cleaned_power'], packet1['cleaned_power'])  )


            station.set_parameter(stationParameters.dirty_fft_channels, dirty_channels)

            # implement outlier detection in cleaned power
            antenna_ids = np.array(packet['antenna_names'])
            cleaned_power = packet['cleaned_power']
            median_dipole_power = np.median(cleaned_power)
            bad_dipole_indices = np.where(
                np.logical_or(
                    cleaned_power < 0.5 * median_dipole_power, cleaned_power > 2.0 * median_dipole_power
                )
            )[0]
            # which dipole ids are these
            self.logger.info(
                f'There are {len(bad_dipole_indices)} outliers in cleaned power \n'
                f'Dipole ids: {antenna_ids[bad_dipole_indices]}'
            )

            # Construct flagged set of NRR channel IDs
            channel_ids_to_remove = set()
            for ind in antenna_ids[bad_dipole_indices]:
                # convert TBB IDs to nrr IDs
                nrr_id = tbbID_to_nrrID(ind, antenna_set)

                # Ensure both channels from same antenna are removed
                group_id_to_remove = station.get_channel(int(nrr_id)).get_group_id()
                for channel_id in station.iter_channel_group(group_id_to_remove):
                    channel_ids_to_remove.add(channel_id.get_id())
            self.logger.info(
                f'Removing the following channels from station {station_name}: \n'
                f'{channel_ids_to_remove}'
            )

            # Remove bad antennas (= both channels!) from station, if it exists
            for nrr_id in channel_ids_to_remove:
                station.remove_channel(nrr_id)

            for channel_ind in channel_ids_to_remove:
                flagged_channel_ids[channel_ind].append('rfi_outliers_cleaned_power')

            station.set_parameter(stationParameters.flagged_channels, flagged_channel_ids)

            # Set spectral amplitude to zero for channels with RFI
            spectra_before_flag = []
            for channel in station.iter_channels():
                trace_fft = channel.get_frequency_spectrum()
                spectra_before_flag.append(np.copy(trace_fft))
                sample_rate = channel.get_sampling_rate()

                # Reject DC and first harmonic
                trace_fft[0] *= 0.0
                trace_fft[1] *= 0.0

                # Remove dirty channels
                trace_fft[dirty_channels] *= 0.0

                channel.set_frequency_spectrum(trace_fft, sample_rate)
            self.__median_spectrum[station.get_id()] = np.median(np.abs(np.asarray(spectra_before_flag)), axis=0)

    def end(self, event=None):
        if event is not None:
            for station in event.get_stations():
                self.plot_median_freq_spectrum(event, station, rfi_cleaned=False, flagging=True)
                self.plot_median_freq_spectrum(event, station, rfi_cleaned=True, flagging=False)

    def plot_median_freq_spectrum(self, event, station, rfi_cleaned: bool = False, flagging: bool = False):
        import matplotlib.pyplot as plt
        if flagging and rfi_cleaned:
            logger.warning("plot_median_freq_spectrum flagging the rfi_cleaned channels in a clean trace is weird, but ok")

        station_name = f'CS{station.get_id():03d}'

        if rfi_cleaned:
            # median spectrum from channels in the station. Since this function is expected to run in the .end() after .run(), the traces there are cleaned
            spectra = []
            for channel in station.iter_channels():
                spectrum = channel.get_frequency_spectrum()
                spectra.append(np.abs(spectrum))
            median_spectrum = np.median(np.array(spectra), axis=0)
            plot_filename = f"{event.get_id()}-{station_name}-rfi_cleaning_flags.pdf"
        else:
            # pre rfi cleaned spectrum stored
            median_spectrum = self.__median_spectrum[station.get_id()]
            plot_filename = f"{event.get_id()}-{station_name}-median_spectrum_after_rfi_cleaning.pdf"

        fig = plt.figure()
        ax = fig.add_subplot()
        log_median_spectrum = np.log10(median_spectrum)
        channel = station.get_channel(station.get_channel_ids()[0])
        freq_MHz = channel.get_frequencies() / units.MHz
        ax.plot(freq_MHz, log_median_spectrum,zorder=1)
        if flagging:
            dirty_channels = station[stationParameters.dirty_fft_channels]
            ax.scatter(freq_MHz[dirty_channels], log_median_spectrum[dirty_channels], marker="x", color="red", zorder=2)
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("Log-Spectral Power [ADU]")
        ax.set_title(f"{station_name} Median frequency spectrum")
        import os 
        os.makedirs(str(event.get_id()), exist_ok=True)
        plt.savefig(f"{event.get_id()}/" + plot_filename)
