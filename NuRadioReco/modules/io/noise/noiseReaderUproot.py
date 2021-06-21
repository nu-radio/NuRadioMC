import logging
import uproot
import numpy as np
from scipy import interpolate
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
logger = logging.getLogger('noiseImporter')

# layout to read the ARIANNA data for unknown formatting uproot does not guess right
ARIANNA_uproot_interpretation = {
        # TTimeStamp objects in root are 2 ints
        # * first is time in sec since 01/01/1970,
        # * second one is nanoseconds
        # will return a jagged array of shape n_events x [t_s, t_ns]
        "time": uproot.interpretation.jagged.AsJagged(
            uproot.interpretation.numerical.AsDtype('>i4'), header_bytes=6),
        # Interpretation of the trigger mask.
        # - Bit 0: Thermal trigger
        # - Bit 1: Forced trigger
        # - Bit 2: External trigger (not used with current DAQ)
        # - Bit 3: L1 Trigger satisfied: the L1 trigger cuts away events with
        #          a large fraction of power in a single frequency.
        #          true = event PASSES, false = event would be cut by this L1
        #          (may be in data via scaledown)
        # - Bit 4: 0 <not used>
        # - Bit 5: whether event is written thanks to L1 scaledown
        # - Bit 6: whether throwing away events based on L1 triggers
        # - Bit 7: flag events that took too long to get data from dCards to MB
        # NB: No idea why the number of header_bytes in the root files is so odd.
        "trigger": uproot.interpretation.jagged.AsJagged(
            uproot.interpretation.numerical.AsDtype("uint8"), header_bytes=7)
        }

# arianna tigger map
ARIANNA_TRIGGER = {
        "thermal" : 2**0,
        "forced"  : 2**1,
        "external": 2**2,
        "l1":       2**3,
        # not_used: 2**4,
        "l1_scaledown":     2**5,
        "l1_enabled" :      2**6,
        "exceeding_buffer": 2**7
        }

def find_stops(stop_block_array, first_only=False):
    """
    find the stop bits set in an array of stop blocks
    implementation matches the one ARIANNA ROOT software
    
    Paramteters
    -----------
    stop_block_array: array of 8 bit blocks where (typically 1 or 2) stop bits are nonzero

    Returns
    -------
    array of found stop positions
    """

    # mask to find set bits within a block
    bitcomparison_mask = 2**np.arange(8)


    # stop block consists of 8bit blocks
    # find blocks with stop bit set
    found_stops = []
    stop_set = np.flatnonzero(stop_block_array)
    # loop over blocks with a stop set
    for block_i in stop_set:
        bits_set = np.flatnonzero(stop_block_array[block_i]&bitcomparison_mask)
        for bit_i in bits_set:
            found_stops.append((block_i+1)*8 - (bit_i+1))
            if first_only:
                return found_stops[0]
    return np.array(found_stops)

find_stops_vectorized = np.vectorize(find_stops)

class ARIANNADataReader:
    """
    Imports recorded noise from ARIANNA station.
    The recorded noise needs to match the station geometry and sampling
    as chosen with channelResampler and channelLengthAdjuster

    For different stations, new noise files need to be used.
    Collect forced triggers from any type of station to use for analysis.
    A seizable fraction of data is recommended for accuracy.

    The noise will be random.
    This module therefore might produce non-reproducible results on a single event basis,
    if run several times.
    """

    def __init__(self, filenames, *args, **kwargs):
        if isinstance(filenames, str):
            filenames = [filenames]
        self.__filenames = filenames
        self.__event_ids = None
        self.__run_numbers = None
        self.__parse_event_ids()
        self.__parse_run_numbers()
        self.__i_events_per_file = np.zeros((len(filenames), 2), dtype=int)
        i_event = 0
        for i_file, filename in enumerate(filenames):
            file = uproot.open(filename)
            events_in_file = file['CalibTree'].num_entries
            self.__i_events_per_file[i_file] = [i_event, i_event + events_in_file]
            i_event = events_in_file

    def get_filenames(self):
        return self.__filenames
  
    def get_event_ids(self):
        if self.__event_ids is None:
            return self.__parse_event_ids()
        return self.__event_ids
  
    def __parse_event_ids(self):
        self.__event_ids = np.array([], dtype=int)
        for filename in self.__filenames:
            file = uproot.open(filename)
            self.__event_ids = np.append(self.__event_ids, file['CalibTree']['EventHeader./EventHeader.fNum'].array(library='np').astype(int))

    def get_run_numbers(self):
        if self.__run_numbers is None:
            self.__parse_run_numbers()
        return self.__run_numbers

    def __parse_run_numbers(self):
        self.__run_numbers = np.array([], dtype=int)
        for filename in self.__filenames:
            file = uproot.open(filename)
            self.__run_numbers = np.append(self.__event_ids, file['CalibTree']['EventMetadata./EventMetadata.fRun'].array(library='np').astype(int))
  
    def get_n_events(self):
        return self.get_event_ids().shape[0]

    def get_events(self,bunchsize=1, entry_start=0, entry_stop=np.inf, exclude_incomplete=False, randomize=False, read_times=False, read_triggers=False, apply_ars_roll=True):
        """
        get a generator for the event loop to return noise events
        
        Paramteters
        -----------
        bunchsize: : int (default 1)
            number of events per bunch to return
        entry_start: int (default 0)
            first entry in the files to read
        entry_stop: int (default inf)
            last entry in the files to read
        exclude_incomplete_bunches: bool (default False)
            stop bunch loop after last complete bunch. May be useful when setting randomize=True
        randomize: bool (default False)
            shuffle order in which bunches are read out
        read_times: bool (default False)
            read in the event posix_time and datetime conversion
        read_triggers: bool (default False)
            read in the trigger flags
        apply_ars_roll: bool (default True)
            roll each event by the found stop.
  
        Returns
        -------
        array of traces (self.data)
        """
        
        # limit entry_start and entry_stop to available range
        if entry_start < self.__i_events_per_file[0][0]:
            logger.warning("Limiting entry_start to {first}".format(first=self.__i_events_per_file[0][0]))
            entry_start = self.__i_events_per_file[0][0]
        if entry_stop > self.__i_events_per_file[-1][1] + 1:
            logger.warning("Limiting entry_stop to {last}".format(last=self.__i_events_per_file[-1][1] + 1))
            entry_stop = self.__i_events_per_file[-1][1] + 1

        # define starts and stops of bunches
        bunch_starts = list(np.arange(entry_start, entry_stop, bunchsize))
        bunch_stops = list(bunch_starts[1:])
        if not exclude_incomplete:
            if entry_stop not in bunch_stops:
                bunch_stops.append(entry_stop)
        bunch_starts_stops = zip(bunch_starts, bunch_stops)

        # shuffle bunches if requested
        if randomize:
            logger.info("randomizing bunch order...")
            np.random.shuffle(bunch_starts_stops)

        # loop over bunches    
        for bunch_start, bunch_stop in bunch_starts_stops:
            #print(bunch_start, bunch_stop)
            data = []
            trigger = []
            posix_times = []
            run = []
            station_id = []
            stops = []
            # loop over input files to extract needed data
            for i_file, filename in enumerate(self.get_filenames()):
                #print(self.__i_events_per_file[i_file])
                if bunch_start > self.__i_events_per_file[i_file][1]:
                    continue
                if bunch_stop <= self.__i_events_per_file[i_file][0]:
                    continue #break
                # bunch start/stop to local file entries
                read_start = max(bunch_start, self.__i_events_per_file[i_file][0])-self.__i_events_per_file[i_file][0]
                read_stop = min(bunch_stop, self.__i_events_per_file[i_file][1]+1)-self.__i_events_per_file[i_file][0]
                #logger.info("reading data for file {noise_file}", noise_file=noise_file)

                with uproot.open(filename) as noise_file:
                    noise_tree = noise_file["CalibTree"]
                    data.append(np.array(noise_tree["AmpOutData."]["AmpOutData.fData"].array(entry_start=read_start, entry_stop=read_stop)))
                    if read_triggers:
                        # trigger jagged array only consists of single number, so drop the array [:,0]
                        logger.debug("reading trigger info")
                        trigger.append(np.array(noise_tree['EventHeader.']['EventHeader.fTrgInfo'].array(interpretation = ARIANNA_uproot_interpretation['trigger'], entry_start=read_start, entry_stop=read_stop))[:,0])

                    if read_times:
                        # consists of posix time [s] and [ns]
                        logger.debug("reading times")
                        event_times_file = np.array(noise_tree['EventHeader.']['EventHeader.fTime'].array(interpretation = ARIANNA_uproot_interpretation['time'], entry_start=read_start, entry_stop=read_stop)[:,0])
                        posix_times.append(event_times_file)


                    # read the run number
                    run.append(noise_tree['EventMetadata./EventMetadata.fRun'].array(library="np", entry_start=read_start, entry_stop=read_stop))
                    # read the station id
                    station_id.append(noise_tree['EventMetadata./EventMetadata.fStnId'].array(library="np", entry_start=read_start, entry_stop=read_stop))

                    # roll the array
                    stop_block_array = np.array(noise_tree['RawData./RawData.fStop'].array(library="np", entry_start=read_start, entry_stop=read_stop))
                    stops.append(find_stops_vectorized(stop_block_array, True))

            self.data = np.concatenate(data) * units.mV

            if read_times:
                self.posix_time = np.concatenate(posix_times)
                self.datetime = np.array([np.datetime64(int(t), 's') for t in self.posix_time])
            if read_triggers:
                self.trigger = np.concatenate(trigger)
            self.station_id = np.concatenate(station_id)
            self.run_number = np.concatenate(run)
            self.nevts = len(self.data)

            # skip events that don't have a proper information of the stop point
            self.stops = np.concatenate(stops)
            if apply_ars_roll:
                for i in range(len(self.data)):
                    #roll each event around the found stop
                    self.data[i] = np.roll(self.data[i], -self.stops[i], axis=1)
            yield self.data
