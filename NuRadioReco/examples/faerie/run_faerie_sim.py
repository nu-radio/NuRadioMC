import NuRadioReco.modules.io.coreas.readFAERIEShower
import NuRadioReco.modules.io.eventWriter

from NuRadioReco.detector import detector
from NuRadioReco.utilities import units, signal_processing

from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.base_trace import BaseTrace

from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData

from NuRadioMC.examples.RNO_G_trigger_simulation.simulate import \
    detector_simulation_with_data_driven_noise, rnog_flower_board_high_low_trigger_simulations

import NuRadioReco.modules.channelReadoutWindowCutter
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.efieldToVoltageConverterPerEfield
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
from NuRadioReco.modules.measured_noise.RNO_G.noiseImporter import noiseImporter

from NuRadioReco.examples.faerie.detector import FAERIEDetector

import datetime as dt
import numpy as np
import logging
import argparse
import copy
from collections import deque
import os 
import math

import pymongo
import certifi

client = pymongo.MongoClient(
    "mongodb://radio.zeuthen.desy.de:27017/",
    tls=True,
    tlsCAFile=certifi.where()
)
DEFAULT_PEDESTAL_V = 1.5 # *units.V
# Per-channel clip thresholds (mV) from pedestal characterization.
# Averaged over station 23 runs 1000 and 3400 from pedestal.root.
# RADIANT 12-bit ADC: 0-2.5V range, pedestal at ~1.5V (not midpoint).
# (negative_clip, positive_clip) = (-pedestal_mV, 2500 - pedestal_mV)
CLIP_THRESHOLDS_MV = {
    0: (-1453, +1047),
    1: (-1465, +1035),
    2: (-1540, +960),
    3: (-1552, +948),
    4: (-1304, +1196),
    5: (-1465, +1035),
    6: (-1472, +1028),
    7: (-1470, +1030),
    8: (-1514, +986),
    9: (-1523, +977),
    10: (-1472, +1028),
    11: (-1370, +1130),
    12: (-1461, +1039),
    13: (-1414, +1086),
    14: (-1483, +1017),
    15: (-1468, +1032),
    16: (-1494, +1006),
    17: (-1562, +938),
    18: (-1562, +938),
    19: (-1442, +1058),
    20: (-1465, +1035),
    21: (-1544, +956),
    22: (-1468, +1032),
    23: (-1482, +1018),
}
TRIGGER_CHANNELS = [0, 1, 2, 3]
TILE_OVERLAP = 200  # samples at 5 GHz (~40 ns)

def RNO_G_HighLow_Thresh(lgRate_per_hz):
    return (-859 + np.sqrt(39392706 - 3602500 * lgRate_per_hz)) / 1441.0

def get_trigger_noise_vrms(det, station_id, trigger_channels, temperature=300):
    vrms_per_channel = []
    for channel_id in trigger_channels:
        resp = det.get_signal_chain_response(station_id, channel_id, trigger=True)
        vrms_per_channel.append(
            signal_processing.calculate_vrms_from_temperature(
                temperature=temperature, response=resp))
    return vrms_per_channel

def pad_traces(event, det, pad_before=200 * units.ns, pad_after=400 * units.ns,trigger_channels=[0]):
    """ Makes sure all traces have the same length and starting time. """
    sim_station = event.get_station().get_sim_station()

    tstarts = []
    tends = []
    pulse_times = []
    for electric_field in sim_station.get_electric_fields():
        if electric_field.get_channel_ids()[0] not in trigger_channels:
            ## don't care about non-trigger channels regarding start time
            continue
        if len(electric_field.get_times()) <= 200:
            print(f"!!!!!!!!!!!!!! Warning: Electric field with only {len(electric_field.get_times())} samples found. !!!!!!!!!!!!!!")
            print("Event ID:", event.get_id(), "Station ID:", sim_station.get_id(),"ch",electric_field.get_channel_ids())
            print("E-field min/max:", electric_field.get_trace().min(), electric_field.get_trace().max())
            # dtime = np.linalg.norm(electric_field.get_position())/(3e8 * units.m / units.s)  # time it takes for light to travel the distance
            # electric_field.set_trace_start_time(dtime)
            continue
        times = electric_field.get_times()
        tstarts.append(times[0])
        tends.append(times[-1])
        pulse_times.append(times[np.argmax(electric_field.get_hilbert_envelope_mag())])
    print("trace tstarts (ns):", [f"{t/units.ns:.1f}" for t in tstarts])
    print("trace tends   (ns):", [f"{t/units.ns:.1f}" for t in tends])
    print("pulse times  (ns):", [f"{t/units.ns:.1f}" for t in pulse_times])
    if len(tstarts) == 0 or len(tends) == 0:
        tstart = 0 * units.ns
        tend = 0 * units.ns
    else:
        tstart = np.min(tstarts) - pad_before
        tend = np.max(tends) + pad_after
        pulse_time = np.min(pulse_times)
    # if tstart > pulse_time - 20 * units.ns:
    #     ## front padding too large, use the pulse time minus 20 ns
    #     print("\ttrace tstart (ns):", f"{tstart/units.ns:.1f}")
    #     print("\tpulse time  (ns):", f"{pulse_time/units.ns:.1f}")
    #     print( f"front padding [{pad_before/units.ns:.2f}] is too large, use the pulse time minus 20 ns")
    #     tstart = pulse_time - 20 * units.ns

    t_readout_window = det.get_number_of_samples(sim_station.get_id(), 0) / \
        det.get_sampling_frequency(sim_station.get_id(), 0)
    if tend - tstart < t_readout_window:
        tend = tstart + t_readout_window + pad_after * units.ns

    # assumes all efields have the same sampling rate
    n_samples = int((tend - tstart) * electric_field.get_sampling_rate())
    if n_samples % 2 != 0:
        n_samples += 1
    print("event:",event.get_id()," t_readout_window",t_readout_window/units.ns,"ns","tstart",tstart/units.ns,"tend",tend/units.ns)
    print(n_samples,"samples at",electric_field.get_sampling_rate()/units.GHz,"GHz")
    for electric_field in sim_station.get_electric_fields():
        readout = BaseTrace()
        readout.set_trace(np.zeros((3, n_samples)), electric_field.get_sampling_rate(), tstart)

        # if len(electric_field.get_trace()) > 100: ## assumes short traces are not useful
        # new_efield = readout + electric_field ##
        # readout.add_to_trace(new_efield)
        # try:
        readout.add_to_trace(electric_field,raise_error=False)
            # readout.add_to_trace(electric_field)
        # except:
        #     ## typically fail when efield has too few samples or outside readout window (surface channel)
        #     # print(f"!!!!!!!!!!!!!! Warning couldn't add_to_trace, use zero trace !!!!!!!!!!!!!!")
        #     # print("Event ID:", event.get_id(), "Station ID:", sim_station.get_id(),"ch",electric_field.get_channel_ids())
        #     # print("E-field shape",electric_field.get_trace().shape,"\nmin/max:", electric_field.get_trace().min(), electric_field.get_trace().max())
        #     pass
        electric_field.set_trace(readout.get_trace(), "same", tstart)



def split_events(event, det, trigger_channels,num_channels_per_event=4):
    """ Split an event with more than 4 channels into multiple events with 4 channels. """

    det.set_event(event)
    station = event.get_station()
    # if (len(det.get_channel_ids(station.get_id())) == len(trigger_channels) and
    #     np.all(det.get_channel_ids(station.get_id()) == trigger_channels)):
    #     return [event]
    if (len(det.get_channel_ids(station.get_id())) == num_channels_per_event and
        np.all(det.get_channel_ids(station.get_id()) == np.arange(num_channels_per_event))):
        return [event]

    # if len(det.get_channel_ids(station.get_id())) == len(trigger_channels):
    #     raise ValueError("Some thing unexpected happend. The event has only 4 channels but "
    #                      f"the channel ids {det.get_channel_ids(station.get_id())} do not "
    #                      f"match the trigger channels ({trigger_channels})")

    # Split the event into multiple events
    all_sim_channel_ids = np.array([efields.get_channel_ids()[0] for efields in station.get_sim_station().get_electric_fields()])
    string_sim_channel_ids = ["{}".format(ch_ids) for ch_ids in all_sim_channel_ids] ## mimic list of strings

    sorted_string_sim_channel_ids = sorted(string_sim_channel_ids)
    num_from_sorted_string = [ int(ch) for ch in sorted_string_sim_channel_ids ]
    argsort_from_sorted_num = np.argsort(num_from_sorted_string)
    # print("argsort_from_num[:20]",argsort_from_sorted_num[:20])
    
    channel_to_index_map = {ch_id: index for ch_id, index in enumerate(argsort_from_sorted_num)}
    # print("channel_to_index_map",channel_to_index_map)


    # sim_channel_ids = np.unique([efields.get_channel_ids() for efields in station.get_sim_station().get_electric_fields()])
    # channel_positions = np.array([det.get_relative_position(station.get_id(), sim_channel_id) for sim_channel_id in sim_channel_ids])
    
    ## testing channel ids and positions
    # for chid,chpos in zip(all_sim_channel_ids, channel_positions):
    #     print(f"Sim channel IDs\n{chid[0]} at {chpos}")

    # unique_xy_positions = np.unique(channel_positions[:, :2], axis=0)
    # n_batches = len(unique_xy_positions)
    n_batches = np.ceil( len(all_sim_channel_ids)/ num_channels_per_event ).astype(int) ## round up

    # sim_channel_ids_batches = [[] for _ in range(n_batches)]
    # for sim_channel_id, xy_position in zip(sim_channel_ids, channel_positions[:, :2]):
    #     idx = np.arange(n_batches)[np.all(unique_xy_positions == xy_position, axis=1)][0]
    #     sim_channel_ids_batches[idx].append(sim_channel_id)
    
    # sim_channel_ids_batches = np.array(all_sim_channel_ids[argsort_from_sorted_num]).reshape(n_batches, num_channels_per_event).tolist()
    sim_channel_ids_batches = [[] for _ in range(n_batches)]
    for i in range(n_batches):
        channels_in_batch = np.array([ch for ch in range(num_channels_per_event)]) + i*num_channels_per_event
        for ch in channels_in_batch:
            if ch < len(all_sim_channel_ids):
                sim_channel_ids_batches[i].append(all_sim_channel_ids[channel_to_index_map[ch]])

    events = []

    for ievent,sim_channel_ids_batch in enumerate(sim_channel_ids_batches):
        # print("sim_channel_ids_batch",sim_channel_ids_batch)
        new_event = copy.deepcopy(event)
        new_event.set_id(ievent)

        # if len(sim_channel_ids_batch) != len(trigger_channels):
        #     raise ValueError("Some thing unexpected happend. The batch has not the same number of channels as the trigger channels "
        #                      f"sim_channel_ids_batch: {sim_channel_ids_batch}, trigger_channels: {trigger_channels}")

        new_sim_station = NuRadioReco.framework.sim_station.SimStation(station.get_id())  # set sim station id to 0
        new_sim_station.set_is_neutrino() # HACK: Since the sim. efields are always at the exact positions as the antenna(channels).
        new_station = NuRadioReco.framework.station.Station(station.get_id())
        new_station.set_sim_station(new_sim_station)
        new_event.set_station(new_station)  # overwrites existing station
        events.append(new_event)

        # sort the sim_channel_ids_batch by depth
        # depth = np.array([det.get_relative_position(station.get_id(), sim_channel_id)[2] for sim_channel_id in sim_channel_ids_batch])
        # sort = np.argsort(depth)

        # sorted_sim_channel_ids_batch = np.array(sim_channel_ids_batch)[sort]
        sorted_sim_channel_ids_batch = np.array(sim_channel_ids_batch)  # assume batch already sorted by ch-id 
        print("  sorted_sim_channel_ids_batch",sorted_sim_channel_ids_batch)
        # for sim_channel_id, new_id in zip(sorted_sim_channel_ids_batch, trigger_channels):
        for sim_channel_id, new_id in zip(sorted_sim_channel_ids_batch, np.arange(len(sorted_sim_channel_ids_batch))):  ## assume already sorted by ch-id 
            for efield in station.get_sim_station().get_electric_fields_for_channels([sim_channel_id]):
                efield_new = copy.deepcopy(efield)
                efield_new.set_channel_ids([new_id])
                new_sim_station.add_electric_field(efield_new)
                # print(f"  Adding sim efield channel {efield_new.get_channel_ids()[0]} at {efield_new.get_position()}")
                # print(f"    min/max: {efield_new.get_trace().min()}/{efield_new.get_trace().max()} with {(efield_new.get_trace()).shape} samples")


    return events
     
## FT noise injection from https://github.com/nu-radio/NuRadioMC/blob/cr_proxy_with_FT_noise/NuRadioMC/examples/08_RNO_G_CR_proxy_simulation/simulate_fixed_response_v9.py
from collections import deque
class FTNoisePool:
    """Streaming pool of forced-trigger noise traces from ROOT files.

    Loads one ROOT file at a time (~560 events, ~210 MB in memory).
    Events flagged in the clean mask are skipped.
    """

    def __init__(self, ft_dir, station_id=23, seed=None, clean_mask_path=None):
        self._station_id = station_id
        self._rng = np.random.default_rng(seed)
        self._buffer = deque()
        self._files_loaded = 0

        # Discover ROOT files
        self._ft_files = sorted([
            os.path.join(ft_dir, f)
            for f in os.listdir(ft_dir)
            if f.startswith(f"station{station_id}_run") and f.endswith(".root")
        ])
        self._rng.shuffle(self._ft_files)

        if not self._ft_files:
            raise FileNotFoundError(
                f"No FT ROOT files matching station{station_id}_run*.root in {ft_dir}")

        self._file_idx = 0

        # Load contamination mask
        self._flagged = set()
        if clean_mask_path and os.path.exists(clean_mask_path):
            mask_data = np.load(clean_mask_path)
            run_nums = mask_data['runNum']
            evt_nums = mask_data['eventNum']
            is_clean = mask_data['is_clean']
            for r, e, c in zip(run_nums, evt_nums, is_clean):
                if c == 0:
                    self._flagged.add((int(r), int(e)))
            print(f"FTNoisePool: loaded clean mask, {len(self._flagged)} flagged events")

        print(f"FTNoisePool: {len(self._ft_files)} ROOT files in {ft_dir}")

    def _load_next_file(self):
        """Load one ROOT file's events into the buffer."""

        max_retries = 10
        for attempt in range(max_retries):
            if self._file_idx >= len(self._ft_files):
                self._file_idx = 0
                self._rng.shuffle(self._ft_files)
                print("FTNoisePool: cycled through all files, reshuffling")

            fpath = self._ft_files[self._file_idx]
            self._file_idx += 1
            self._files_loaded += 1

            try:
                reader = readRNOGData()
                reader.begin(
                    [fpath],
                    convert_to_voltage=True,
                    apply_baseline_correction="median",
                    selectors=[lambda einfo: einfo.triggerType == "FORCE"],
                    select_runs=False,
                    read_calibrated_data=False,
                )
            except Exception as e:
                print(f"FTNoisePool: skipping corrupt file "
                      f"{os.path.basename(fpath)}: {e}")
                continue

            loaded = 0
            skipped = 0
            try:
                for evt in reader.run():
                    station = evt.get_station(self._station_id)
                    if station is None:
                        continue

                    run_num = evt.get_run_number()
                    evt_num = evt.get_id()
                    if (run_num, evt_num) in self._flagged:
                        skipped += 1
                        continue

                    traces = {}
                    for channel in station.iter_channels():
                        traces[channel.get_id()] = channel.get_trace().copy()

                    if traces:
                        self._buffer.append(traces)
                        loaded += 1
            except Exception as e:
                print(f"FTNoisePool: error reading events from "
                      f"{os.path.basename(fpath)}: {e}")

            try:
                reader.end()
            except Exception:
                pass

            if self._files_loaded <= 3 or self._files_loaded % 50 == 0:
                print(f"FTNoisePool: loaded {loaded} events from "
                      f"{os.path.basename(fpath)} (skipped {skipped} flagged)")

            if loaded > 0:
                return

        raise RuntimeError("FTNoisePool: failed to load events after "
                           f"{max_retries} files")

    def get_noise_event(self):
        """Pop one event's traces from the buffer.

        Returns dict {ch_id: np.array} in Volts, 2048 samples at 3.2 GHz.
        """
        if not self._buffer:
            self._load_next_file()
        return self._buffer.popleft()


def upsample_trace(trace, target_n_samples):
    """Upsample via FFT zero-padding above Nyquist."""
    n_orig = len(trace)
    spec = np.fft.rfft(trace)
    new_spec = np.zeros(target_n_samples // 2 + 1, dtype=complex)
    new_spec[:len(spec)] = spec
    return np.fft.irfft(new_spec, n=target_n_samples) * (target_n_samples / n_orig)


def tile_noise_overlap_add(tiles, target_length, overlap=TILE_OVERLAP):
    """Tile upsampled noise segments with Hann crossfade overlap-add.

    Parameters
    ----------
    tiles : list of 1D arrays
        Each array is an upsampled FT noise trace (3200 samples at 5 GHz).
    target_length : int
        Desired output length in samples.
    overlap : int
        Number of samples to overlap between adjacent tiles.

    Returns
    -------
    np.array of length target_length
    """
    if not tiles:
        return np.zeros(target_length)

    n_tile = len(tiles[0])
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(overlap) / overlap))

    total_len = n_tile + (len(tiles) - 1) * (n_tile - overlap)
    result = np.zeros(max(total_len, target_length + overlap))

    pos = 0
    for tile in tiles:
        windowed = tile.copy()
        windowed[:overlap] *= ramp
        windowed[-overlap:] *= ramp[::-1]
        result[pos:pos + n_tile] += windowed
        pos += n_tile - overlap

    return result[:target_length]


_ft_noise_pool = None
_last_ft_events = None
_readout_to_trigger_transfer = {}
def _get_readout_to_trigger_transfer(ch_id, n_samples, det, station_id):
    """Compute and cache the transfer function to convert readout-path
    FT noise to trigger-path noise: trigger_response / readout_response."""
    key = (ch_id, n_samples)
    if key not in _readout_to_trigger_transfer:
        sr = 5.0 * units.GHz
        ff = np.fft.rfftfreq(n_samples, d=1.0 / sr)
        readout = rnogHardwareResponse.get_filter(
            ff, station_id, ch_id, det,
            sim_to_data=True, is_trigger=False)
        trigger = rnogHardwareResponse.get_filter(
            ff, station_id, ch_id, det,
            sim_to_data=True, is_trigger=True)
        # Regularize where readout response is near zero
        readout_abs = np.abs(readout)
        max_r = np.max(readout_abs)
        safe_readout = np.where(
            readout_abs > 1e-3 * max_r, readout, max_r)
        _readout_to_trigger_transfer[key] = trigger / safe_readout
    return _readout_to_trigger_transfer[key]

def forced_trigger_injection(event,station,detector,**kwargs):
    ## run hardware response then add FT noise
    rnogHardwareResponse.run(event, station, detector, sim_to_data=True)
    if _ft_noise_pool is None:
        print("noise pool is None... RETURN without adding noise")
        return
    # Determine tiling geometry from first channel's trace length
    first_ch = next(station.iter_channels())
    n_internal = len(first_ch.get_trace())
    n_up = int(round(2048 * (5.0 / 3.2)))  # 3200 samples per FT event at 5 GHz
    stride = n_up - TILE_OVERLAP
    n_tiles = max(1, math.ceil(n_internal / stride))
    print(f"Injecting FT noise: n_internal={n_internal}, n_up={n_up}, stride={stride}, n_tiles={n_tiles}")

    # Pop n_tiles FT events (each has all 24 channels)
    global _last_ft_events
    ft_events = [_ft_noise_pool.get_noise_event() for _ in range(n_tiles)]
    _last_ft_events = ft_events

    for channel in station.iter_channels():
        ch_id = channel.get_id()
        trace = channel.get_trace()

        tiles = []
        for ft_evt in ft_events:
            ft_ch = ft_evt.get(ch_id)
            if ft_ch is not None:
                tiles.append(upsample_trace(ft_ch, n_up))

        if tiles:
            noise = tile_noise_overlap_add(tiles, n_internal)

            # Inject noise into trigger COPY only (for trigger
            # evaluation). Readout channels get FT noise post-cutter
            # in resampler_with_ft_noise_and_clip to avoid zeros from
            # the readout window extending before the internal trace.
            if ch_id in TRIGGER_CHANNELS and channel.has_extra_trigger_channel():
                trig_ch = channel.get_trigger_channel()
                trig_trace = trig_ch.get_trace()
                n_trig = len(trig_trace)
                print("n_trig", n_trig, "n_internal", n_internal,"start time", trig_ch.get_trace_start_time()/units.ns,"ns")

                # Transform FT noise from readout path to trigger
                # path using hardware response ratio. FT noise has
                # readout response baked in; trigger copies need
                # trigger-path noise instead.
                transfer = _get_readout_to_trigger_transfer(
                    ch_id, n_internal, detector, station.get_id())
                noise_fft = np.fft.rfft(noise)
                trig_noise = np.fft.irfft(
                    noise_fft * transfer, n=n_internal)

                trig_ch.set_trace(
                    trig_trace + trig_noise[:n_trig],
                    trig_ch.get_sampling_rate())

_original_resampler = NuRadioReco.modules.channelResampler.channelResampler()
# _noise_importer_instance = noiseImporter()
# _noise_importer_instance.begin(
#             noise_files=ft_files,
#             match_station_id=True,
#             scramble_noise_file_order=True,
#             random_seed=args.ft_seed,
#             inject_trigger_copies=True,
#             trigger_channels=DEEP_TRIGGER_CHANNELS,
#             hardware_response_incorporator=hw_resp,
#             reader_kwargs={
#                 "selectors": ft_selectors,
#                 "select_runs": False,
#                 "convert_to_voltage": True,
#                 "apply_baseline_correction": "median",
#             },
#         )
# _noise_importer = _noise_importer_instance
def resampler_with_noise_and_clip(event, station, detector, **kwargs):
    """Resample, optionally inject FT noise for readout, then clip."""
    _original_resampler.run(event, station, detector, **kwargs)

    if isinstance(station, NuRadioReco.framework.sim_station.SimStation):
        return

    has_triggered = station.has_triggered()
    # Inject FT noise into readout channels (stage 2)
    if _ft_noise_pool is not None:
        # _noise_importer.run(event, station, detector)
        _noise = _ft_noise_pool.get_noise_event()
        for channel in station.iter_channels():
            print("channel sampling rate", channel.get_sampling_rate()/units.GHz,"GHz ,lenght",channel.get_trace().shape)
            print("trace start time (ns)", channel.get_trace_start_time()/units.ns,"ns")
            print("trace length (ns)", len(channel.get_trace())/channel.get_sampling_rate()/units.ns,"ns")
            trace = channel.get_trace()
            if has_triggered:
                channel.set_trace(
                    trace + _noise.get(channel.get_id()),
                    channel.get_sampling_rate())
            else: 
                channel.set_trace(_noise.get(channel.get_id()),
                    channel.get_sampling_rate())
        # logger.debug("Stage 2: readout noise injected")
        print("Stage 2: readout noise injected")

    # ADC saturation clipping
    if _adc_clip_range is not None:
        lo, hi = _adc_clip_range
        for channel in station.iter_channels():
            trace = channel.get_trace()
            channel.set_trace(
                np.clip(trace, lo, hi),
                channel.get_sampling_rate())
                       
if __name__ == "__main__":
    # Parse eventfile as argument   
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('inputfilename', type=str, nargs='*',
                        help='path to NuRadioMC simulation result')


    parser.add_argument('--add_noise', action='store_true', help='Add noise to the traces')

    parser.add_argument('--depth', nargs="?", type=float, default=None, help='If specified, used to select simulated pulses at a given depth.')

    parser.add_argument('--detector_file', type=str, nargs='?',
                        default=None,
                        help='path to detectordescription')

    parser.add_argument('--noise_type', type=str, nargs='?',
                        default="rayleigh",
                        help='Specify noise type')

    parser.add_argument('--output_file', type=str, nargs='?',
                        default=None,
                        help='path to detectordescription')

    parser.add_argument('--station', type=int, nargs='?',
                        default=11,
                        help='station to simulate')
    
    parser.add_argument("--ft_noise_dir", type=str,
                        default=None,
                        help="Path to FT ROOT files for noise injection")
    
    parser.add_argument("--ft_seed", type=int,
                        default=None,
                        help="RNG seed for FT noise pool file shuffling")
    
    parser.add_argument("--ft_clean_mask", type=str,
                        default=None,
                        help="Path to clean_mask_station*.npz for FT contamination filtering")
    parser.add_argument("--pedestal_voltage", type=float, default=DEFAULT_PEDESTAL_V,
                        help="ADC pedestal voltage in V (default: 1.5)")
    parser.add_argument("--event_time", type=str, default="2022-10-01")
    args = parser.parse_args()


    # Load the real detector response
    # try:
    det_rnog = detector.rnog_detector.Detector(
        select_stations=[args.station], detector_file=args.detector_file, database_connection="RNOG_public", always_query_entire_description=False)
    # except:
    #     det_rnog = detector.rnog_detector.Detector(select_stations=[args.station],
    #         detector_file="/work/icecube/users/noppadol/cr_sims/rnog_all_stations_2022-10-01.json.xz")
    
    event_time = dt.datetime.fromisoformat(args.event_time)
    det_rnog.update(event_time)
    # det_rnog.update(dt.datetime(2022, 10, 1))
    det_ch = det_rnog.get_channel(args.station, 0)
    adc_min = det_ch.get("adc_min_voltage", 0) * units.V
    adc_max = det_ch.get("adc_max_voltage", 2.5) * units.V
    _adc_clip_range = (adc_min - args.pedestal_voltage * units.V,
                       adc_max - args.pedestal_voltage * units.V)

    trigger_channels = TRIGGER_CHANNELS
    num_channels_per_event = 24  # number of channels per event in the input file

    threshold_1Hz = RNO_G_HighLow_Thresh(0)
    print(f"Trigger threshold: {threshold_1Hz:.3f} sigma (1 Hz rate)")
    
    thresholds = {
        "hilo_sigma_1Hz": threshold_1Hz
    }

    # rnog_resp_ch0 = det_rnog.get_signal_chain_response(args.station, 0, trigger=True)
    # vrms_thermal = signal_processing.calculate_vrms_from_temperature(300 * units.kelvin, response=rnog_resp_ch0)
    # print(f"Thermal noise amplitude: {vrms_thermal / units.mV} mV")
    min_freq = 10 * units.MHz
    max_freq = 1200 * units.MHz
    vrms_300K_in_min_max = signal_processing.calculate_vrms_from_temperature(
        300 * units.kelvin, bandwidth=[min_freq, max_freq])
    trigger_noise_vrms = get_trigger_noise_vrms(
        det_rnog, args.station, trigger_channels, temperature=300)
    
    print(f"{threshold_1Hz:.2e} , \n{vrms_300K_in_min_max:.2e}") ## 3.76 sigma, 1.57e-5 V before signal chain
    print("vrms trigger",trigger_noise_vrms) ## ~3-4mV (after singal chain)


    # FT noise Vrms on trigger channels, transformed from readout path to
    # trigger path using the hardware response transfer function
    # (trigger_response / readout_response). Measured from 200 FT noise
    # realizations with the transfer function applied.
    TRIGGER_VRMS_FT = {
        0: 4.102e-3, 1: 4.627e-3, 2: 3.703e-3, 3: 2.625e-3,  # Volts
    }
    # Initialize FT noise pool
    if args.ft_noise_dir:
        _ft_noise_pool = FTNoisePool(
            ft_dir=args.ft_noise_dir,
            station_id=args.station,
            seed=args.ft_seed,
            clean_mask_path=args.ft_clean_mask,
        )
    else:
        print("WARNING: --ft_noise_dir not set, running without FT noise injection")

    channelReadoutWindowCutter = NuRadioReco.modules.channelReadoutWindowCutter.channelReadoutWindowCutter()
    channelReadoutWindowCutter.begin()

    channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
    channelResampler.begin()

    eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
    if args.output_file is not None:
        outputfilename = args.output_file
    else:
        outputfilename = args.inputfilename[0].replace(".hdf5", ".nur")

    eventWriter.begin(filename=outputfilename, max_file_size=1024 * 3)

    readFAERIEShower = NuRadioReco.modules.io.coreas.readFAERIEShower.readFAERIEShower()
    readFAERIEShower.begin(
        args.inputfilename, logger_level=logging.INFO
    )

    efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
    # efieldToVoltageConverter.begin(post_pulse_time=100 * units.ns, pre_pulse_time=100 * units.ns)
    efieldToVoltageConverter.begin(post_pulse_time=400 * units.ns, pre_pulse_time=200  * units.ns)

    efieldToVoltageConverterPerEfield = NuRadioReco.modules.efieldToVoltageConverterPerEfield.efieldToVoltageConverterPerEfield()

    rnogHardwareResponse = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
    rnogHardwareResponse.begin(trigger_channels=trigger_channels)

    channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
    channelGenericNoiseAdder.begin()
    
    # channelIceThermalNoiseAdder = NuRadioReco.modules.channelIceThermalNoiseAdder.channelIceThermalNoiseAdder()
    # channelIceThermalNoiseAdder.begin()

    dummy_detector_for_positions_only = FAERIEDetector()

    mode = {
        'Channels': True,
        'ElectricFields': False,
        'SimChannels': True,
        'SimElectricFields': False
    }

    for combined_event in readFAERIEShower.run(depth=args.depth, station_id=args.station):

        for edx, event in enumerate(split_events(combined_event, dummy_detector_for_positions_only, trigger_channels,num_channels_per_event=num_channels_per_event)):
            dummy_detector_for_positions_only.set_event(event)
            pad_traces(event, det_rnog,trigger_channels=trigger_channels)

            shower = event.get_first_sim_shower()
            for sdx, station in enumerate(event.get_stations()):
                print("Printing triggers before setting pre_trigger_times:")
                for trigger in station.get_triggers().values():
                    print("trigger from station.get_triggers():", trigger)
                    trigger.set_pre_trigger_times(250 * units.ns)
                    station.set_trigger(trigger)
                sim_station = station.get_sim_station()

                # if (edx + sdx) % 100 == 0:
                if (edx + sdx) % 1 == 0:
                    print(f"Processing event: {event.get_id()} station {station.get_id()}")
                    print(f"Energy: {shower.get_parameter(shp.energy) / units.PeV} PeV, "
                        f"Zenith: {shower.get_parameter(shp.zenith) / units.deg}, "
                        f"Azimuth: {shower.get_parameter(shp.azimuth) / units.deg}")

                # Temporary sanity checks - to apply the correct noise and filter the event
                # can only have 4 channels with IDs [0, 1, 2, 3] (and they should be at the
                # correct depths)

                ## skip assertion for now due to change in expectation
                # assert np.all(dummy_detector_for_positions_only.get_channel_ids(station.get_id()) == trigger_channels), "Expected channels [0, 1, 2, 3]"
                # channel_depths = np.array([dummy_detector_for_positions_only.get_relative_position(
                #     station.get_id(), channel_id)[2] for channel_id in dummy_detector_for_positions_only.get_channel_ids(station.get_id())])
                # assert np.all(np.argsort(channel_depths) == trigger_channels), "Expected channels to be sorted by depth"

                if args.add_noise and args.noise_type == "data-driven":
                    detector_simulation_with_data_driven_noise(
                        event, station, det_rnog, trigger_channels=trigger_channels)
                elif args.add_noise and args.noise_type == "FT-injection":
                    efieldToVoltageConverter.run(event, station, det_rnog, channel_ids=np.arange(num_channels_per_event))
                    efieldToVoltageConverterPerEfield.run(event, station, det_rnog)
                    forced_trigger_injection(event,station,det_rnog,
                                             trigger_channels=trigger_channels,
                                             num_channels_per_event=num_channels_per_event)
                else:
                    assert args.noise_type == "rayleigh", "Only 'rayleigh' and 'data-driven' noise is supported."
                    efieldToVoltageConverter.run(event, station, det_rnog, channel_ids=np.arange(num_channels_per_event))
                    efieldToVoltageConverterPerEfield.run(event, station, det_rnog)

                    if args.add_noise:
                        channelGenericNoiseAdder.run(
                            event, station, det_rnog,
                            amplitude=vrms_300K_in_min_max, min_freq=min_freq, max_freq=max_freq,
                            type='rayleigh')
                        # channelIceThermalNoiseAdder.run(
                        #     event,station,det_rnog,passband=[min_freq,max_freq]
                        # )

                    rnogHardwareResponse.run(event, station, det_rnog, sim_to_data=True)
                print("Running rnog_flower_board_high_low_trigger_simulations with thresholds:", thresholds)
                print("Trigger noise RMS from FT:", TRIGGER_VRMS_FT)
                print("trigger_noise_vrms from Temperature:", trigger_noise_vrms)
                rnog_flower_board_high_low_trigger_simulations(
                    event, station, det_rnog, trigger_channels=trigger_channels,
                    trigger_channel_noise_vrms=list(TRIGGER_VRMS_FT.values()), # TRIGGER_VRMS_FT, #trigger_noise_vrms,
                    high_low_trigger_thresholds=thresholds)
                print('triggered?:',station.has_triggered())
                channelReadoutWindowCutter.run(event, station, det_rnog)
                if args.add_noise and args.noise_type == "FT-injection":
                    resampler_with_noise_and_clip(event,station,det_rnog)
                else:
                    channelResampler.run(event, station, det_rnog)
            # print("running eventWriter with mode",mode)
            eventWriter.run(event, mode=mode)
