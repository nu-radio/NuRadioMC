from reco import Reco
import defs
import utils
import numpy as np
from detector import Detector
import matplotlib.pyplot as plt
from snr import SNR
import defs 
import dedisperse_new
import reco 

class CSW:

    def __init__(self):
        self.azimuth_range = (-np.pi, np.pi)
        self.elevation_range = (-np.pi/2, np.pi/2)
        self.res = 100
        self.radius = 38 / defs.cvac
        self.zoom_window = 80

    def get_correlation_function(self, channel_signals, channel_times, channels_to_include, channel_id, reference_ch, solution,  detectorpath, station_id, mappath):
        det = Detector(detectorpath)
        channel_positions = det.get_channel_positions(station_id = station_id, channels = channels_to_include)
        cable_delays = det.get_cable_delays(station_id = station_id, channels = channels_to_include)

        ttcs = utils.load_ttcs(mappath, channels_to_include)
        origin_xyz = channel_positions[0]

        elevation_vals = np.linspace(*self.elevation_range, self.res)
        azimuth_vals = np.linspace(*self.azimuth_range, self.res)
        ee, aa = np.meshgrid(elevation_vals, azimuth_vals)

        pts = utils.ang_to_cart(ee.flatten(), aa.flatten(), radius = self.radius, origin_xyz = origin_xyz)

        sig_a = channel_signals[channel_id]
        sig_b = channel_signals[reference_ch]
        tvals_a = channel_times[channel_id]
        tvals_b = channel_times[reference_ch]

        t_ab = utils.calc_relative_time(channel_id, reference_ch, src_pos = pts, ttcs = ttcs, comp = solution,
                                            channel_positions = channel_positions, cable_delays = cable_delays)
        
        score = utils.corr_score(sig_a, sig_b, tvals_a, tvals_b, t_ab)

        return t_ab, score 

    
    def get_arrival_delays_AraRoot_xcorr(
        self, channel_signals, channel_times, channels_to_include, reference_ch, reco_delays, solution, detectorpath, station_id, mappath
    ):

        # Calculate the time delay between each channel and the reference channel
        delays = {}
        for ch_ID in channels_to_include:

            if ch_ID == reference_ch:
                # Delay between the reference channel and itself will be 0
                delay = 0

            else:

                # Load  the cross correlation for this channel and the reference
                xcorr_times, xcorr_volts = self.get_correlation_function(channel_signals, channel_times, channels_to_include, ch_ID, reference_ch, solution, detectorpath, station_id, mappath)

                # AraRoot always compares channels with smaller IDs to channels with
                #   larger IDs but we always want to compare to the reference channel.
                # Correct for this if the current ch_ID is larger than the reference ch_ID

                #if ch_ID > reference_ch:
                #    xcorr_times *= -1

                # Identify the `zoom_window` nanosecond window around the
                #   reconstructed delay
                zoomed_indices = np.where(
                    (np.abs( xcorr_times - reco_delays[ch_ID] )) < self.zoom_window // 2
                )[0]
                zoomed_indices_filled = np.arange(min(zoomed_indices), max(zoomed_indices), 1)

                #print(reco_delays[ch_ID], ch_ID, "reco delay")
                #print(xcorr_times - reco_delays[ch_ID], "zoom")
                #print(xcorr_times[np.argmax(xcorr_volts)], "delay")
                #print(zoomed_indices[0], np.argmax(xcorr_volts[zoomed_indices]))
                # Calculate the time of maximum correlation from this
                #   window of expected signal delay.
                if len(zoomed_indices) == 0:

                    # Software triggers are so short, a channel may not have
                    #   signal during the time window where signal is expected.
                    #   As a result, `zoom_indices` will have no entries.
                    # Just find the time of peak correlation for the full
                    #   waveform in this scenario.
                    delay = xcorr_times[ np.argmax(xcorr_volts) ]
                    #print(delay, np.argmax(xcorr_volts), "delay 1")
                    # If the event can't find the time window to zoom in on
                    #   and its not a software trigger, warn user
                else:
                    delay = xcorr_times[
                        np.argmax(xcorr_volts[zoomed_indices_filled]) # index of max xcorr in zoomed array
                        + zoomed_indices_filled[0] # Adjusted by first zoomed_index
                    ]
            #print(delay, ch_ID, "final delay")
            
            delays[ch_ID] = delay

        return delays



    def get_arrival_delays_reco(self, reco_results, channels_to_include, detectorpath, reference_ch, solution, ttcs):

        det = Detector(detectorpath)
        channel_positions = det.get_channel_positions(station_id = 11, channels = channels_to_include)
        cable_delays = det.get_cable_delays(station_id = 11, channels = channels_to_include)
        origin_xyz = channel_positions[0]
        ee, aa = np.meshgrid(reco_results["elevation"], reco_results["azimuth"])
        src_pos = utils.ang_to_cart(ee.flatten(), aa.flatten(), self.radius, origin_xyz)
        arrival_times = {}
        for ch in channels_to_include:
            src_pos_local = utils.to_antenna_rz_coordinates(src_pos, channel_positions[ch])
            arrival_times[ch] = ttcs[ch].get_travel_time(src_pos_local, comp = solution)

        reference_arrival_time = arrival_times[reference_ch]
        arrival_delays = {}
        for ch in channels_to_include:
            arrival_delays[ch] = arrival_times[ch] - reference_arrival_time + cable_delays[ch] - cable_delays[reference_ch]

        return arrival_delays

    def run(self, event, station, detectorpath, station_id, channels_to_include, solution):
        
        warning = 0

        dedisperse = dedisperse_new.Dedisperse()
        reco_obj = reco.Reco() 
        mappath = reco_obj.build_travel_time_maps(detectorpath, station_id, channels_to_include)
        reco_results, max_corr = reco_obj.run(event, station, detectorpath, station_id, channels_to_include, True, self.res)

        channel_times, channel_signals = dedisperse.run(event, station)
        
        channels_to_csw = channels_to_include

        reference_ch = -123456
        reference_ch_max_voltage = -1
        for ch_ID in channels_to_csw:
            this_max_voltage = np.max(channel_signals[ch_ID])
            if this_max_voltage > reference_ch_max_voltage:
                reference_ch_max_voltage = this_max_voltage
                reference_ch = ch_ID

        arrival_delays_reco = self.get_arrival_delays_reco(reco_results, channels_to_include, detectorpath, reference_ch, solution, utils.load_ttcs(mappath, channels_to_include))
    
        arrival_delays = self.get_arrival_delays_AraRoot_xcorr(
        channel_signals, channel_times, channels_to_include, reference_ch, arrival_delays_reco, solution, detectorpath, station_id, mappath
        )

        expected_signal_time = np.asarray(channel_times[reference_ch])[
            np.argmax( np.asarray(channel_signals[reference_ch]) )
        ]

        # Initialize the final CSW waveform time and voltage arrays using the
        #   reference channel's time array resized to size of the channel with
        #   the shortest waveform's waveform
        shortest_wf_ch = 123456
        shortest_wf_length = np.inf
        for ch_ID in channels_to_csw:
            if len(channel_signals[ch_ID]) < shortest_wf_length:
                shortest_wf_length = len(channel_signals[ch_ID])
                shortest_wf_ch = ch_ID

        csw_values = np.zeros((1, shortest_wf_length))
        csw_times = np.asarray(
            channel_times[reference_ch])[:shortest_wf_length]
        csw_dt = csw_times[1] - csw_times[0]

        # Roll the waveform from each channel so the starting time of each
        for ch_ID in channels_to_csw:
            values = np.asarray(channel_signals[ch_ID])
            times = np.asarray(channel_times[ch_ID]) - (arrival_delays[ch_ID]//csw_dt)*csw_dt
            
            rebinning_shift = (
                (csw_times[0] - times[0])
                % csw_dt
                # Take the remainder of the start time difference with csw_dt.
                #   If this is ~0, the waveforms have the same binning
                #   otherwise, this reveals how much to shift the waveform by.
                # For example, waveform 1 yeilds: (3.4 - 0.9) % 0.5 = 0
                #   waveform 2 yeilds (1.6 - 1.4) % 0.5 = 0.2
                #   and waveform 3 yeilds (0.9 - 0.5) % 0.5 = 0.4
            )
            if csw_dt - 0.0001 > abs(rebinning_shift) > 0.0001:
                #print(warning, "1")
                #print(csw_dt - 0.0001, rebinning_shift)
                warning += 10_00_00

            # Trim this waveform's length to match the CSW length
            if len(times) > len(csw_times):
                trim_ammount = len(times) - len(csw_times)
                if (
                    ( times[0] - csw_times[0] < 0 ) # this wf has an earlier start time than the CSW
                    and ( times[-1] - csw_times[-1] <= csw_dt/2) # this wf has a earlier or equal end time than the CSW
                ): # We need to trim from the beginning of the waveform
                    times  = times [trim_ammount:]
                    values = values[trim_ammount:]
                elif (
                    ( times[0] - csw_times[0] > -csw_dt/2) # this wf has a later or equal start time than the CSW
                    and (times[-1] - csw_times[-1] > 0) # this wf has a later end time than the CSW
                ): # we need to trim from the end of the waveform
                    times  = times [:-trim_ammount]
                    values = values[:-trim_ammount]
                elif (
                    ( times[0] - csw_times[0] < 0 ) # this wf starts earlier than the CSW
                    and ( times[-1] - csw_times[-1] > 0 ) # this wf ends later than the CSW
                ): # we need to trim from both ends of the waveform
                    leading_trimmable = np.argwhere(
                        np.round(times,5) < np.round(csw_times[0], 5) )
                    trailing_trimmable = np.argwhere(
                        np.round(times, 5) > np.round(csw_times[-1], 5) )
                    times  = times [ len(leading_trimmable) : -len(trailing_trimmable) ]
                    values = values[ len(leading_trimmable) : -len(trailing_trimmable) ]

            roll_shift_bins = (csw_times[0] - times[0]) / csw_dt
            roll_shift_time = roll_shift_bins*(times[1] - times[0])
            if abs(roll_shift_bins) % 1.0 > 0.0001:
                # roll_shift is not close to an integer. Add to the warning
                warning += 10
            roll_shift_bins = int(roll_shift_bins)
            if abs(roll_shift_bins)>len(times):
                # More waveform to roll than there is time in the waveform,
                #   so add to the warning tracker. 
                # Software triggers are so short, this sometimes occurs for them.
                #   Don't warn in this scenario.
                warning += 10_00_00_00
            if roll_shift_bins > 0 and abs(roll_shift_bins)<len(times):
                # Rolling from front to back, check that signal region isn't in the front
                if times[0] <= expected_signal_time <= times[roll_shift_bins]:
                    warning += 10_00
            elif roll_shift_bins < 0 and abs(roll_shift_bins)<len(times):
                # Rolling from back to front, check that signal region isn't in the back
                if  times[roll_shift_bins]  <= expected_signal_time <= times[-1]:
                    warning += 10_00
            rolled_wf = np.roll( values, -roll_shift_bins )
            rolled_times = np.linspace(
                times[0] + roll_shift_time,
                times[-1] + roll_shift_time,
                len(times)
            )
            
            # Add this channel's waveform to the CSW
            csw_values = np.sum( np.dstack( (csw_values[0], rolled_wf) ), axis=2)

        # Un-nest the csw. csw.shape was (1,len(csw_times)) but is now len(csw_times)
        csw_values = np.squeeze(csw_values)

        #print(snr.get_snr_single(csw_times, csw_values))
        #print(snr.get_snr(eventpath))
        
        return (csw_times, csw_values)



