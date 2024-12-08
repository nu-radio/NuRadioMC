from detector import Detector 
from propagation import TravelTimeCalculator
import numpy as np
import utils, reco_utils, preprocessing
import defs

class Reco:

    def __init__(self):
        self.z_range = (-650, 150)
        self.r_max = 1100
        self.num_pts_z = 100
        self.num_pts_r = 100
        self.ior_model = defs.ior_exp3
         

    
    def build_travel_time_maps(self, detectorpath, station_id, channels_to_include):

        det = Detector(detectorpath)
        channel_positions = det.get_channel_positions(station_id, channels_to_include)

        z_range_map = (self.z_range[0] - 1, self.z_range[1] + 1)
        r_max_map = self.r_max + 1

        mapdata = {}
        for channel, xyz in channel_positions.items():
            ttc = TravelTimeCalculator(tx_z = xyz[2],
                                   z_range = z_range_map,
                                   r_max = r_max_map,
                                   num_pts_z = 5 * self.num_pts_z,
                                   num_pts_r = 5 * self.num_pts_r)
        
            ttc.set_ior_and_solve(self.ior_model)

            mapdata[channel] = ttc.to_dict()

        return mapdata

    def run(self, event, station, detectorpath, station_id, channels_to_include, do_envelope, res):
        
        mappath = self.build_travel_time_maps(detectorpath, station_id, channels_to_include)
        
        channel_signals = {}
        channel_times = {}
        for ch in station.iter_channels():
            trace = ch.get_trace()
            times = ch.get_times()
            channel_signals[ch.get_id()] = trace 
            channel_times[ch.get_id()] = times
        
        if do_envelope:
            channel_signals = preprocessing.envelope(channel_signals)

        det = Detector(detectorpath)
        channel_positions = det.get_channel_positions(station_id = station_id, channels = channels_to_include)
        cable_delays = det.get_cable_delays(station_id = station_id, channels = channels_to_include)

        azimuth_range = (-np.pi, np.pi)
        elevation_range = (-np.pi/2, np.pi/2)

        elevation_vals = np.linspace(*elevation_range, res)
        azimuth_vals = np.linspace(*azimuth_range, res)
        ee, aa = np.meshgrid(elevation_vals, azimuth_vals)

        radius = 38 / defs.cvac
        origin_xyz = channel_positions[0]  # use PA CH0- as origin of the coordinate system

        reco = reco_utils.interferometric_reco_ang(channel_signals, channel_times, mappath,
                                               rad = radius, origin_xyz = origin_xyz, elevation_range = elevation_range, azimuth_range = azimuth_range,
                                               num_pts_elevation = res, num_pts_azimuth = res, channels_to_include = channels_to_include,
                                               channel_positions = channel_positions, cable_delays = cable_delays)
        
        maxcorr_point, maxcorr = utils.get_maxcorr_point(reco)

        return maxcorr_point, maxcorr



