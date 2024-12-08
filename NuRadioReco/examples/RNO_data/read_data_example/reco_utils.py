import utils, pickle, itertools
import numpy as np
from propagation import TravelTimeCalculator

def calc_corr_score(channel_signals, channel_times, pts, ttcs, channel_pairs_to_include, channel_positions, cable_delays,
                    comps = ["direct_ice", "direct_air", "reflected"]):
    
    scores = []
    corr_function = {}
    for (ch_a, ch_b) in channel_pairs_to_include:
        print(f"channel comparison: {ch_a}<-->{ch_b}")
        sig_a = channel_signals[ch_a]
        sig_b = channel_signals[ch_b]
        tvals_a = channel_times[ch_a]
        tvals_b = channel_times[ch_b]
        corr_function[ch_a, ch_b] = {}
        for comp in comps:
            t_ab = utils.calc_relative_time(ch_a, ch_b, src_pos = pts, ttcs = ttcs, comp = comp,
                                            channel_positions = channel_positions, cable_delays = cable_delays)
            score = utils.corr_score_batched(sig_a, sig_b, tvals_a, tvals_b, t_ab)            
            corr_function[ch_a, ch_b][comp] = [t_ab, score]
            scores.append(np.nan_to_num(score, nan = 0.0))

    return np.mean(scores, axis = 0)



def build_interferometric_map_3d(channel_signals, channel_times, channel_pairs_to_include, channel_positions, cable_delays,
                                 coord_start, coord_end, num_pts, ttcs):

    x_vals = np.linspace(coord_start[0], coord_end[0], num_pts[0])
    y_vals = np.linspace(coord_start[1], coord_end[1], num_pts[1])
    z_vals = np.linspace(coord_start[2], coord_end[2], num_pts[2])
    
    xx, yy, zz = np.meshgrid(x_vals, y_vals, z_vals, indexing = 'ij')
    pts = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis = -1)

    intmap = calc_corr_score(channel_signals, channel_times, pts, ttcs, channel_pairs_to_include,
                             channel_positions = channel_positions, cable_delays = cable_delays)
    assert len(intmap) == len(pts)
    intmap = np.reshape(intmap, num_pts, order = "C")

    return x_vals, y_vals, z_vals, intmap

# all coordinates and coordinate ranges are given in natural feet
def interferometric_reco_3d(channel_signals, channel_times, mappath,
                            coord_start, coord_end, num_pts,
                            channels_to_include, channel_positions, cable_delays):

    ttcs = utils.load_ttcs(mappath, channels_to_include)
    channel_pairs_to_include = list(itertools.combinations(channels_to_include, 2))
    x_vals, y_vals, z_vals, intmap = build_interferometric_map_3d(channel_signals, channel_times, channel_pairs_to_include,
                                                                  channel_positions = channel_positions, cable_delays = cable_delays,
                                                                  coord_start = coord_start, coord_end = coord_end, num_pts = num_pts,
                                                                  ttcs = ttcs)
       
    reco_event = {
        "x": x_vals,
        "y": y_vals,
        "z": z_vals,
        "map": intmap
    }

    return reco_event

def build_interferometric_map_ang(channel_signals, channel_times, channel_pairs_to_include, channel_positions, cable_delays,
                                  rad, origin_xyz, elevation_range, azimuth_range, num_pts_elevation, num_pts_azimuth, ttcs):

    elevation_vals = np.linspace(*elevation_range, num_pts_elevation)
    azimuth_vals = np.linspace(*azimuth_range, num_pts_azimuth)

    ee, aa = np.meshgrid(elevation_vals, azimuth_vals)
    # ang_pts = np.stack([ee.flatten(), aa.flatten()], axis = -1)

    # convert to cartesian points
    pts = utils.ang_to_cart(ee.flatten(), aa.flatten(), radius = rad, origin_xyz = origin_xyz)

    intmap = calc_corr_score(channel_signals, channel_times, pts, ttcs, channel_pairs_to_include,
                             channel_positions = channel_positions, cable_delays = cable_delays)
    assert len(intmap) == len(pts)
    intmap = np.reshape(intmap, (num_pts_elevation, num_pts_azimuth), order = "C")

    return elevation_vals, azimuth_vals, intmap

def interferometric_reco_ang(channel_signals, channel_times, mappath,
                             rad, origin_xyz, elevation_range, azimuth_range, num_pts_elevation, num_pts_azimuth,
                             channels_to_include, channel_positions, cable_delays):

    ttcs = utils.load_ttcs(mappath, channels_to_include)
    channel_pairs_to_include = list(itertools.combinations(channels_to_include, 2))
    elevation_vals, azimuth_vals, intmap = build_interferometric_map_ang(channel_signals, channel_times, channel_pairs_to_include,
                                                                         channel_positions = channel_positions, cable_delays = cable_delays,
                                                                         rad = rad, origin_xyz = origin_xyz, elevation_range = elevation_range, azimuth_range = azimuth_range,
                                                                         num_pts_elevation = num_pts_elevation, num_pts_azimuth = num_pts_azimuth, ttcs = ttcs)

    reco_event = {
        "elevation": elevation_vals,
        "azimuth": azimuth_vals,
        "radius": rad,
        "map": intmap
    }
    
    return reco_event
