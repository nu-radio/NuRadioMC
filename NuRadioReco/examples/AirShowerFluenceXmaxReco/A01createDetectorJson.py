"""
Helper script that generates the detector description for
dual-polarized antennas on a square grid. This is a generic
example for typical air-shower arrays such as AERA, Auger-Radio,
LOFAR, TUNKA-REX, GRAND, etc.
"""

import json
import copy
import numpy as np
from radiotools import helper as hp
from NuRadioReco.utilities import units

class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


antenna = "SKALA_InfFirn"
ori_EW = [0,0,90*units.deg, 90*units.deg]
ori_NW = [0,0,90*units.deg, 0]

# antenna = "LOFAR_LBA"
# ori_EW = [90 * units.deg, 135 * units.deg, 0, 0]
# ori_NW = [90 * units.deg, (135+180) * units.deg, 0, 0]

orientation_theta, orientation_phi, rotation_theta, rotation_phi = ori_NW
a1 = hp.spherical_to_cartesian(orientation_theta, orientation_phi)
a2 = hp.spherical_to_cartesian(rotation_theta, rotation_phi)
a3 = np.cross(a1, a2)
if np.linalg.norm(a3) < 0.9:
    raise AssertionError("the two vectors that define the antenna orientation are not othorgonal to each other")

ctmp = {
    "adc_n_samples": 512,
    "adc_sampling_frequency": 1,
    "channel_id": 0,
    "commission_time": "{TinyDate}:2017-11-01T00:00:00",
    "decommission_time": "{TinyDate}:2038-01-01T00:00:00",
}

stmp = {
    "commission_time": "{TinyDate}:2017-11-04T00:00:00",
    "decommission_time": "{TinyDate}:2038-01-01T00:00:00",
    "pos_altitude": 0,
    "pos_site": "LOFAR",
}

det = {}
det['channels'] = {}
det['stations'] = {}

xx = np.arange(-0.15,0.151,0.05) * units.km
yy = np.arange(-0.15,0.151,0.05) * units.km
# yy = [0]
z = 1 * units.cm

station_id = 1
channel_id_counter = 0
channel_group_counter = 0
for x in xx:
    for y in yy:
        channel_group_counter += 1
        for ori in [ori_EW, ori_NW]:
            channel_id_counter += 1
            ori_theta, ori_phi, rot_theta, rot_phi = ori
            cab_length = 0
            cab_time_delay = 0
            Tnoise = 300
            channel = copy.copy(ctmp)
            channel['ant_position_x'] = x
            channel['ant_position_y'] = y
            channel['ant_position_z'] = z
            channel['ant_type'] = antenna
            channel['channel_id'] = channel_id_counter
            channel['channel_group_id'] = channel_group_counter
            channel["ant_orientation_phi"] = ori_phi/units.deg
            channel["ant_orientation_theta"] = ori_theta/units.deg
            channel["ant_rotation_phi"] = rot_phi/units.deg
            channel["ant_rotation_theta"] = rot_theta/units.deg
            channel["cab_length"] = cab_length
            channel["cab_time_delay"] = cab_time_delay
            channel["noise_temperature"] = Tnoise
            channel["station_id"] = station_id
            det['channels']["{:d}".format(channel_id_counter)] = channel
    det['stations'][f"{station_id:d}"] = copy.copy(stmp)
    det['stations'][f"{station_id:d}"]['station_id'] = station_id
    det['stations'][f"{station_id:d}"]['pos_easting'] = 0
    det['stations'][f"{station_id:d}"]['pos_northing'] = 0

    with open(f"grid_array_{antenna}.json", 'w') as fout:
        json.dump(det, fout, indent=4, sort_keys=True, cls=NpEncoder)

