import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import numpy as np
import sys
from NuRadioReco.detector import generic_detector as detector
from NuRadioReco.utilities import units
from radiotools import helper as hp
from astropy.time import Time

det = detector.GenericDetector(json_filename=sys.argv[1], default_station=101)
det.update(Time.now())

sid = 101
l = 1 * units.m

data = []

for cid in det.get_channel_ids(sid):
    ori_zen, ori_az, rot_zen, rot_az = det.get_antenna_orientation(sid, cid)
    x, y, z = det.get_relative_position(sid, cid)

    dx, dy, dz = hp.spherical_to_cartesian(ori_zen, ori_az)
    xx = [x, x - l * dx]
    yy = [y, y - l * dy]
    zz = [z, z - l * dz]
    dx, dy, dz = hp.spherical_to_cartesian(rot_zen, rot_az)
    xx.insert(0, x + 0.5 * l * dx)
    yy.insert(0, y + 0.5 * l * dy)
    zz.insert(0, z + 0.5 * l * dz)
    print(f"channel {cid:d}, rot = {dx:.1f}, {dy:.1f}, {dz:.1f}, ({rot_zen/units.deg:.0f}, {rot_az/units.deg:.0f})")

    data.append(go.Scatter3d(x=xx, y=yy, z=zz, mode='lines+markers'))

    if "LPDA" in det.get_antenna_type(sid, cid):
        ori = hp.spherical_to_cartesian(ori_zen, ori_az)
        rot = hp.spherical_to_cartesian(rot_zen, rot_az)
        dx, dy, dz = hp.spherical_to_cartesian(ori_zen, ori_az)
        v = np.cross(ori, rot)
        xx = [x]
        yy = [y]
        zz = [z]
        xx.extend([x - l * dx + 0.5 * l * v[0], x - l * dx - 0.5 * l * v[0], x])
        yy.extend([y - l * dy + 0.5 * l * v[1], y - l * dy - 0.5 * l * v[1], y])
        zz.extend([z - l * dz + 0.5 * l * v[2], z - l * dz - 0.5 * l * v[2], z])

        data.append(go.Mesh3d(x=xx, y=yy, z=zz, opacity=0.5, color='blue',
                              alphahull=-1, delaunayaxis='x'))
        data.append(go.Mesh3d(x=xx, y=yy, z=zz, opacity=0.5, color='blue',
                              alphahull=-1, delaunayaxis='y'))
#         data.append(go.Scatter3d(x=xx, y=yy, z=zz, mode='markers'))

fig = go.Figure(data=data)
fig.update_layout(scene=dict(aspectmode='data', aspectratio=dict(x=1, y=1, z=2)))
fig.show()
