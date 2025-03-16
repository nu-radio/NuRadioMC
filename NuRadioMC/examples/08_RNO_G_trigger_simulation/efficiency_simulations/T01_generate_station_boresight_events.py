"""
This file generates the input files for the electron neutrino SNR
curves simulation using a phased array. For creating a test file, run:

python T01generate_event_list.py

The file will create several folders containing neutrinos that interact at a
given angle with respect to the horizon. The

WARNING: This file needs NuRadioMC installed. https://github.com/nu-radio/NuRadioMC
"""

from __future__ import absolute_import, division, print_function
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder
import numpy as np
import os
import logging
import argparse
from NuRadioReco.detector.RNO_G import rnog_detector
import datetime as dt


parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
parser.add_argument('--station_id', type=int, default=0, help='station id')
args = parser.parse_args()

if not os.path.exists(f"data/input/station{args.station_id}"):
    os.mkdir(f"data/input/station{args.station_id}")

pos=[0,0]
z=-95.5

# If using a real station, get the station location and top and bottom antenna of PA.
if args.station_id != 0:
    det = rnog_detector.Detector(
        detector_file=None, log_level=logging.INFO,
        always_query_entire_description=False, select_stations=args.station_id)

    det.update(dt.datetime(2023, 8, 3))

    pos = det.get_absolute_position(args.station_id)
    rel0=det.get_channel(args.station_id,0)['channel_position']['position']
    rel3=det.get_channel(args.station_id,3)['channel_position']['position']
    z=pos[2]+np.mean([rel0[2],rel3[2]])
    print(f"Simulating around center x0={pos[0]:.2f}m, y0={pos[1]:.2f}m")




z12 = z*units.m
rho = 100 * units.m

obs_angles = np.linspace(-10., 10., 41) * units.deg

# Elevation of the vertex point seen from the middle point of the array
theta_file=[]
for obs_angle in obs_angles:
    
    zmin = z12 - 0.001 * units.m
    zmax = z12 
    rmin = rho
    rmax = rho + 0.1 * units.m

    # define simulation volume as a small box roughly at boresight.
    volume={
        'fiducial_xmax':100.1*units.m,
        'fiducial_xmin':99.9*units.m,
        'fiducial_zmin': zmin,
        'fiducial_zmax': zmax,
        'fiducial_ymin': -.001*units.m,
        'fiducial_ymax': 0.001*units.m,
        'x0':pos[0], 
        'y0':pos[1]

    }

    # Hacky offset to get roughly the correct view angle in simulation.
    # 52.9 from Cherencov angle and small correction from ray-tracing.
    theta_c = (52.9+.5) * units.deg
    theta_nu = 90 * units.deg - theta_c - obs_angle

    #One direction is fine.
    phimin = 0. * units.deg
    phimax = 0.1 * units.deg

    #if (theta_nu < 0):
    #    theta_nu = -theta_nu
    #    phimin = 180 * units.deg
    #    phimax = 180.1 * units.deg

    thetas = np.array([theta_nu, theta_nu + 0.01 * units.deg])
    thetamins = thetas[0:-1]
    thetamaxs = thetas[1:]

    Es = np.array([18, 18.1])
    Es = 10 ** Es
    Emins = Es[0:-1]
    Emaxs = Es[1:]
    theta_file.append(float(f'{thetamins[0] / units.deg:.2f}'))

    nevt = 1e2
    nevt_perfile = 1e2

    for thetamin, thetamax in zip(thetamins, thetamaxs):

        folderlist = []
        for ifolder, Emin, Emax in zip(range(len(Emins)), Emins, Emaxs):
            folderlist.append(
                "{:.2f}_{}_{}_{:.2e}_{:.2e}".format(thetamin / units.deg, 12, str(ifolder).zfill(2), Emin,
                                                    Emax))

        for folder, Emin, Emax in zip(folderlist, Emins, Emaxs):

            outname = f"data/input/station{args.station_id}/{folder}.hdf5"
            print(Emin / units.PeV, Emax / units.PeV)
            print(outname)

            # Random sample of e, mu, tau and random cc or nc interaction.
            generate_eventlist_cylinder(
                outname,
                nevt,
                Emin,
                Emax,
                volume,
                thetamin=thetamin,
                thetamax=thetamax,
                phimin=phimin,
                phimax=phimax,
                start_event_id=1,
                flavor=[12,14,16],
                n_events_per_file=nevt_perfile,
                deposited=True,
                interaction_type='ccnc'
            )

theta_file=theta_file[::-1]
for i in range(len(theta_file)):
    print(theta_file[i])

# Save angle to a file for command line args
np.savetxt("boresight_angles.txt", theta_file[i])
#print(np.array(theta_file)[::-1])
