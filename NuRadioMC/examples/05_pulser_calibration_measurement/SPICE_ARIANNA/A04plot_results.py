from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import logging
import numpy as np
from scipy import signal
import argparse
from datetime import datetime
from NuRadioReco.utilities import units, io_utilities
from radiotools import plthelpers as php
# plt.switch_backend('agg')

logger = logging.getLogger("NuRadioMC.A04plot_results")
logger.setLevel(logging.WARNING)

if __name__ == "__main__":
    i = 3

    results = io_utilities.read_pickle("sim_results_{:02d}.pkl".format(i), encoding='latin1')
    d = results['depth']
    zen = np.array(results['exp'])[:, 0]
    az = np.array(results['exp'])[:, 1]

    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)
    ax.scatter(d, (np.array(results['corr_LPDA'])[:, 0] - zen) / units.deg, label='corr LPDA',
            s=10, marker='o', alpha=0.5)
    ax.scatter(d, (np.array(results['corr_dipole'])[:, 0] - zen) / units.deg, label='corr dipole',
            s=10, marker='d', alpha=0.5)
    #     ax.scatter(d, (np.array(results['time_LPDA'])[:, 0] - zen) / units.deg, label='times LPDA',
    #                s=10, marker='^', alpha=0.5)
    #     ax.scatter(d, (np.array(results['time_dipole'])[:, 0] - zen) / units.deg, label='times dipole',
    #                s=10, marker='x', alpha=0.5)
    ax.set_ylabel("zenith (reco - exp) [deg]")
    ax.legend()

    ax2.scatter(d, (np.array(results['corr_LPDA'])[:, 1] - az) / units.deg, label='corr LPDA',
                s=10, marker='o', alpha=0.5)
    ax2.scatter(d, (np.array(results['corr_dipole'])[:, 1] - az) / units.deg, label='corr dipole',
                s=10, marker='d', alpha=0.5)
    #     ax2.scatter(d, (np.array(results['time_LPDA'])[:, 1] - az) / units.deg, label='times LPDA',
    #                 s=10, marker='^', alpha=0.5)
    #     ax2.scatter(d, (np.array(results['time_dipole'])[:, 1] - az) / units.deg, label='times dipole',
    #                 s=10, marker='x', alpha=0.5)
    ax2.set_ylabel("azimuth (reco - exp) [deg]")
    ax2.set_xlabel("depth [m]")
    fig.tight_layout()
    fig.savefig("plots/results_{:02d}_zen_az.png".format(i))

    fig, ax = plt.subplots(1, 1)
    ax.scatter(d, np.array(results['pol_angle']) / units.deg,
                 s=20, marker='o', alpha=0.5)
    ax.set_ylabel("polarization angle [deg]")
    ax.set_xlabel("depth [m]")
    fig.savefig("plots/results_{:02d}_pol_angle.png".format(i))
    fig.tight_layout()




    fig, ax = php.get_histograms([(np.array(results['corr_LPDA'])[:, 0] - zen) / units.deg,
                        (np.array(results['corr_dipole'])[:, 0] - zen) / units.deg,
                        (np.array(results['time_LPDA'])[:, 0] - zen) / units.deg,
                        (np.array(results['time_dipole'])[:, 0] - zen) / units.deg,
                        (np.array(results['corr_LPDA'])[:, 1] - az) / units.deg,
                        (np.array(results['corr_dipole'])[:, 1] - az) / units.deg,
                        (np.array(results['time_LPDA'])[:, 1] - az) / units.deg,
                        (np.array(results['time_dipole'])[:, 1] - az) / units.deg],
                        bins=np.linspace(-15, 15, 30),
                        titles=['corr LPDA', 'corr_dipole', 'time_LPDA', 'time_dipole', '','','',''],
                        xlabels=[r'$\Delta$zen [deg]', r'$\Delta$zen [deg]', r'$\Delta$zen [deg]', r'$\Delta$zen [deg]',
                                r'$\Delta$az [deg]', r'$\Delta$az [deg]', r'$\Delta$az [deg]', r'$\Delta$az [deg]'])
    fig.savefig("plots/results_{:02d}_hist.png".format(i))
    plt.show()
