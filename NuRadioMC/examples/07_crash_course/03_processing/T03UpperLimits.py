from NuRadioMC.examples.Sensitivities import E2_fluxes3 as limit
from NuRadioMC.utilities.cross_sections import get_interaction_length
from NuRadioReco.utilities import units
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json

"""
This file explains how to make an upper limit plot starting from an effective
volume simulation.

To run it, use:

python T03UpperLimits.py effective_volumes_file.json
"""

def get_E2_upper_limits(energy_centres,
                        effective_volumes,
                        upper_nevt=2.44,
                        independent_stations=1,
                        livetime=1*units.year,
                        solid_angle=4*np.pi):
    """
    This function returns an upper limit flux multiplied by the square of the energy.

    Parameters
    ----------
    energy_centres: array of floats
        Bin centre energy
    effective_volumes: array of floats
        Effective volumes
    upper_nevt: integer
        The number of events that correspond to the desired confidence level.
        For an experiment detecting no events, the 90% CL limit for the mean
        number of events is 2.44.
    independent_stations: integer
        If the effective volume file contains only one simulated station, the
        result can be multiplied by this number to get an estimate for the whole
        array. Careful! This only works if stations are independent, and they
        are not for high energies, most of the time.
    livetime: float
        Time the array is expected to take data
    solid_angle: float
        Solid angle to multiply the effective volume. By default we consider the
        whole sky (4 pi)

    Returns
    -------
    upper_flux_E2: array of floats
        Differential pper limit flux times neutrino energy squared
    """

    log_energy_centres = np.log10(energy_centres)
    log_delta_energy = log_energy_centres[1] - log_energy_centres[0]
    log_left_bins = log_energy_centres - log_delta_energy/2
    log_right_bins = log_energy_centres + log_delta_energy/2

    left_bins = 10 ** log_left_bins
    right_bins = 10 ** log_right_bins

    effective_areas = effective_volumes / get_interaction_length(energy_centres, cross_section_type='ctw')
    effective_areas *= independent_stations

    upper_flux = upper_nevt / ( effective_areas * solid_angle * livetime * (right_bins-left_bins) )
    upper_flux_E2 = upper_flux * energy_centres**2

    return upper_flux_E2

parser = argparse.ArgumentParser(description='Upper limit calculation')
parser.add_argument('volumes_file', type=str, default=None,
                    help='Path to effective volumes file')
args = parser.parse_args()

volumes_file = args.volumes_file

"""
First we open the effective volume file for the whole sky and save the
effective volumes and energies.
"""
with open(volumes_file, 'r') as f:
    fin = json.load(f)

trigger_label = "all_triggers"
Veffs = np.array( fin[trigger_label]["Veffs"] ) * units.m3
Veffs_unc = np.array( fin[trigger_label]["Veffs_uncertainty"] ) * units.m3
energy_centres = np.array( fin["energies"] ) * units.eV

"""
This function helps us produce the plot you can see on several proposals and
papers.
"""
fig, ax = limit.get_E2_limit_figure(diffuse = True,
                    show_ice_cube_EHE_limit=True,
                    show_ice_cube_HESE_fit=False,
                    show_ice_cube_HESE_data=True,
                    show_ice_cube_mu=True,
                    show_anita_I_III_limit=True,
                    show_auger_limit=True,
                    show_neutrino_best_fit=True,
                    show_neutrino_best_case=True,
                    show_neutrino_worst_case=True,
                    show_ara=True,
                    show_grand_10k=True,
                    show_grand_200k=False,
                    show_radar=False)

"""
Now we calculate the upper limits. We could have used the function add_limit
in examples.Sensitivities.E2_fluxes3.py to create the same plot. However, using
the function get_E2_upper_limits defined above helps you demistify these
upper limit calculations.

Be sure to adjust the independent_stations variable. If the simulation contains
all the stations in the array, independent_stations should be one. If the simulation
contains only one station, independent_stations should equal the total number
of stations in the array, although overlapping effective volumes will not be taken
into account this way.
"""
upper_limits_E2 = get_E2_upper_limits(energy_centres, Veffs,
                                      independent_stations=1, livetime=5*units.year)
units_flux_plot = units.GeV / units.cm2 / units.s / units.sr
plt_label, = plt.step(energy_centres/units.GeV, upper_limits_E2/units_flux_plot, where='mid',
                      linestyle='-', color='mediumorchid', linewidth=3, label='Our cool study - 5 years')

labels = [plt_label]
plt.legend(handles=labels, loc=2)
plt.show()
