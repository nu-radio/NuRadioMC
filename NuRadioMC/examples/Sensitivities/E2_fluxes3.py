import matplotlib
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rc('axes', labelsize=18)
legendfontsize = 11

import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.utilities import units
from NuRadioMC.utilities import fluxes
import os

energyBinsPerDecade = 1.
plotUnitsEnergy = units.eV
plotUnitsEnergyStr = "eV"
plotUnitsFlux = units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1
DIFFUSE = True

# Unless you would like to work with the layout or the models/data from other experiments,
# you don't need to change anything below here
# --------------------------------------------------------------------
# Other planned experiments

# GRAND white paper,
# numerical values, Bustamante
GRAND_energy = np.array(([48192296.5, 67644231.1, 94947581.6, 133271428.0, 187063990.0, 262568931.0, 368550053.0, 517308507.0, 726110577.0, 1019191760.0, 1430569790.0, 2007992980.0, 2818482440.0, 3956111070.0, 5552922590.0, 7794257720.0, 10940266600.0, 15356104100.0, 21554313200.0, 30254315500.0, 42465913900.0,
                          59606499400.0, 83665567300.0, 117435636000.0, 164836371000.0, 231369543000.0, 324757606000.0, 455840043000.0, 639831498000.0, 898087721000.0, 1260584320000.0, 1769396010000.0, 2483580190000.0, 3486031680000.0, 4893104280000.0, 6868115880000.0, 9640304610000.0, 13531436400000.0, 18993151900000.0, 26659388600000.0]))

GRAND_energy *= units.GeV

GRAND_10k = np.array(([8.41513361e-08, 7.38147706e-08, 5.69225180e-08, 3.46647934e-08,
                       1.95651137e-08, 1.40651565e-08, 1.25782087e-08, 1.24621707e-08,
                       1.31123151e-08, 1.45812119e-08, 1.65528260e-08, 1.91930521e-08,
                       2.31554429e-08, 2.87477813e-08, 3.55164030e-08, 4.42563884e-08,
                       5.63965197e-08, 7.45183330e-08, 1.01159657e-07, 1.39040439e-07,
                       1.98526677e-07, 2.61742251e-07, 3.40870828e-07, 4.82745531e-07,
                       6.55876763e-07, 9.07706655e-07, 1.67125879e-06, 1.76142511e-05,
                       2.55022320e-04, 1.88371074e-03, 6.71431813e-03, 1.14286198e-02,
                       1.14294614e-02, 1.72447830e-02, 7.48579143e-02, 3.31883351e-01,
                       8.57786094e-01, 1.24824516e+00, 1.42294586e+00, 1.80135089e+00]))

GRAND_10k *= (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
# GRAND_10k /= 2 #halfdecade bins
GRAND_10k *= energyBinsPerDecade
# The expected sensitivities for GRAND are given for 3 years, rescaling them to 10 years
GRAND_10k *= 3 / 10

GRAND_200k = np.array(([4.26219753e-09, 3.58147708e-09, 2.75670137e-09, 1.85254042e-09,
                        1.13825106e-09, 7.70141315e-10, 6.51758930e-10, 6.35878242e-10,
                        6.69261628e-10, 7.37439217e-10, 8.38784832e-10, 9.81688683e-10,
                        1.18493794e-09, 1.45699379e-09, 1.80867621e-09, 2.26948852e-09,
                        2.91952068e-09, 3.86790849e-09, 5.24530715e-09, 7.31211288e-09,
                        9.98848945e-09, 1.33523293e-08, 1.80893102e-08, 2.46582187e-08,
                        3.41054825e-08, 5.39140368e-08, 3.36553610e-07, 4.57179717e-06,
                        3.59391218e-05, 1.47550853e-04, 3.33777479e-04, 4.92873322e-04,
                        6.68381070e-04, 1.72553598e-03, 7.06643413e-03, 2.10754560e-02,
                        4.06319101e-02, 5.88162853e-02, 7.45423652e-02, 8.83700084e-02]))

GRAND_200k *= (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
# GRAND_200k /= 2 #halfdecade bins
GRAND_200k *= energyBinsPerDecade
# The expected sensitivities for GRAND are given for 3 years, rescaling them to 10 years
GRAND_200k *= 3 / 10

# RADAR proposed from https://arxiv.org/pdf/1710.02883.pdf

Radar = np.array(([
    (1.525e+01, 6.870e-09, 3.430e-07),
    (1.575e+01, 9.797e-10, 3.113e-08),
    (1.625e+01, 4.728e-09, 1.928e-07),
    (1.675e+01, 6.359e-09, 3.706e-07),
    (1.725e+01, 9.128e-09, 8.517e-07),
    (1.775e+01, 1.619e-08, 1.835e-06),
    (1.825e+01, 2.995e-08, 2.766e-06),
    (1.875e+01, 5.562e-08, 8.253e-06),
    (1.925e+01, 1.072e-07, 1.849e-05)]))

Radar[:, 0] = 10 ** Radar[:, 0] * units.eV
Radar[:, 1] *= (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
Radar[:, 1] /= 2  # halfdecade bins
Radar[:, 1] *= energyBinsPerDecade
Radar[:, 2] *= (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
Radar[:, 2] /= 2  # halfdecade bins
Radar[:, 2] *= energyBinsPerDecade
# --------------------------------------------------------------------
# Published data and limits

# IceCube
# log (E^2 * Phi [GeV cm^02 s^-1 sr^-1]) : log (E [Gev])
# Phys Rev D 98 062003 (2018)
# Numbers private correspondence Shigeru Yoshida
ice_cube_limit = np.array(([
    (6.199999125, -7.698484687),
    (6.299999496, -8.162876678),
    (6.400000617, -8.11395291),
    (6.500000321, -8.063634144),
    (6.599999814, -8.004841781),
    (6.699999798, -7.944960162),
    (6.799999763, -7.924197388),
    (6.899999872, -7.899315263),
    (7.299999496, -7.730561153),
    (7.699999798, -7.670680637),
    (8.100001583, -7.683379711),
    (8.500000321, -7.748746801),
    (8.899999872, -7.703060304),
    (9.299999496, -7.512907553),
    (9.699999798, -7.370926525),
    (10.10000158, -7.134626026),
    (10.50000032, -6.926516638),
    (10.89999987, -6.576523031)
]))

ice_cube_limit[:, 0] = 10 ** ice_cube_limit[:, 0] * units.GeV
ice_cube_limit[:, 1] = 10 ** ice_cube_limit[:, 1] * (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
ice_cube_limit[:, 1] *= energyBinsPerDecade

# Fig. 2 from PoS ICRC2017 (2018) 981
# IceCube preliminary
# E (GeV); E^2 dN/dE (GeV cm^-2 s-1 sr-1); yerror down; yerror up

# HESE 6 years
# ice_cube_hese = np.array(([
#
#     (6.526e+04,  2.248e-08,  9.96e-9,  1.123e-8),
#     (1.409e+05,  2.692e-08,  5.91e-9,  7.56e-9),
#     (3.041e+05,  7.631e-09,  3.746e-9, 4.61e-9),
#     (6.644e+05,  2.022e-09,  7.03e-10, 0.),
#     (1.434e+06,  5.205e-09,  3.183e-9,  4.57e-9),
#     (3.096e+06,  4.347e-09,  3.142e-9,  5.428e-9),
#     (6.684e+06,  1.544e-09,  5.37e-10, 0.),
#     (1.46e+07,  4.063e-09,   1.353e-9, 0.),
#     (3.153e+07,  6.093e-09,  2.03e-9,  0.),
#     (6.806e+07,  1.046e-08,  3.641e-9, 0.)
# ]))
# ice_cube_hese[:, 0] = ice_cube_hese[:, 0] * units.GeV
# ice_cube_hese[:, 1] = ice_cube_hese[:, 1] * (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
# ice_cube_hese[:, 1] *= 3
# ice_cube_hese[:, 2] = ice_cube_hese[:, 2] * (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
# ice_cube_hese[:, 2] *=  3
# ice_cube_hese[:, 3] = ice_cube_hese[:, 3] * (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
# ice_cube_hese[:, 3] *= 3

# HESE 8 years
# log(E (GeV)); log(E^2 dN/dE (GeV cm^-2 s-1 sr-1)); format x y -dy +dy
ice_cube_hese = np.array(([
(4.78516, 	 -7.63256, 		0.223256, 	0.167442),
(5.10938, 	 -7.66977, 		0.139535, 	0.102326),
(5.42969, 	 -8.36744, 		0.930233, 	0.297674),
(5.75391, 	 -8.51628, 		0.2, 	0.),
(6.07813, 	 -8.38605, 		0.604651, 	0.288372),
(6.39844, 	 -8.35814, 		0.455814, 	0.334884),
(6.72266, 	 -9.0, 		0.2 , 	0)
]))

# get uncertainties in right order
ice_cube_hese[:, 2] = 10 ** ice_cube_hese[:, 1] - 10 ** (ice_cube_hese[:, 1] - ice_cube_hese[:, 2])
ice_cube_hese[:, 3] = 10 ** (ice_cube_hese[:, 1] + ice_cube_hese[:, 3]) - 10 ** ice_cube_hese[:, 1]

ice_cube_hese[:, 0] = 10 ** ice_cube_hese[:, 0] * units.GeV
ice_cube_hese[:, 1] = 10 ** ice_cube_hese[:, 1] * (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
ice_cube_hese[:, 1] *= 3

ice_cube_hese[:, 2] *= (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
ice_cube_hese[:, 2] *= 3

ice_cube_hese[:, 3] *= (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
ice_cube_hese[:, 3] *= 3

# Ice cube
# ice cube nu_mu data points 9.5 years analysis
nu_mu_data = np.array([[4.64588, -7.69107, -7.87555, -7.5549, ],
                       [5.44266, -7.95022, -8.06881, -7.85359, ],
                       [6.25755, -8.51245, -8.8287, -8.29283],
                       [7.29276, -8.40264, 0, 0]])
# nu_mu_data[:, 2] = 10 ** nu_mu_data[:, 1] - 10 ** (nu_mu_data[:, 1] - nu_mu_data[:, 2])
# nu_mu_data[:, 3] = 10 ** (nu_mu_data[:, 1] + nu_mu_data[:, 3]) - 10 ** nu_mu_data[:, 1]
# convert energy to correct units
nu_mu_data[:, 1:] = (10 ** nu_mu_data[:, 1:]) * 3 * (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)  # convert from single flavor to all flavor limit
nu_mu_data[:, 2] = np.abs(nu_mu_data[:, 1] - nu_mu_data[:, 2])
nu_mu_data[:, 3] = np.abs(nu_mu_data[:, 1] - nu_mu_data[:, 3])

nu_mu_data[-1, 3] = 0
nu_mu_data[-1, 2] = 2e-9 * 3 * (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
nu_mu_data[:, 0] = 10 ** nu_mu_data[:, 0] * units.GeV

# ApJ slope=-2.13, offset=0.9 (https://arxiv.org/pdf/1607.08006.pdf)
# ICR2017 slope=-2.19, offset=1.01 (https://pos.sissa.it/301/1005/)
# ICRC2019 slope=2.28, offset=1.44
# 9.5 years analysis 2.37+0.08-0.09, offset 1.36 + 0.24 - 0.25, Astrophysical normalization @ 100TeV: 1.36 × 10−8GeV−1cm−2s−1sr−1

# using ICRC2019 results for now which are the last published results. No Piecewise-Unfolding was already done at that time, so we don't have "data points".
nu_mu_slope = -2.28
nu_mu_slope_up = -(2.28 + 0.08)
nu_mu_slope_down = -(2.28 - 0.09)
nu_mu_offset = 1.44
nu_mu_offset_up = 1.44 + 0.25
nu_mu_offset_down = 1.44 - 0.24
nu_mu_show_data_points = False


def ice_cube_nu_fit(energy, slope=nu_mu_slope, offset=nu_mu_offset):
    flux = 3 * offset * (energy / (100 * units.TeV)) ** slope * 1e-18 * \
        (units.GeV ** -1 * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
    return flux


def get_ice_cube_mu_range():
    energy = np.arange(1e2, 5e6, 1e5) * units.GeV
#     upper = np.maximum(ice_cube_nu_fit(energy, offset=0.9, slope=-2.), ice_cube_nu_fit(energy, offset=1.2, slope=-2.13)) # APJ
#     upper = np.maximum(ice_cube_nu_fit(energy, offset=1.01, slope=-2.09),
#                     ice_cube_nu_fit(energy, offset=1.27, slope=-2.19), ice_cube_nu_fit(energy, offset=1.27, slope=-2.09))  # ICRC
    slope = nu_mu_slope
    slope_up = nu_mu_slope_up
    slope_down = nu_mu_slope_down
    offset_up = nu_mu_offset_up
    offset_down = nu_mu_offset_down
    upper = np.maximum(ice_cube_nu_fit(energy, offset=offset_up, slope=slope_up),
#                     ice_cube_nu_fit(energy, offset=offset_up, slope=slope),
                    ice_cube_nu_fit(energy, offset=offset_up, slope=slope_down))  # 9.5 years
    upper *= energy ** 2
#     lower = np.minimum(ice_cube_nu_fit(energy, offset=0.9, slope=-2.26),
#                        ice_cube_nu_fit(energy, offset=0.63, slope=-2.13)) #ApJ
#     lower = np.minimum(ice_cube_nu_fit(energy, offset=1.01, slope=-2.29),
#                        ice_cube_nu_fit(energy, offset=0.78, slope=-2.19))  # ICRC
    lower = np.minimum(ice_cube_nu_fit(energy, offset=offset_down, slope=slope_up),
#                        ice_cube_nu_fit(energy, offset=offset_down, slope=slope),
                       ice_cube_nu_fit(energy, offset=offset_down, slope=slope_down))  # 9.5 years
    lower *= energy ** 2
    return energy, upper, lower


def get_ice_cube_hese_range():
    energy = np.arange(1e5, 5e6, 1e5) * units.GeV
    upper = np.maximum(ice_cube_nu_fit(energy, offset=2.46, slope=-2.63),
                       ice_cube_nu_fit(energy, offset=2.76, slope=-2.92))
    upper *= energy ** 2
    lower = np.minimum(ice_cube_nu_fit(energy, offset=2.46, slope=-3.25),
                       ice_cube_nu_fit(energy, offset=2.16, slope=-2.92))
    lower *= energy ** 2
    return energy, upper, lower

# IceCube Glashow
# Paper: https://doi.org/10.1038/s41586-021-03256-1
# Dataset: https://doi.org/10.21234/gr2021
# https://icecube.wisc.edu/data-releases/2021/03/icecube-data-for-the-first-glashow-resonance-candidate/
# NB: the csv file gives per-flavor, but we want all flavor, so multiply by 3


i3_glashow_data = np.genfromtxt(os.path.join(os.path.dirname(__file__), 'data', "icecube_glashow.csv"),
    skip_header=2, delimiter=',', names=['E_min', 'E_max', 'y', 'y_lower', 'y_upper'])
i3_glashow_emin = i3_glashow_data['E_min'] * units.GeV / plotUnitsEnergy
i3_glashow_emax = i3_glashow_data['E_max'] * units.GeV / plotUnitsEnergy
i3_glashow_y = 3. * i3_glashow_data['y'] * (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1) * 1E-8 / plotUnitsFlux
i3_glashow_y_lower = 3. * i3_glashow_data['y_lower'] * (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1) * 1E-8 / plotUnitsFlux
i3_glashow_y_upper = 3. * i3_glashow_data['y_upper'] * (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1) * 1E-8 / plotUnitsFlux

'''
Regarding ANITA Limits
# ====================================================

ANITA uses a *super* unusual differential limit bin width.

A limit is generally given by:

     E dN                           Sup
---------------  =   -----------------------------------
dE dA dOmega dt       T  * Efficiency * Aeff * BinWidth

For most experiments, BinWidth is transformed into log space for convenience:
               BinWidth = LN(10) * dlog10(E)
And then a decade wide binning is assumed: dlog10(E) = 1

But, in ANITA, they set BinWidth = 4 (!!!!!!!!!!!!!!)
See eq D1 of the ANITA-III paper.
"... the factor Delta = 4 follows the normalization convention..."

This means that the ANITA limit is a factor of LN(10)/4 too strong
when naively compared to other experiments, e.g. IceCube.
So, below, we multply by by 4/LN(10) to fix the bin width.

'''

# ANITA I - III
# https://arxiv.org/abs/1803.02719
# Phys. Rev. D 98, 022001 (2018)
anita_limit = np.array(([
    (9.94e17, 3.79e-14 * 9.94e17 / 1e9),
    (2.37e18, 2.15e-15 * 2.37e18 / 1e9),
    (5.19e18, 2.33e-16 * 5.19e18 / 1e9),
    (1.10e19, 3.64e-17 * 1.10e19 / 1e9),
    (3.55e19, 4.45e-18 * 3.55e19 / 1e9),
    (1.11e20, 9.22e-19 * 1.11e20 / 1e9),
    (4.18e20, 2.97e-19 * 4.18e20 / 1e9),
    (9.70e20, 1.62e-19 * 9.70e20 / 1e9)
]))
anita_limit[:, 0] *= units.eV
anita_limit[:, 1] *= (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
anita_limit[:, 1] *= (4 / np.log(10))  # see discussion above about strange anita binning
anita_limit[:, 1] *= energyBinsPerDecade

# ANITA I - IV
# https://arxiv.org/abs/1902.04005
# Phys. Rev. D 99, 122001 (2019)
# NB: The ANITA I-IV is indeed weaker than the ANITA I-III limit (!!)
# The reason is not understood, but can be seen easily comparing the two limits side-by-side
anita_i_iv_limit = np.array(([
    (1.000e+18, 3.1098E+04),
    (3.162e+18, 3.7069E+03),
    (1.000e+19, 6.0475E+02),
    (3.162e+19, 2.5019E+02),
    (1.000e+20, 1.4476E+02),
    (3.162e+20, 1.5519E+02),
    (1.000e+21, 2.0658E+02)
]))
anita_i_iv_limit[:, 0] *= units.eV
anita_i_iv_limit[:, 1] *= (units.eV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
anita_i_iv_limit[:, 1] *= (4 / np.log(10))  # see discussion above about strange anita binning
anita_i_iv_limit[:, 1] *= energyBinsPerDecade

'''
Regarding Auger Limits
# ====================================================

Auger publishes a limit that only applies to a single flavor.
So, to make it an all-flavor limit, the limit must be multiplied by 3.
Because this is a limit, multiplying by 3 makes the limit *weaker*.
Also, Auger uses half decade bins, so that must be corrected to a single decade.
The net factor of 3/2, on a log-log plot, leaves the limit's position (relative
to other experiments) essentially unchanged.

'''

# Auger neutrino limit (2019, 14.7 years)
# JCAP 10 (2019) 022
# https://arxiv.org/abs/1906.07422
auger_limit = np.array(([
    (5.677E+16, 9.398E-08),
    (1.771E+17, 2.298E-08),
    (5.677E+17, 1.467E-08),
    (1.771E+18, 1.881E-08),
    (5.677E+18, 3.382E-08),
    (1.771E+19, 7.179E-08),
    (5.677E+19, 1.725E-07),
    (1.771E+20, 4.412E-07)
]))
auger_limit[:, 0] *= units.eV
auger_limit[:, 1] *= (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
auger_limit[:, 1] /= 2  # half-decade binning
auger_limit[:, 1] *= 3  # correction for 3 flavors
auger_limit[:, 1] *= energyBinsPerDecade

# ARA Published 2sta x 1yr analysis level limit:
ara_1year = np.array((
[9.80923e+15, 3.11547e+16, 9.79329e+16, 3.1057e+17, 9.75635e+17,
3.0924e+18, 9.80103e+18, 3.07732e+19, 9.75028e+19],
[0.000166106, 1.24595e-05, 4.06271e-06, 2.04351e-06, 1.48811e-06,
1.42649e-06, 1.50393e-06, 2.10936e-06, 3.25384e-06]
))

ara_1year = ara_1year.T

ara_1year[:, 0] *= units.eV
ara_1year[:, 1] *= (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
ara_1year[:, 1] /= 1  # binning is dLogE = 1
ara_1year[:, 1] *= energyBinsPerDecade

# Analysis from https://doi.org/10.1103/PhysRevD.102.043021  https://arxiv.org/abs/1912.00987
# 2 stations (A2 and A3), approx 1100 days of livetime per station
ara_4year_E, ara_4year_limit, t1, t2 = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', "limit_a23.txt"), unpack=True)
ara_4year_E *= units.eV
ara_4year_limit *= units.eV * units.cm ** -2 * units.second ** -1 * units.sr ** -1
ara_4year_limit *= energyBinsPerDecade

# ARA 2023 projection
'''
This estimate is built by using the actual recorded livetime for the ARA stations 
through June 2021. Specifically:
1747 days of A1
5627 days of A2 + A3 + A4
826 days of A5

And then adding projected livetime
A1: 7/12 of a year for 2021, then 1 year of 2022, and 1 year of 2023
A2: no more data for 2021, no data for 2022, 1 year of 2023
A3: 7/12 of a year for 2021, then 1 year of 2022, and 1 year of 2023
A4: no more data for 2021, no data for 2022, 1 year of 2023
A5: 7/12 of a year for 2021, then 1 year of 2022, and 1 year of 2023

We do include different effective areas for A1, A2/3/4, and A5,
since A1 is smaller (only being at 100m), while A5 is larger (having the phased array).

We also included the trigger level and analysis level estimate.
The analysis level option assumes the A2 analysis efficiency for stations A1-4,
and a (preliminary) analysis efficiency estimate from the phased-array analysis.

'''
ara_2023_E_TL, ara_2023_limit_TL, t1, t2 = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', "limit_ara_2023_projected_trigger.txt"), unpack=True)
ara_2023_E_TL *= units.GeV
ara_2023_limit_TL *= units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1
ara_2023_limit_TL *= energyBinsPerDecade
ara_2023_limit_TL *= 2.44  # convert to 90%CL limit to be comparable with information from other experiments

ara_2023_E, ara_2023_limit, t1, t2 = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', "limit_ara_2023_projected_analysis.txt"), unpack=True)
ara_2023_E *= units.GeV
ara_2023_limit *= units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1
ara_2023_limit *= energyBinsPerDecade
ara_2023_limit *= 2.44  # convert to 90%CL limit to be comparable with information from other experiments

ARIANNA_HRA = np.array([[1.00000003e+07, 3.16228005e+07, 9.99999984e+07, 3.16227997e+08,
                         9.99999984e+08, 3.16228010e+09, 9.99999998e+09, 3.16228010e+10,
                         1.00000002e+11, 3.16227988e+11, 1.00000002e+12],
                         [8.66913580e-06, 4.31024784e-06, 3.02188396e-06, 1.95297917e-06,
                          1.67624432e-06, 2.09537200e-06, 2.90309617e-06, 4.41176250e-06,
                          7.49194972e-06, 1.33386048e-05, 2.57394786e-05]])
ARIANNA_HRA = ARIANNA_HRA.T
ARIANNA_HRA[:, 0] *= units.GeV
ARIANNA_HRA[:, 1] /= 1
ARIANNA_HRA[:, 1] *= (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
ARIANNA_HRA[:, 1] *= energyBinsPerDecade

from scipy.interpolate import interp1d


def get_TAGZK_flux(energy):
    """
    GZK neutrino flux from TA best fit from D. Bergmann privat communications
    """

    TA_data = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', "TA_combined_fit_m3.txt"))
    E = TA_data[:, 0] * units.GeV
    f = TA_data[:, 1] * plotUnitsFlux / E ** 2
    get_TAGZK_flux = interp1d(E, f, bounds_error=False, fill_value="extrapolate")
    return get_TAGZK_flux(energy)


def get_TAGZK_flux_ICRC2021(energy):
    """
    GZK neutrino flux from TA best fit ICRC2021
    https://pos.sissa.it/395/338/
    """
    TA_data = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', "TA_GZKprediction_ICRC2021.txt"))
    E = TA_data[:, 0] * units.GeV
    f = TA_data[:, 1] * plotUnitsFlux / E ** 2
    get_TAGZK_flux = interp1d(E, f, bounds_error=False, fill_value="extrapolate")
    return get_TAGZK_flux(energy)


def get_proton_10(energy):
    """
    10% proton flux at source for astrophysical parameters determined by Auger data, by van Vliet et al.
    """
    vanVliet_reas = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', "ReasonableNeutrinos1.txt"))
    E = vanVliet_reas[0, :] * units.GeV
    f = vanVliet_reas[1, :] * plotUnitsFlux / E ** 2
    getE = interp1d(E, f, bounds_error=False, fill_value="extrapolate")
    return getE(energy)


def get_GZK_Auger_best_fit(energy):
    Heinze_band = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', "talys_neu_bands.out"))
    E = Heinze_band[:, 0] * units.GeV
    f = Heinze_band[:, 1] / units.GeV / units.cm ** 2 / units.s / units.sr
    getE = interp1d(E, f, bounds_error=False, fill_value="extrapolate")
    return getE(energy)


def get_E2_limit_figure(diffuse=True,
                        show_ice_cube_EHE_limit=True,
                        show_ice_cube_HESE_data=True,
                        show_ice_cube_HESE_fit=True,
                        show_ice_cube_mu=True,
                        show_icecube_glashow=True,
                        show_anita_I_III_limit=False,
                        show_anita_I_IV_limit=True,
                        show_auger_limit=True,
                        show_ara=True,
                        show_ara_2023=False,
                        show_ara_2023_TL=False,
                        show_arianna=True,
                        show_neutrino_best_fit=True,
                        show_neutrino_best_case=True,
                        show_neutrino_worst_case=True,
                        show_grand_10k=True,
                        show_grand_200k=False,
                        show_radar=False,
                        show_Heinze=True,
                        show_TA=False,
                        show_TA_nominal=False,
                        show_TA_ICRC2021=False,
                        show_RNOG=False,
                        show_IceCubeGen2_whitepaper=False,
                        show_IceCubeGen2_ICRC2021=False,
                        shower_Auger=True,
                        show_ara_1year=False,
                        show_prediction_arianna_200=False):

    # Limit E2 Plot
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Neutrino Models
    # Version for a diffuse flux and for a source dominated flux
    if diffuse:
        legends = []
        # TA combined fit
        if(show_TA):
            TA_data_low = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', "TA_combined_fit_low_exp_uncertainty.txt"))
            TA_data_high = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', "TA_combined_fit_high_exp_uncertainty.txt"))
            TA_m3 = ax.fill_between(TA_data_low[:, 0] * units.GeV / plotUnitsEnergy,
                                     TA_data_low[:, 1], TA_data_high[:, 1],
                              label=r'UHECRs TA combined fit (1$\sigma$), Bergman et al.', color='C0', alpha=0.5, zorder=-1)
            legends.append(TA_m3)
        if(show_TA_nominal):
            TA_data = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', "TA_combined_fit_m3.txt"))
            E = TA_data[:, 0] * units.GeV
            f = TA_data[:, 1] * plotUnitsFlux
            TA_nominal, = ax.plot(E / plotUnitsEnergy, f / plotUnitsFlux, "k-.", label="UHECRs TA combined fit, Bergman et al.")
            legends.append(TA_nominal)
        if(show_TA_ICRC2021):
            TA_data = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', "TA_GZKprediction_ICRC2021.txt"))
            E = TA_data[:, 0] * units.GeV
            f = TA_data[:, 1] * plotUnitsFlux
            TA_nominal, = ax.plot(E / plotUnitsEnergy, f / plotUnitsFlux, "k-.", label="UHECRs TA combined fit, Bergman et al.")
            legends.append(TA_nominal)
        if(shower_Auger):

            vanVliet_max_1 = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', "MaxNeutrinos1.txt"))
            vanVliet_max_2 = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', "MaxNeutrinos2.txt"))
            vanVliet_reas = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', "ReasonableNeutrinos1.txt"))

            vanVliet_max = np.maximum(vanVliet_max_1[1, :], vanVliet_max_2[1, :])

            prot10, = ax.plot(vanVliet_reas[0, :] * units.GeV / plotUnitsEnergy, vanVliet_reas[1, :],
                              label=r'10% protons in UHECRs (Auger), m=3.4, van Vliet et al.', linestyle='--', color='k')

            prot = ax.fill_between(vanVliet_max_1[0, :] * units.GeV / plotUnitsEnergy, vanVliet_max,
                                   vanVliet_reas[1, :] / 50, color='0.9', label=r'allowed from UHECRs (Auger), van Vliet et al.', zorder=-2)
            legends.append(prot10)
            legends.append(prot)

        if(show_Heinze):
            Heinze_band = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', "talys_neu_bands.out"))
#             best_fit, = ax.plot(Heinze_band[:, 0] * units.GeV / plotUnitsEnergy, Heinze_band[:, 1] * Heinze_band[:, 0] ** 2, c='k',
#                                 label=r'UHECR (Auger) combined fit, Heinze et al.', linestyle='-.')

#             Auger_bestfit = ax.fill_between(Heinze_band[:, 0],
#                                      Heinze_band[:, 2] * Heinze_band[:, 0] ** 2, Heinze_band[:, 3] * Heinze_band[:, 0] ** 2,
#                               label=r'UHECRs Auger combined fit, Heinze et al.', color='C1', alpha=0.5, zorder=1)

            Heinze_evo = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', "talys_neu_evolutions.out"))
            Auger_bestfit = ax.fill_between(Heinze_evo[:, 0] * units.GeV / plotUnitsEnergy,
                                     Heinze_evo[:, 3] * Heinze_band[:, 0] ** 2, Heinze_evo[:, 4] * Heinze_band[:, 0] ** 2,
                              label=r'UHECRs Auger combined fit (3$\sigma$), Heinze et al.', color='C1', alpha=0.5, zorder=1)

#             Heinze_evo = np.loadtxt(os.path.join(os.path.dirname(__file__), "talys_neu_evolutions.out"))
#             best_fit_3s, = ax.plot(Heinze_evo[:, 0] * units.GeV / plotUnitsEnergy, Heinze_evo[:, 6] * Heinze_evo[:, 0] **
#                             2, color='0.5', label=r'UHECR (Auger) combined fit + 3$\sigma$, Heinze et al.', linestyle='-.')
            legends.append(Auger_bestfit)
#             legends.append(best_fit_3s)

        first_legend = plt.legend(handles=legends, loc=4, fontsize=legendfontsize, handlelength=4)

        plt.gca().add_artist(first_legend)
    else:
        tde = np.loadtxt(os.path.join(os.path.dirname(__file__), "TDEneutrinos.txt"))
        ll_grb = np.loadtxt(os.path.join(os.path.dirname(__file__), "LLGRBneutrinos.txt"))
        pulsars = np.loadtxt(os.path.join(os.path.dirname(__file__), "Pulsar_Fang+_2014.txt"))
        clusters = np.loadtxt(os.path.join(os.path.dirname(__file__), "cluster_Fang_Murase_2018.txt"))

        # Fang & Metzger
        data_ns_merger = np.array((
        [164178.9149064658, 6.801708660714134e-11],
        [336720.74740929523, 1.4132356758632395e-10],
        [835305.4165187279, 3.649486484772094e-10],
        [2160958.1870687287, 9.239856704429993e-10],
        [8002898.345863899, 3.085108101843864e-9],
        [20681309.183352273, 6.161686112812425e-9],
        [61887155.98482043, 1.0930162266889215e-8],
        [132044526.30261868, 1.3058095134564553e-8],
        [253159530.43095005, 1.0506528126116853e-8],
        [436411840.74101496, 6.5380862245683814e-9],
        [635712891.7663972, 3.910881690144994e-9],
        [944515747.1734984, 1.773891442500038e-9],
        [1211896737.260516, 9.059026812485584e-10],
        [1586152410.074502, 3.578063705414756e-10],
        [1948716060.6917415, 1.358461111073226e-10],
        [2344547243.608689, 5.26053655617631e-11]))

        data_ns_merger[:, 0] *= units.GeV
        data_ns_merger[:, 1] *= units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1

        ns_merger, = ax.plot(data_ns_merger[:, 0] / plotUnitsEnergy, data_ns_merger[:, 1] / plotUnitsFlux, color='palevioletred', label='NS-NS merger, Fang & Metzger 1707.04263', linestyle=(0, (3, 5, 1, 5)))

        ax.fill_between(tde[:, 0] * units.GeV / plotUnitsEnergy, tde[:, 2] * 3, tde[:, 3] * 3, color='thistle', alpha=0.5)
        p_tde, = ax.plot(tde[:, 0] * units.GeV / plotUnitsEnergy, tde[:, 1] * 3, label="TDE, Biehl et al. (1711.03555)", color='darkmagenta', linestyle=':', zorder=1)

        ax.fill_between(ll_grb[:, 0] * units.GeV / plotUnitsEnergy, ll_grb[:, 2] * 3, ll_grb[:, 3] * 3, color='0.8')
        p_ll_grb, = ax.plot(ll_grb[:, 0] * units.GeV / plotUnitsEnergy, ll_grb[:, 1] * 3, label="LLGRB, Boncioli et al. (1808.07481)", linestyle='-.', c='k', zorder=1)

        p_pulsar = ax.fill_between(pulsars[:, 0] * units.GeV / plotUnitsEnergy, pulsars[:, 1], pulsars[:, 2], label="Pulsar, Fang et al. (1311.2044)", color='wheat', alpha=0.5)
        p_cluster, = ax.plot(clusters[:, 0] * units.GeV / plotUnitsEnergy, clusters[:, 1], label="Clusters, Fang & Murase, (1704.00015)", color="olive", zorder=1, linestyle=(0, (5, 10)))

        first_legend = plt.legend(handles=[p_tde, p_ll_grb, p_pulsar, p_cluster, ns_merger], loc=3, fontsize=legendfontsize, handlelength=4)

        plt.gca().add_artist(first_legend)

    #-----------------------------------------------------------------------

    if show_grand_10k:
        ax.plot(GRAND_energy / plotUnitsEnergy, GRAND_10k / plotUnitsFlux, linestyle=":", color='saddlebrown')
        if energyBinsPerDecade == 2:
            ax.annotate('GRAND 10k',
                            xy=(0.9e10 * units.GeV / plotUnitsEnergy, 2e-8), xycoords='data',
                            horizontalalignment='left', color='saddlebrown', rotation=50, fontsize=legendfontsize)
        else:
            ax.annotate('GRAND 10k',
                xy=(1.2e19 * units.eV / plotUnitsEnergy, 5e-8), xycoords='data',
                horizontalalignment='left', va="top", color='saddlebrown', rotation=47, fontsize=legendfontsize)

    if show_grand_200k:
        ax.plot(GRAND_energy / plotUnitsEnergy, GRAND_200k / plotUnitsFlux, linestyle=":", color='saddlebrown')
        ax.annotate('GRAND 200k',
                    xy=(1e10 * units.GeV / plotUnitsEnergy, 6e-9), xycoords='data',
                    horizontalalignment='left', color='saddlebrown', rotation=40, fontsize=legendfontsize)
    if show_radar:
        ax.fill_between(Radar[:, 0] / plotUnitsEnergy, Radar[:, 1] / plotUnitsFlux,
                        Radar[:, 2] / plotUnitsFlux, facecolor='None', hatch='x', edgecolor='0.8')
        ax.annotate('Radar',
                    xy=(1e9 * units.GeV / plotUnitsEnergy, 4.5e-8), xycoords='data',
                    horizontalalignment='left', color='0.7', rotation=45, fontsize=legendfontsize)

    if show_ice_cube_EHE_limit:
        ax.plot(ice_cube_limit[2:, 0] / plotUnitsEnergy, ice_cube_limit[2:, 1] / plotUnitsFlux, color='dodgerblue')
        if energyBinsPerDecade == 2:
            ax.annotate('IceCube',
                    xy=(0.6e7 * units.GeV / plotUnitsEnergy, 2e-8), xycoords='data',
                    horizontalalignment='center', color='dodgerblue', rotation=0, fontsize=legendfontsize)
        else:
            ax.annotate('IceCube',
                    xy=(3e6 * units.GeV / plotUnitsEnergy, 3e-8), xycoords='data',
                    horizontalalignment='center', color='dodgerblue', rotation=0, fontsize=legendfontsize)

    if show_ice_cube_HESE_data:
        # data points
        uplimit = np.copy(ice_cube_hese[:, 3])
        uplimit[np.where(ice_cube_hese[:, 3] == 0)] = 1
        uplimit[np.where(ice_cube_hese[:, 3] != 0.)] = 0

        ax.errorbar(ice_cube_hese[:, 0] / plotUnitsEnergy, ice_cube_hese[:, 1] / plotUnitsFlux, yerr=ice_cube_hese[:, 2:].T / plotUnitsFlux, uplims=uplimit, color='dodgerblue', marker='o', ecolor='dodgerblue', linestyle='None', zorder=3)

    if show_ice_cube_HESE_fit:
        ice_cube_hese_range = get_ice_cube_hese_range()
        ax.fill_between(ice_cube_hese_range[0] / plotUnitsEnergy, ice_cube_hese_range[1] / plotUnitsFlux,
                        ice_cube_hese_range[2] / plotUnitsFlux, hatch='//', edgecolor='dodgerblue', facecolor='azure', zorder=2)
        plt.plot(ice_cube_hese_range[0] / plotUnitsEnergy, ice_cube_nu_fit(ice_cube_hese_range[0],
                                                                           offset=2.46, slope=-2.92) * ice_cube_hese_range[0] ** 2 / plotUnitsFlux, color='dodgerblue')

    if show_ice_cube_mu:
        # mu fit
        ice_cube_mu_range = get_ice_cube_mu_range()
        ax.fill_between(ice_cube_mu_range[0] / plotUnitsEnergy, ice_cube_mu_range[1] / plotUnitsFlux,
                        ice_cube_mu_range[2] / plotUnitsFlux, hatch='\\', edgecolor='dodgerblue', facecolor='azure', zorder=2)
        plt.plot(ice_cube_mu_range[0] / plotUnitsEnergy,
                 ice_cube_nu_fit(ice_cube_mu_range[0]) * ice_cube_mu_range[0] ** 2 / plotUnitsFlux,
                 color='dodgerblue')

        ax.annotate('IceCube',
                    xy=(3e6 * units.GeV / plotUnitsEnergy, 3e-8), xycoords='data',
                    horizontalalignment='center', color='dodgerblue', rotation=0, fontsize=legendfontsize)

        # Extrapolation
        energy_placeholder = np.array(([1e14, 1e19])) * units.eV
        plt.plot(energy_placeholder / plotUnitsEnergy,
                 ice_cube_nu_fit(energy_placeholder) * energy_placeholder ** 2 / plotUnitsFlux,
                 color='dodgerblue', linestyle=':')

        uplimit = np.copy(nu_mu_data[:, 3])
        uplimit[np.where(nu_mu_data[:, 3] == 0)] = 1
        uplimit[np.where(nu_mu_data[:, 3] != 0.)] = 0

        if nu_mu_show_data_points:
            ax.errorbar(nu_mu_data[:, 0] / plotUnitsEnergy, nu_mu_data[:, 1] / plotUnitsFlux,
                        yerr=nu_mu_data[:, 2:].T / plotUnitsFlux, uplims=uplimit, color='dodgerblue',
                        marker='o', ecolor='dodgerblue', linestyle='None', zorder=3,
                        markersize=7)

    if show_icecube_glashow:
        # only plot the Glashow data point (the first (0) and last (2) entries are upper limits)
        point = 1
        glashow_x = (i3_glashow_emax[point] - i3_glashow_emin[point]) / 2 + i3_glashow_emin[point]
        glashow_y = i3_glashow_y[point]
        ax.errorbar(
            x=glashow_x,
            y=glashow_y,
            xerr=[[glashow_x - i3_glashow_emin[point]], [i3_glashow_emax[point] - glashow_x]],
            yerr=[[glashow_y - i3_glashow_y_lower[point]], [i3_glashow_y_upper[point] - glashow_y]],
            marker='o', markersize=7, color='dodgerblue', ecolor='dodgerblue',
            )

    if show_anita_I_III_limit:
        ax.plot(anita_limit[:, 0] / plotUnitsEnergy, anita_limit[:, 1] / plotUnitsFlux, color='darkorange')
        if energyBinsPerDecade == 2:
            ax.annotate('ANITA I - III',
                        xy=(7e9 * units.GeV / plotUnitsEnergy, 1e-6), xycoords='data',
                        horizontalalignment='left', color='darkorange', fontsize=legendfontsize)
        else:
            ax.annotate('ANITA I - III',
                        xy=(7e9 * units.GeV / plotUnitsEnergy, 5e-7), xycoords='data',
                        horizontalalignment='left', color='darkorange', fontsize=legendfontsize)

    if show_anita_I_IV_limit:
        ax.plot(anita_i_iv_limit[:, 0] / plotUnitsEnergy, anita_i_iv_limit[:, 1] / plotUnitsFlux, color='darkorange')
        if energyBinsPerDecade == 2:
            ax.annotate('ANITA I - IV',
                        xy=(7e9 * units.GeV / plotUnitsEnergy, 1e-6), xycoords='data',
                        horizontalalignment='left', color='darkorange', fontsize=legendfontsize)
        else:
            ax.annotate('ANITA I - IV',
                        xy=(7e9 * units.GeV / plotUnitsEnergy, 5e-7), xycoords='data',
                        horizontalalignment='left', color='darkorange', fontsize=legendfontsize)

    if show_auger_limit:
        ax.plot(auger_limit[:, 0] / plotUnitsEnergy, auger_limit[:, 1] / plotUnitsFlux, color='forestgreen')
        if energyBinsPerDecade == 2:
            ax.annotate('Auger',
                        xy=(8e16 * units.eV / plotUnitsEnergy, 2.1e-7), xycoords='data',
                        horizontalalignment='left', color='forestgreen', rotation=0, fontsize=legendfontsize)
        else:
            ax.annotate('Auger',
                        xy=(9.9e16 * units.eV / plotUnitsEnergy, 4e-8), xycoords='data',
                        horizontalalignment='right', color='forestgreen', rotation=-50, fontsize=legendfontsize)

    if show_ara_1year:
        ax.plot(ara_1year[:, 0] / plotUnitsEnergy, ara_1year[:, 1] / plotUnitsFlux, color='indigo')
#         ax.plot(ara_4year[:,0]/plotUnitsEnergy,ara_4year[:,1]/ plotUnitsFlux,color='indigo',linestyle='--')
        if energyBinsPerDecade == 2:
            ax.annotate('ARA',
                        xy=(5e8 * units.GeV / plotUnitsEnergy, 6e-7), xycoords='data',
                        horizontalalignment='left', color='indigo', rotation=0, fontsize=legendfontsize)
        else:
            ax.annotate('ARA',
                    xy=(2e10 * units.GeV / plotUnitsEnergy, 1.05e-6), xycoords='data',
                    horizontalalignment='left', color='indigo', rotation=0, fontsize=legendfontsize)
    if show_ara:
        ax.plot(ara_4year_E / plotUnitsEnergy, ara_4year_limit / plotUnitsFlux, color='indigo')
#         ax.plot(ara_4year[:,0]/plotUnitsEnergy,ara_4year[:,1]/ plotUnitsFlux,color='indigo',linestyle='--')
        if energyBinsPerDecade == 2:
            ax.annotate('ARA',
                        xy=(5e8 * units.GeV / plotUnitsEnergy, 6e-7), xycoords='data',
                        horizontalalignment='left', color='indigo', rotation=0, fontsize=legendfontsize)
        else:
            ax.annotate('ARA',
                    xy=(2e10 * units.GeV / plotUnitsEnergy, 1.05e-6), xycoords='data',
                    horizontalalignment='left', color='indigo', rotation=0, fontsize=legendfontsize)
    if show_ara_2023:
        ax.plot(ara_2023_E / plotUnitsEnergy, ara_2023_limit / plotUnitsFlux, color='grey', linestyle='--')
        if energyBinsPerDecade == 2:
            ax.annotate('ARA 2023',
                        xy=(2E16 * units.eV / plotUnitsEnergy, 6e-7), xycoords='data',
                        horizontalalignment='left', color='grey', rotation=0, fontsize=legendfontsize)
        else:
            ax.annotate('ARA 2023',
                    xy=(4E17 * units.eV / plotUnitsEnergy, 6e-8), xycoords='data',
                    horizontalalignment='left', color='grey', rotation=0, fontsize=legendfontsize)

    if show_ara_2023_TL:
        ax.plot(ara_2023_E_TL / plotUnitsEnergy, ara_2023_limit_TL / plotUnitsFlux, color='grey', linestyle='--')
        if energyBinsPerDecade == 2:
            ax.annotate('ARA 2023 \n(TL)',
                        xy=(1E16 * units.eV / plotUnitsEnergy, 6e-7), xycoords='data',
                        horizontalalignment='left', color='grey', rotation=0, fontsize=legendfontsize)
        else:
            ax.annotate('ARA 2023 \n(TL)',
                    xy=(1E16 * units.eV / plotUnitsEnergy, 6e-8), xycoords='data',
                    horizontalalignment='left', color='grey', rotation=0, fontsize=legendfontsize)

    if show_arianna:
        ax.plot(ARIANNA_HRA[:, 0] / plotUnitsEnergy, ARIANNA_HRA[:, 1] / plotUnitsFlux, color='red')
#         ax.plot(ara_4year[:,0]/plotUnitsEnergy,ara_4year[:,1]/ plotUnitsFlux,color='indigo',linestyle='--')
        if energyBinsPerDecade == 2:
            ax.annotate('ARIANNA',
                        xy=(5e8 * units.GeV / plotUnitsEnergy, 6e-7), xycoords='data',
                        horizontalalignment='left', color='red', rotation=0, fontsize=legendfontsize)
        else:
            ax.annotate('ARIANNA',
                    xy=(1e8 * units.GeV / plotUnitsEnergy, 1.05e-6), xycoords='data',
                    horizontalalignment='right', color='red', rotation=0, fontsize=legendfontsize)

    if show_IceCubeGen2_whitepaper:
        # flux limit for 5 years
        gen2_E = np.array([1.04811313e+07, 1.32571137e+07, 1.67683294e+07, 2.12095089e+07,
                 2.68269580e+07, 3.39322177e+07, 4.29193426e+07, 5.42867544e+07,
                 6.86648845e+07, 8.68511374e+07, 1.09854114e+08, 1.38949549e+08,
                 1.75751062e+08, 2.22299648e+08, 2.81176870e+08, 3.55648031e+08,
                 4.49843267e+08, 5.68986603e+08, 7.19685673e+08, 9.10298178e+08,
                 1.15139540e+09, 1.45634848e+09, 1.84206997e+09, 2.32995181e+09,
                 2.94705170e+09, 3.72759372e+09, 4.71486636e+09, 5.96362332e+09,
                 7.54312006e+09, 9.54095476e+09, 1.20679264e+10, 1.52641797e+10,
                 1.93069773e+10, 2.44205309e+10, 3.08884360e+10, 3.90693994e+10,
                 4.94171336e+10, 6.25055193e+10, 7.90604321e+10]) * units.GeV
        gen2_flux = np.array([4.31746055e-09, 3.35020540e-09, 3.05749445e-09, 2.03012634e-09,
                 1.63506056e-09, 1.40330116e-09, 1.15462951e-09, 9.20314379e-10,
                 8.30543484e-10, 7.39576420e-10, 6.62805773e-10, 6.53304354e-10,
                 5.41809080e-10, 5.34471053e-10, 5.23081048e-10, 5.20402393e-10,
                 5.02987613e-10, 5.15328224e-10, 5.07092113e-10, 5.21866877e-10,
                 5.30937694e-10, 5.38624813e-10, 5.66520488e-10, 5.71916762e-10,
                 5.93193816e-10, 6.19149497e-10, 6.54847181e-10, 6.78492966e-10,
                 7.15178112e-10, 7.64935941e-10, 8.08811879e-10, 8.58068389e-10,
                 9.13675213e-10, 9.87276891e-10, 1.06320301e-09, 1.15183347e-09,
                 1.25627989e-09, 1.36100197e-09, 1.49171667e-09]) * plotUnitsFlux
        ax.plot(gen2_E / plotUnitsEnergy, gen2_flux / 2 / plotUnitsFlux, color='purple', linestyle=":")
#         ax.plot(ara_4year[:,0]/plotUnitsEnergy,ara_4year[:,1]/ plotUnitsFlux,color='indigo',linestyle='--')
        ax.annotate('IceCube-Gen2 radio',
                    xy=(.8e8 * units.GeV / plotUnitsEnergy, 1.6e-10), xycoords='data',
                    horizontalalignment='left', color='purple', rotation=0, fontsize=legendfontsize)
        
    if show_IceCubeGen2_ICRC2021:
        # https://pos.sissa.it/395/1183/
        # flux limit for 10 years
        gen2_E, gen2_flux = np.loadtxt(os.path.join(os.path.dirname(__file__), "data/Gen2radio_sensitivity_ICRC2021.txt"))
        gen2_E *= units.eV
        gen2_flux *= units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1
        ax.plot(gen2_E / plotUnitsEnergy, gen2_flux / plotUnitsFlux, color='purple', linestyle=":")
#         ax.plot(ara_4year[:,0]/plotUnitsEnergy,ara_4year[:,1]/ plotUnitsFlux,color='indigo',linestyle='--')
        ax.annotate('IceCube-Gen2 radio',
                    xy=(.8e8 * units.GeV / plotUnitsEnergy, 1.6e-10), xycoords='data',
                    horizontalalignment='left', color='purple', rotation=0, fontsize=legendfontsize)
    if show_RNOG:
        # flux limit for 5 years
        RNOG_E = np.array([1.77827941e+07, 5.62341325e+07, 1.77827941e+08, 5.62341325e+08,
                           1.77827941e+09, 5.62341325e+09, 1.77827941e+10, 5.62341325e+10]) * units.GeV
        RNOG_flux = np.array([4.51342568e-08, 1.57748718e-08, 1.03345333e-08, 7.98437261e-09,
                              7.22245212e-09, 7.62588582e-09, 9.28033358e-09, 1.28698605e-08]) * plotUnitsFlux
        ax.plot(RNOG_E / plotUnitsEnergy, RNOG_flux / 0.7 / 2 / plotUnitsFlux, color='red', linestyle="-.")  # uses 70% uptime from RNO-G whitepaper and resacling to 10years
#         ax.plot(ara_4year[:,0]/plotUnitsEnergy,ara_4year[:,1]/ plotUnitsFlux,color='indigo',linestyle='--')
        ax.annotate('RNO-G',
                    xy=(8e18 * units.eV / plotUnitsEnergy, 1e-8), xycoords='data',
                    horizontalalignment='left', va="top", color='red', rotation=10, fontsize=legendfontsize)

    if show_prediction_arianna_200:
        # 10 year sensitivity
        arianna_200 = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', "expected_sensivity_ARIANNA-200.txt"))
        arianna_200[:, 0] *= units.GeV
        arianna_200[:, 1] *= units.GeV * units.cm ** -2 * units.s ** -1
        print(arianna_200)

        _plt4, = ax.plot(arianna_200[:, 0] / plotUnitsEnergy, arianna_200[:, 1] / plotUnitsFlux, label='ARIANNA-200 (5 years)', color='blue', linestyle="--")
        ax.annotate('ARIANNA-200',
                    xy=(.9e19 * units.eV / plotUnitsEnergy, 3.15e-9), xycoords='data',
                    horizontalalignment='left', color='blue', rotation=30, fontsize=legendfontsize)

#         labels.append(_plt4)

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_xlabel(f'neutrino energy [{plotUnitsEnergyStr}]')
    ax.set_ylabel(r'$E^2\Phi$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]')

    if diffuse:
        ax.set_ylim(1e-12, 10e-6)
        ax.set_xlim(1e14 * units.eV / plotUnitsEnergy, 1e20 * units.eV / plotUnitsEnergy)
    else:
        ax.set_ylim(1e-11, 2e-6)
        ax.set_xlim(1e5, 1e11)

    plt.tight_layout()
    return fig, ax


def add_limit(ax, limit_labels, E, Veffsr, n_stations, label, livetime=3 * units.year, linestyle='-', color='r', linewidth=3, band=False):
    """
    add limit curve to limit plot
    """
    E = np.array(E)
    Veffsr = np.array(Veffsr)

    if band:

        limit_lower = fluxes.get_limit_e2_flux(energy=E,
                                         veff_sr=Veffsr[0],
                                         livetime=livetime,
                                         signalEff=n_stations,
                                         energyBinsPerDecade=energyBinsPerDecade,
                                         upperLimOnEvents=2.44,
                                         nuCrsScn='ctw')
        limit_upper = fluxes.get_limit_e2_flux(energy=E,
                                         veff_sr=Veffsr[1],
                                         livetime=livetime,
                                         signalEff=n_stations,
                                         energyBinsPerDecade=energyBinsPerDecade,
                                         upperLimOnEvents=2.44,
                                         nuCrsScn='ctw')

        plt1 = ax.fill_between(E / plotUnitsEnergy, limit_upper / plotUnitsFlux,
        limit_lower / plotUnitsFlux, color=color, alpha=0.2)

    else:

        limit = fluxes.get_limit_e2_flux(energy=E,
                                         veff_sr=Veffsr,
                                         livetime=livetime,
                                         signalEff=n_stations,
                                         energyBinsPerDecade=energyBinsPerDecade,
                                         upperLimOnEvents=2.44,
                                         nuCrsScn='ctw')

    #         _plt, = ax.plot(E/plotUnitsEnergy,limit/ plotUnitsFlux, linestyle=linestyle, color=color,
    #                         label="{2}: {0} stations, {1} years".format(n_stations,int(livetime/units.year),label),
    #                         linewidth=linewidth)
        _plt, = ax.plot(E / plotUnitsEnergy, limit / plotUnitsFlux, linestyle=linestyle, color=color,
                        label="{1}: {0} years".format(int(livetime / units.year), label),
                        linewidth=linewidth)

        limit_labels.append(_plt)

    return limit_labels


if __name__ == "__main__":
    # 50 meter
    veff = np.array((
    [1e+16, 3.162277660168379e+16, 1e+17, 3.1622776601683795e+17, 1e+18, 3.1622776601683794e+18, 1e+19, 3.162277660168379e+19],
    [0.007467602898257461, 0.06986834193426224, 0.5333379226865426, 2.1410489793474383, 5.896654567671568, 11.343574036186226, 18.415350759353128, 27.81614390854279]
    )).T

    veff[:, 0] *= units.eV
    veff[:, 1] *= units.km ** 3 * units.sr
    veff_label = 'One current design'

#     strawman_veff_pa = np.array(( [1.00000000e+16, 3.16227766e+16, 1.00000000e+17, 3.16227766e+17, 1.00000000e+18, 3.16227766e+18, 1.00000000e+19, 3.16227766e+19],
#                               [1.82805666e+07, 1.34497197e+08, 6.32044851e+08, 2.20387046e+09, 4.86050340e+09, 8.18585201e+09, 1.25636305e+10, 1.83360237e+10])).T
#
#     strawman_veff_pa[:,0] *= units.eV
#     strawman_veff_pa[:,1] *= units.m**3 * units.sr

#     strawman_pa_label = 'Strawman + PA@15m@2s'
#     strawman_pa_label = 'One current design'
    fig, ax = get_E2_limit_figure(diffuse=DIFFUSE)
    labels = []
    labels = add_limit(ax, labels, veff[:, 0], veff[:, 1], n_stations=100, livetime=5 * units.year, label=veff_label)
    labels = add_limit(ax, labels, veff[:, 0], veff[:, 1], n_stations=1000, livetime=5 * units.year, label=veff_label)
    plt.legend(handles=labels, loc=2)
    if DIFFUSE:
        name_plot = "Limit_diffuse.pdf"
    else:
        name_plot = "Limit_sources.pdf"
    plt.savefig(name_plot)
