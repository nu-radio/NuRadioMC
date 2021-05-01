import matplotlib
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rc('axes', labelsize=20)

legendfontsize = 11
import matplotlib.pyplot as plt
plt.rcParams['xtick.top']  = True
plt.rcParams['ytick.right']  = True

import numpy as np
from NuRadioReco.utilities import units
from NuRadioMC.utilities import fluxes
import os
from scipy.interpolate import interp1d

energyBinsPerDecade = 1.
plotUnitsEnergy = units.GeV
plotUnitsFlux = units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
DIFFUSE = True

# Multimessenger Figure as made for https://arxiv.org/abs/2010.12279
# Do not use without citing correct references for lines and the original figure
# --------------------------------------------------------------------
# Compilation of published limits, data and proposed experiments, no particular order
# --------------------------------------------------------------------
# GRAND white paper, https://arxiv.org/abs/1810.09994
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

GRAND_10k *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
GRAND_10k *= energyBinsPerDecade

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

GRAND_200k *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
GRAND_200k *= energyBinsPerDecade

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

Radar[:, 0] = 10**Radar[:, 0] * units.eV
Radar[:, 1] *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
Radar[:, 1] /= 2 #halfdecade bins
Radar[:, 1] *= energyBinsPerDecade
Radar[:, 2] *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
Radar[:, 2] /= 2 #halfdecade bins
Radar[:, 2] *= energyBinsPerDecade

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

ice_cube_limit[:, 0] = 10**ice_cube_limit[:, 0] * units.GeV
ice_cube_limit[:, 1] = 10**ice_cube_limit[:, 1] * (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
ice_cube_limit[:, 1] *= energyBinsPerDecade


# HESE 8 years
#log(E (GeV)); log(E^2 dN/dE (GeV cm^-2 s-1 sr-1)); format x y -dy +dy
ice_cube_hese = np.array(([
    (4.78516,	-7.63256,	0.223256,	0.167442),
    (5.10938,	-7.66977,	0.139535,	0.102326),
    (5.42969,	-8.36744,	0.930233,	0.297674),
    (5.75391,	-8.51628,	0.2,       	0.	),
    (6.07813,	-8.38605,	0.604651,	0.288372),
    (6.39844,	-8.35814,	0.455814,	0.334884),
    (6.72266,	-9.0,    	0.2 ,    	0)
]))

# get uncertainties in right order
ice_cube_hese[:, 2] = 10**ice_cube_hese[:, 1] - 10**(ice_cube_hese[:, 1]-ice_cube_hese[:, 2])
ice_cube_hese[:, 3] = 10**(ice_cube_hese[:, 1]+ice_cube_hese[:, 3]) - 10**ice_cube_hese[:, 1]

ice_cube_hese[:, 0] = 10**ice_cube_hese[:, 0] * units.GeV
ice_cube_hese[:, 1] = 10**ice_cube_hese[:, 1] * (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
ice_cube_hese[:, 1] *= 3

ice_cube_hese[:, 2] *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
ice_cube_hese[:, 2] *=  3

ice_cube_hese[:, 3] *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
ice_cube_hese[:, 3] *= 3


# IceCube
def ice_cube_nu_fit(energy, slope=-2.19, offset=1.01):
    # ApJ slope=-2.13, offset=0.9
    flux = 3 * offset * (energy / (100 * units.TeV))**slope * 1e-18 * \
        (units.GeV**-1 * units.cm**-2 * units.second**-1 * units.sr**-1)
    return flux


def get_ice_cube_mu_range():
    energy = np.arange(1e5, 5e6, 1e5) * units.GeV
#     upper = np.maximum(ice_cube_nu_fit(energy, offset=0.9, slope=-2.), ice_cube_nu_fit(energy, offset=1.2, slope=-2.13)) # APJ
    upper = np.maximum(ice_cube_nu_fit(energy, offset=1.01, slope=-2.09),
                    ice_cube_nu_fit(energy, offset=1.27, slope=-2.19),ice_cube_nu_fit(energy, offset=1.27, slope=-2.09)) # ICRC
    upper *= energy**2
#     lower = np.minimum(ice_cube_nu_fit(energy, offset=0.9, slope=-2.26),
#                        ice_cube_nu_fit(energy, offset=0.63, slope=-2.13)) #ApJ
    lower = np.minimum(ice_cube_nu_fit(energy, offset=1.01, slope=-2.29),
                       ice_cube_nu_fit(energy, offset=0.78, slope=-2.19)) #ICRC
    lower *= energy**2
    return energy, upper, lower


def get_ice_cube_hese_range():
    energy = np.arange(1e5, 5e6, 1e5) * units.GeV
    upper = np.maximum(ice_cube_nu_fit(energy, offset=2.46, slope=-2.63),
                       ice_cube_nu_fit(energy, offset=2.76, slope=-2.92))
    upper *= energy**2
    lower = np.minimum(ice_cube_nu_fit(energy, offset=2.46, slope=-3.25),
                       ice_cube_nu_fit(energy, offset=2.16, slope=-2.92))
    lower *= energy**2
    return energy, upper, lower


# ANITA I - III
# Phys. Rev. D 98, 022001 (2018)
anita_limit_old = np.array(([
    (9.94e17,	3.79e-14 * 9.94e17 / 1e9),
    (2.37e18,	2.15e-15 * 2.37e18 / 1e9),
    (5.19e18,	2.33e-16 * 5.19e18 / 1e9),
    (1.10e19,	3.64e-17 * 1.10e19 / 1e9),
    (3.55e19,	4.45e-18 * 3.55e19 / 1e9),
    (1.11e20,	9.22e-19 * 1.11e20 / 1e9),
    (4.18e20,	2.97e-19 * 4.18e20 / 1e9),
    (9.70e20,	1.62e-19 * 9.70e20 / 1e9)
]))

anita_limit_old[:, 0] *= units.eV
anita_limit_old[:, 1] *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
anita_limit_old[:, 1] /= 2
anita_limit_old[:, 1] *= energyBinsPerDecade

#ANITA I - IV
#PHYS. REV. D99,122001 (2019)
anita_limit = np.array(([
(1000000000000000000, 2.8504802271185626e-14),
(1488426085696288800, 9.185752805548236e-15),
(2080567538217180200, 3.410255242645634e-15),
(3367229575211422700, 9.539117464664601e-16),
(5117889550210760000, 3.0740074665268947e-16),
(9792849742266053000, 5.784889690095893e-17),
(17233109056135154000, 2.2093386751530843e-17),
(26746855243890946000, 1.028715022524397e-17),
(47068165596092400000, 4.277081649153833e-18),
(100000000000000000000, 1.4585940117250412e-18),
(235899637070160530000, 6.238525681114845e-19),
(522614936908700700000, 3.253082534408512e-19),
(958999060746002900000, 2.0681106472222013e-19)
]))

anita_limit[:, 0] *= units.eV
anita_limit[:, 1] *= (units.cm**-2*units.sr**-1*units.second**-1)
anita_limit[:, 1] *= anita_limit[:, 0]
anita_limit[:, 1] /= 2
anita_limit[:, 1] *= energyBinsPerDecade


# Auger neutrino limit
# Auger 9 years, all flavour (x3)
auger_limit_old = np.array(([
    (16.7523809524, 4.462265901e-07),
    (17.2523809524, 1.103901153e-07),
    (17.7523809524, 6.487559078e-08),
    (18.2380952381, 7.739545498e-08),
    (18.7523809524, 1.387743075e-07),
    (19.2571428571, 3.083827665e-07),
    (19.7523809524, 7.467202523e-07),
    (20.2476190476, 1.998499395e-06)
]))
auger_limit_old[:, 0] = 10 ** auger_limit_old[:, 0] * units.eV
auger_limit_old[:, 1] *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
auger_limit_old[:, 1] /= 2  # half-decade binning
auger_limit_old[:, 1] *= energyBinsPerDecade


#Auger neutrino limit
# A. Aab et al JCAP10(2019)022
auger_limit = np.array(([
(56346254419996160, 9.40953759836932e-8),
(177474085951862620, 2.2741975840514183e-8),
(558991036913918700, 1.4554443171397434e-8),
(1760656930132613400, 1.8566176166031132e-8),
(5545550144664245000, 3.2765628449464426e-8),
(17747408595186264000, 7.08304929506588e-8),
(60535052084373815000, 1.8009705365764743e-7),
(178893744973230050000, 4.4872718282328947e-7)
]))

auger_limit[:,0] *= units.eV
auger_limit[:,1] *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
auger_limit[:,1] *= 3 #single flavour limit
auger_limit[:, 1] /= 2  # half-decade binning
auger_limit[:, 1] *= energyBinsPerDecade


#Auger CR data (Extracted from plot M. Ahlers for simplicity)
auger_cr = np.array(([
(352698897.6092299, 6.286099325959247e-7),
(457515775.8585651, 4.778935283063509e-7),
(563181709.9317356, 3.483518447161274e-7),
(711713741.1769367, 2.648395546099635e-7),
(876227737.6733545, 2.0135499463083514e-7),
(1136358073.479402, 1.437055126475309e-7),
(1436630079.3568363, 1.2138629160879932e-7),
(1768146499.8323948, 8.483284200792424e-8),
(2233938885.090647, 6.054661417472254e-8),
(2824461309.811831, 5.223148502933225e-8),
(3568524412.4402666, 3.7278481976925465e-8),
(4393395675.589912, 2.83424753125118e-8),
(5552540257.054003, 2.2006390106970647e-8),
(7020865401.430125, 1.9388205552650998e-8),
(8876063499.8851, 1.637698046252276e-8),
(11223267953.450876, 1.4428548344172285e-8),
(14188911670.456398, 1.218761858560785e-8),
(17471488594.240013, 9.664758464903206e-9),
(22672782672.047684, 8.163424801967293e-9),
(27911416815.16653, 6.077229792925731e-9),
(35281127678.64826, 4.921633627974248e-9),
(44525751199.30825, 2.6156942564408127e-9),
(54774322210.40531, 1.611029800643381e-9),
(69176265013.90474, 1.0348989302024764e-9)
]))

auger_cr[:,0] *= units.GeV
auger_cr[:,1] *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
auger_cr_uncert = np.zeros_like(auger_cr)
auger_cr_uncert[-1,0] = 1.0380434265361934e-9 - 3.7558564955732889e-10
auger_cr_uncert[-1,1] = 1.2486930155109053e-9 - 1.0380434265361934e-9
auger_cr_uncert[-2,0] = 1.613100199052136e-9 - 1.439721959223142e-9
auger_cr_uncert[-2,1] = 1.7818503279955935e-9 - 1.613100199052136e-9
auger_cr_uncert[:,0] *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
auger_cr_uncert[:,1] *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)


# ARA Published 2sta x 1yr analysis level limit:
ara_1year = np.array((
[9.80923e+15, 3.11547e+16, 9.79329e+16, 3.1057e+17, 9.75635e+17,
3.0924e+18, 9.80103e+18, 3.07732e+19, 9.75028e+19],
[0.000166106, 1.24595e-05, 4.06271e-06, 2.04351e-06, 1.48811e-06,
1.42649e-06, 1.50393e-06, 2.10936e-06, 3.25384e-06]
))

ara_1year = ara_1year.T

ara_1year[:,0] *= units.eV
ara_1year[:,1] *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
ara_1year[:,1] /= 2 #half-decade binning
ara_1year[:,1] *= energyBinsPerDecade

# Ongoing analysis 2sta x 4yr *trigger* level projected limit (analysis
# level not fully defined yet, Hokanson-Fasig)

ara_4year = np.array((

[9.91245E+15,3.11814E+16,9.80021E+16,3.10745E+17,9.94099E+17,3.0936E+18,9.71449E+18,3.07805E+19,9.75192E+19],
# [1.01518E+16,1.01357E+17,1.01748E+18,1.0234E+19,1.03113E+20],
# [7.27626e-06,5.06909e-07,2.13658e-07,2.02468e-07,4.46012e-07]
[5.35394e-06,1.24309e-06,4.20315e-07,2.24199e-07,1.61582e-07,1.50329e-07,1.63715e-07,2.24543e-07,3.36398e-07]
))

ara_4year = ara_4year.T

ara_4year[:,0] *= units.eV
ara_4year[:,1] *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
ara_4year[:,1] /= 2 #half-decade binning
ara_4year[:,1] *= energyBinsPerDecade

# ARIANNA HRA 2019  JCAP03(2020)053

arianna = np.array((
[9436557.8964637, 0.000008769317199833857],
[30715470.0772479, 0.000004352945903421164],
[99894902.91209452, 0.0000029613032941314635],
[302299625.9638324, 0.0000019452408628429242],
[959271728.9474759, 0.0000016327803180905049],
[2829030060.4728136, 0.000002086371468224478],
[9636453793.064827, 0.000002859392437735097],
[32029820708.066948, 0.000004668759694030687],
[89975534007.67921, 0.0000073607175340000305]
))

arianna[:,0] *= units.GeV
arianna[:,1] *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
arianna[:,1] /= 2 #half-decade binning
arianna[:,1] *= energyBinsPerDecade


# Gamma Rays, Fermin Diffuse (data points as above)

fermi_diffuse_central = np.array((
[21.92167404408887, 1.5340736759945403e-7],
[31.59885931587562, 1.5013860902053343e-7],
[43.219028722563294, 1.379524179890278e-7],
[62.25315864341186, 1.1170145261601652e-7],
[85.12580313519733, 9.635114082768082e-8],
[119.28335576489448, 5.3403270661478004e-8],
[171.87178736843583, 4.704164984928985e-8],
[240.99050448180415, 3.0857725396756525e-8],
[347.484931133411, 3.2854530491322494e-8],
[485.8706903255871, 1.0312317144922938e-8]
))

fermi_diffuse_central[:,0] *= units.GeV
fermi_diffuse_central[:,1] *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)

# lower, upper
fermi_errors = np.array((
[1.704366052008598e-7, 1.3519743124902024e-7],
[1.668163802232929e-7, 1.323212010361125e-7],
[1.4694953152258203e-7, 1.24164611991644e-7],
[1.241095286059241e-7, 9.639390424088851e-8],
[1.048194909584834e-7, 7.97150264958008e-8],
[6.320307758869306e-8, 4.236025804974686e-8],
[5.451568113436708e-8, 3.653775114619792e-8],
[3.890211101701257e-8, 2.1571327546817124e-8],
[4.230101813966531e-8, 2.201991489674978e-8],
[1.82098454591397e-8, 3.104683785409394e-9]
))

fermi_errors[:,0] *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
fermi_errors[:,1] *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
fermi_errors[:,1] -= fermi_diffuse_central[:,1]
fermi_errors[:,1] *= -1
fermi_errors[:,0] -= fermi_diffuse_central[:,1]

fermi_pion  = np.array((
[42.89480906642609, 1.495707893643605e-7],
[8557656.431026021, 2.4189553240499494e-8],
))

fermi_pion[:,0] *= units.GeV
fermi_pion[:,1] *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)

# ------------------------------------------------------------
# Set up figure and add theoretical models
# ------------------------------------------------------------

def get_E2_limit_figure(
                        show_ice_cube_EHE_limit=True,
                        show_ice_cube_HESE_data=True,
                        show_ice_cube_HESE_fit=True,
                        show_ice_cube_mu=True,
                        show_anita_limit=True,
                        show_auger_limit=True,
                        show_auger_cr=True,
                        show_ara=True,
                        show_arianna=True,
                        show_neutrino_best_fit=True,
                        show_neutrino_best_case=True,
                        show_neutrino_worst_case=True,
                        show_grand_10k=True,
                        show_grand_200k=False,
                        show_radar=False,
                        show_fermi=True):

    # Limit E2 Plot
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(11, 6))

    # Theoretical models

    # Diffuse Flux,
    # From Heinze et al, https://arxiv.org/abs/1901.03338, ApJ
    Heinze_band = np.loadtxt(os.path.join(os.path.dirname(__file__), "talys_neu_bands.out"))
    Heinze_band_label = 'Best fit UHECR, Heinze et al.'
    Heinze_band[:,0] *= units.GeV
    Heinze_band[:,1] *= units.GeV**-1 * units.cm**-2 * units.second**-1 * units.sr**-1
    Heinze_band_int = interp1d(Heinze_band[:,0],Heinze_band[:, 1] * Heinze_band[:, 0]**2)

    Heinze_evo = np.loadtxt(os.path.join(os.path.dirname(__file__), "talys_neu_evolutions.out"))
    Heinze_evo_label = r'Best fit UHECR + 3$\sigma$, Heinze et al.'
    Heinze_evo[:,0] *= units.GeV
    Heinze_evo[:,6] *= units.GeV**-1 * units.cm**-2 * units.second**-1 * units.sr**-1
    Heinze_evo_int = interp1d(Heinze_evo[:,0],Heinze_evo[:, 6] * Heinze_evo[:, 0] **2)

    # from van Vliet et al., Phys. Rev. D 100, 021302 (2019), + additional calculations
    vanVliet_max_1 = np.loadtxt(os.path.join(os.path.dirname(__file__), "MaxNeutrinos1.txt"))
    vanVliet_max_2 = np.loadtxt(os.path.join(os.path.dirname(__file__), "MaxNeutrinos2.txt"))
    vanVliet_max = np.maximum(vanVliet_max_1[1, :], vanVliet_max_2[1, :])
    vanVliet_max *= units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
    vanVliet_max_1[0,:] *= units.GeV
    vanVliet_max_label = r'Not excluded from UHECR measurements'
    vanVliet_max_int = interp1d(vanVliet_max_1[0,:],vanVliet_max)

    vanVliet_reas = np.loadtxt(os.path.join(os.path.dirname(__file__), "ReasonableNeutrinos1.txt"))
    vanVliet_reas_label = r'10% protons in UHECRs, van Vliet et al.'
    vanVliet_reas[0, :] *= units.GeV
    vanVliet_reas[1, :] *= units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
    vanVliet_reas_int = interp1d(vanVliet_reas[0, :],vanVliet_reas[1, :])

    vanVliet_TA = np.loadtxt(os.path.join(os.path.dirname(__file__), "TA_data/NuSpectrum_TABestFit.txt"))
    vanVliet_TA_label = r'TA composition'
    vanVliet_TA[0, :] *= units.GeV
    vanVliet_TA[1, :] *= units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
    vanVliet_TA_int = interp1d(vanVliet_TA[0, :],vanVliet_TA[1, :])

    vanVliet_TA_A1 = np.loadtxt(os.path.join(os.path.dirname(__file__), "TA_data/NuSpectrum_TABestFit_scenA_z=1_header.txt"))
    vanVliet_TA_A1_label = r'TA composition'
    vanVliet_TA_A1[0, :] *= units.GeV
    vanVliet_TA_A1[1, :] *= units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
    vanVliet_TA_A1_int = interp1d(vanVliet_TA_A1[0, :],vanVliet_TA_A1[1, :])

    vanVliet_TA_A2 = np.loadtxt(os.path.join(os.path.dirname(__file__), "TA_data/NuSpectrum_TABestFit_scenA_z=5_header.txt"))
    vanVliet_TA_A2_label = r'TA composition'
    vanVliet_TA_A2[0, :] *= units.GeV
    vanVliet_TA_A2[1, :] *= units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
    vanVliet_TA_A2_int = interp1d(vanVliet_TA_A2[0, :],vanVliet_TA_A2[1, :])

    #Sources
    # TDE, Biehl et al. (1711.03555)
    tde = np.loadtxt(os.path.join(os.path.dirname(__file__),"TDEneutrinos.txt"))
    tde[:,0] *= units.GeV
    tde[:,1] *= units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
    tde[:,1] *= 3 # all flavor
    tde_label = "TDE, Biehl et al. (1711.03555)"
    tde_int = interp1d(tde[:,0],tde[:,1])

    # LLGRB, Boncioli et al. (1808.07481)
    ll_grb = np.loadtxt(os.path.join(os.path.dirname(__file__),"LLGRBneutrinos.txt"))
    ll_grb[:,0] *= units.GeV
    ll_grb[:,1] *= units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
    ll_grb[:,1] *= 3 # all flavor
    ll_grb[:,2] *= units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
    ll_grb[:,2] *= 3 # all flavor
    ll_grb[:,3] *= units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
    ll_grb[:,3] *= 3 # all flavor
#     ll_grb_label = "LLGRB, Boncioli et al. (1808.07481)"
    ll_grb_label = "Neutrinos from low-luminosity GRBs"
    ll_grb_int = interp1d(ll_grb[:,0],ll_grb[:,1])

    # Pulsar, Fang et al. (1311.2044)
    pulsars = np.loadtxt(os.path.join(os.path.dirname(__file__),"Pulsar_Fang+_2014.txt"))
    pulsars[:,0] *= units.GeV
    pulsars[:,1] *= units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
    pulsars[:,2] *= units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
    pulsars_label = "Pulsar, Fang et al. (1311.2044)"
    pulsars_int = interp1d(pulsars[:,0],np.average((pulsars[:,1],pulsars[:,2]),axis=0))

    # Clusters, Fang & Murase, (1704.00015)
    clusters = np.loadtxt(os.path.join(os.path.dirname(__file__),"cluster_Fang_Murase_2018.txt"))
    clusters[:,0] *= units.GeV
    clusters[:,1] *= units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
    clusters_label = "Clusters, Fang & Murase, (1704.00015)"
    clusters_int = interp1d(clusters[:,0],clusters[:,1])

    # NS-NS merger, Fang & Metzger (1707.04263)
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

    data_ns_merger[:,0] *= units.GeV
    data_ns_merger[:,1] *= units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
    data_ns_merger_label = 'NS-NS merger, Fang & Metzger 1707.04263'
    data_ns_merger_int = interp1d(data_ns_merger[:,0],data_ns_merger[:,1])

    # GRB afterglow, Murase (0707.1140)
    grb_afterglow = np.loadtxt(os.path.join(os.path.dirname(__file__),"grb_ag_wind_optimistic.dat"))
    grb_afterglow[:,0] = 10**grb_afterglow[:,0]*units.GeV
    grb_afterglow[:,1] = 10**grb_afterglow[:,1]*units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
    grb_afterglow_label = "GRB afterglow, Murase 0707.1140"
    grb_afterglow_int = interp1d(grb_afterglow[:,0],grb_afterglow[:,1])

    # AGN, Murase et al. (1403.4089)
    agn = np.loadtxt(os.path.join(os.path.dirname(__file__),"agn_murase.dat"),delimiter=',')
    agn[:,0] = agn[:,0]*units.GeV
    agn[:,1] = agn[:,1]*units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
    #agn_label = "AGN, Murase et al. 1403.4089"
    agn_label = "From AGN"
    agn_int = interp1d(agn[:,0],agn[:,1])

    # Blazars, Rodrigues et al. (2003.08392), https://arxiv.org/pdf/2003.08392.pdf
    blazars = np.array((
    [6181.392928452677, 7.040339186251669e-12],
    [25345.433371306408, 2.7111595745321162e-11],
    [40221.36678451652, 4.8959792913241266e-11],
    [63828.39552651239, 9.174171298046798e-11],
    [103923.33899733753, 1.5674264306958923e-10],
    [171389.13500163457, 2.533630716140034e-10],
    [293742.35596608516, 3.9469051019538317e-10],
    [536790.7382360712, 5.20687965535225e-10],
    [968439.3407453113, 6.26313025479146e-10],
    [1839178.8135669571, 6.869077174391688e-10],
    [3360952.661511085, 8.573439355694519e-10],
    [5834675.576082829, 1.263584465566179e-9],
    [10000000, 2.0805675382171716e-9],
    [17360183.74462777, 3.971279750100851e-9],
    [30526686.43694, 7.040339186251669e-9],
    [50994212.73151095, 1.0569736755472276e-8],
    [94390951.09495592, 1.3943924632810026e-8],
    [170293382.39163262, 1.646559921330565e-8],
    [260040973.21066046, 1.6772543638206335e-8],
    [457264126.15967125, 1.5868459192725603e-8],
    [754114559.8852615, 1.3192310230045095e-8],
    [1243676765.5645146, 9.46097356192217e-9],
    [1804128260.3490238, 6.911498092781073e-9],
    [2583784373.9647183, 4.436687330978625e-9],
    [3996421706.7111306, 2.4118646996409997e-9],
    [5723479967.868826, 1.1103363181676414e-9],
    [7887437540.468643, 5.303944005129917e-10],
    [10594235871.127207, 2.353194117177983e-10],
    [13692736497.681028, 1.0249323173276398e-10],
    [16812284900.864822, 4.4640866957561916e-11],
    [19610068406.48923, 2.3823488037280444e-11],
    [42885648107.110916, 1.0900167083351738e-12],
    [1e11,1e-13]
    ))
    blazars[:,0] *= units.GeV
    blazars[:,1] *= units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
    blazars_label="Blazars, Rodrigues et al. 2003.08392"
    blazars_int = interp1d(blazars[:,0],blazars[:,1])

    # Use the same energy range for all plots, use interpolated functions

    #plot_energy = np.arange(1e5,1e11,3e5)*units.GeV
    plot_energy = np.array([1e5,1.1e5,1.2e5,1.5e5,1.7e5,2e5,3e5,4e5,6e5,7e5,8e5,9e5,
                            1e6,1.1e6,1.2e6,1.5e6,1.7e6,2e6,3e6,4e6,6e6,7e6,8e6,9e6,
                            1e7,1.1e7,1.2e7,1.5e7,1.7e7,2e7,3e7,4e7,6e7,7e7,8e7,9e7,
                            1e8,1.1e8,1.2e8,1.5e8,1.7e8,2e8,3e8,4e8,6e8,7e8,8e8,9e8,
                            1e9,1.1e9,1.2e9,1.5e9,1.7e9,2e9,3e9,4e9,6e9,7e9,8e9,9e9,
                            1e10,1.1e10,1.2e10,1.5e10,1.7e10,2e10,3e10,4e10,6e10,7e10,8e10,9e10])*units.GeV


    max_all_models = np.maximum(vanVliet_max_int(plot_energy),Heinze_evo_int(plot_energy))


    ta_plot, = ax.plot(plot_energy/plotUnitsEnergy, vanVliet_TA_A2_int(plot_energy)/plotUnitsFlux, c='darkgoldenrod',dashes=[10, 5, 20, 5],label= 'van Vliet et al., TA UHECR')

    auger_plot, = ax.plot(plot_energy/plotUnitsEnergy, vanVliet_reas_int(plot_energy)/plotUnitsFlux, c='darkgoldenrod',linestyle='-.',label= 'van Vliet et al., Auger UHECR')

    best_fit_plot, = ax.plot(plot_energy/plotUnitsEnergy,Heinze_band_int(plot_energy)/plotUnitsFlux, c='darkgoldenrod',linestyle='--',label = "Heinze et al., Auger best fit")

    clusters_plot, = ax.plot(plot_energy/plotUnitsEnergy, clusters_int(plot_energy)/plotUnitsFlux, c='indianred',linestyle='--',label = "Fang & Murase, Clusters")

    pulsars_plot, = ax.plot(plot_energy/plotUnitsEnergy, pulsars_int(plot_energy)/plotUnitsFlux ,c='indianred',linestyle=':',label = "Fang et a.s, Pulsars")

    ll_grb_plot, = ax.plot(plot_energy/plotUnitsEnergy,ll_grb_int(plot_energy)/plotUnitsFlux, c='indianred',linestyle='-.', label = "Boncioli et al., LL GRBs")

    grb_afterglow_plot, = ax.plot(plot_energy/plotUnitsEnergy,grb_afterglow_int(plot_energy)/plotUnitsFlux, c='indianred',linestyle=(0, (3, 5, 1, 5)), label = "Murase, GRB afterglow")

    agn_plot, = ax.plot(plot_energy/plotUnitsEnergy,agn_int(plot_energy)/plotUnitsFlux, c='indianred',linestyle='-.' ,label="Murase et al., AGN")

    blazars_plot, =  ax.plot(plot_energy/plotUnitsEnergy,blazars_int(plot_energy)/plotUnitsFlux, c='indianred',dashes=[5, 7, 5, 7], label = "Rodrigues et al., Blazars")

    dummy_plot = plt.Line2D([0],[0],color="w",label=r'$\bf{Neutrino\ flux\ predictions}$')
    first_legend = plt.legend( handles=[dummy_plot, agn_plot, clusters_plot, blazars_plot,ll_grb_plot, pulsars_plot, grb_afterglow_plot, ta_plot,auger_plot,best_fit_plot], loc=3,fontsize=legendfontsize)

    plt.gca().add_artist(first_legend)


    #-----------------------------------------------------------------------

    if show_grand_10k:
        # 5 years scaling
        ax.plot(GRAND_energy / plotUnitsEnergy, GRAND_10k *3 /5. / plotUnitsFlux, linestyle=":", color='saddlebrown')
        if energyBinsPerDecade == 2:
            ax.annotate('GRAND 10k',
                            xy=(1e10, 1.e-7), xycoords='data',
                            horizontalalignment='left', color='saddlebrown', rotation=40,fontsize=legendfontsize)
        else:
            ax.annotate('GRAND 10k',
                xy=(5e9, 1.2e-8), xycoords='data',
                horizontalalignment='left', color='saddlebrown', rotation=40,fontsize=legendfontsize)



    if show_grand_200k:
        # 5 year scaling
        ax.plot(GRAND_energy / plotUnitsEnergy, GRAND_200k *3 /5. / plotUnitsFlux, linestyle=":", color='saddlebrown')
        ax.annotate('GRAND 200k',
                    xy=(5e9, 6e-10), xycoords='data',
                    horizontalalignment='left', color='saddlebrown', rotation=40,fontsize=legendfontsize)
    if show_radar:
        ax.fill_between(Radar[:, 0] / plotUnitsEnergy, Radar[:, 1] / plotUnitsFlux,
                        Radar[:, 2] / plotUnitsFlux, facecolor='None', hatch='x', edgecolor='0.8')
        ax.annotate('Radar',
                    xy=(1e9, 4.5e-8), xycoords='data',
                    horizontalalignment='left', color='0.7', rotation=45,fontsize=legendfontsize)

    if show_ice_cube_EHE_limit:
        ax.plot(ice_cube_limit[2:, 0] / plotUnitsEnergy, ice_cube_limit[2:, 1] / plotUnitsFlux, color='dodgerblue')
        if energyBinsPerDecade == 2:
            ax.annotate('IceCube',
                    xy=(0.7e7, 4e-8), xycoords='data',
                    horizontalalignment='center', color='dodgerblue', rotation=0,fontsize=legendfontsize)
        else:
            ax.annotate('high-energy neutrinos \n (IceCube)',
                    xy=(4e6, 3e-7), xycoords='data',
                    horizontalalignment='center', color='dodgerblue', rotation=0,fontsize=legendfontsize+2, fontweight='semibold')

    if show_ice_cube_HESE_data:
        # data points
        uplimit = np.copy(ice_cube_hese[:, 3])
        uplimit[np.where(ice_cube_hese[:, 3] == 0)] = 1
        uplimit[np.where(ice_cube_hese[:, 3] != 0.)] = 0


        ax.errorbar(ice_cube_hese[:, 0] / plotUnitsEnergy, ice_cube_hese[:, 1] / plotUnitsFlux, yerr=ice_cube_hese[:,2:].T / plotUnitsFlux, uplims=uplimit, color='dodgerblue', marker='o', ecolor='dodgerblue', linestyle='None',zorder=3)


    if show_ice_cube_HESE_fit:
        ice_cube_hese_range = get_ice_cube_hese_range()
        ax.fill_between(ice_cube_hese_range[0] / plotUnitsEnergy, ice_cube_hese_range[1] / plotUnitsFlux,
                        ice_cube_hese_range[2] / plotUnitsFlux, hatch='//', edgecolor='dodgerblue', facecolor='azure',zorder=2)
        plt.plot(ice_cube_hese_range[0] / plotUnitsEnergy, ice_cube_nu_fit(ice_cube_hese_range[0],
                                                                           offset=2.46, slope=-2.92) * ice_cube_hese_range[0]**2 / plotUnitsFlux, color='dodgerblue')

    if show_ice_cube_mu:
        # mu fit
        ice_cube_mu_range = get_ice_cube_mu_range()
        ax.fill_between(ice_cube_mu_range[0] / plotUnitsEnergy, ice_cube_mu_range[1] / plotUnitsFlux,
                        ice_cube_mu_range[2] / plotUnitsFlux, hatch='\\', edgecolor='dodgerblue', facecolor='azure',zorder=2)
        plt.plot(ice_cube_mu_range[0] / plotUnitsEnergy, ice_cube_nu_fit(ice_cube_mu_range[0],
                                                                         offset=1.01, slope=-2.19) * ice_cube_mu_range[0]**2 / plotUnitsFlux, color='dodgerblue')

        # Extrapolation
        energy_placeholder =   np.array(([1e14,1e17]))*units.eV
        plt.plot(energy_placeholder / plotUnitsEnergy, ice_cube_nu_fit(energy_placeholder,
                                                                         offset=1.01, slope=-2.19) * energy_placeholder**2 / plotUnitsFlux, color='dodgerblue',linestyle='--')


    if show_anita_limit:
        ax.plot(anita_limit[:, 0] / plotUnitsEnergy, anita_limit[:, 1] / plotUnitsFlux, color='darkorange')
#         ax.plot(anita_limit_old[:, 0] / plotUnitsEnergy, anita_limit_old[:, 1] / plotUnitsFlux, color='darkorange')
        if energyBinsPerDecade == 2:
            ax.annotate('ANITA I - IV',
                        xy=(7e9, 1e-6), xycoords='data',
                        horizontalalignment='left', color='darkorange',fontsize=legendfontsize)
        else:
            ax.annotate('ANITA I-IV',
                        xy=(9e9, 4e-7), xycoords='data',
                        horizontalalignment='left', color='darkorange',fontsize=legendfontsize)

    if show_auger_limit:
        ax.plot(auger_limit[:, 0] / plotUnitsEnergy, auger_limit[:, 1] / plotUnitsFlux, color='darkorange')
        if energyBinsPerDecade == 2:
            ax.annotate('Auger',
                        xy=(1.1e8, 2.1e-7), xycoords='data',
                        horizontalalignment='left', color='darkorange', rotation=0,fontsize=legendfontsize)
        else:
            ax.annotate('cosmic rays \n (Auger)',
                        xy=(2e8, 4e-8), xycoords='data',
                        horizontalalignment='left', color='darkorange', rotation=0,fontsize=legendfontsize+2,fontweight='semibold')

    if show_auger_cr:
        ax.errorbar(auger_cr[:,0] / plotUnitsEnergy,auger_cr[:,1] / plotUnitsFlux, yerr= auger_cr_uncert.T/ plotUnitsFlux, linestyle='None',marker='o',color='darkorange')
        ax.annotate('cosmic rays \n (Auger)',
                        xy=(2e10, 3e-7), xycoords='data',
                        horizontalalignment='right', color='darkorange', rotation=0,fontsize=legendfontsize+2,fontweight='semibold')

    if show_ara:
        ax.plot(ara_1year[:,0]/plotUnitsEnergy,ara_1year[:,1]/ plotUnitsFlux,color='indigo')
#         ax.plot(ara_4year[:,0]/plotUnitsEnergy,ara_4year[:,1]/ plotUnitsFlux,color='indigo',linestyle='--')
        if energyBinsPerDecade == 2:
            ax.annotate('ARA ',
                        xy=(5e8, 6e-7), xycoords='data',
                        horizontalalignment='left', color='indigo', rotation=0,fontsize=legendfontsize)
        else:
            ax.annotate('ARA',
                    xy=(2.05e10, 1.35e-6), xycoords='data',
                    horizontalalignment='left', color='indigo', rotation=0,fontsize=legendfontsize)

    if show_arianna:
        ax.plot(arianna[:,0]/plotUnitsEnergy,arianna[:,1]/ plotUnitsFlux,color='deeppink')
        ax.annotate('ARIANNA',
                    xy=(2.9e9, 2.5e-6), xycoords='data',
                    horizontalalignment='left', color='deeppink', rotation=0,fontsize=legendfontsize)

    if show_fermi:
        ax.errorbar(fermi_diffuse_central[:,0]/plotUnitsEnergy,fermi_diffuse_central[:,1]/ plotUnitsFlux,linestyle='None',marker='<',yerr=fermi_errors.T/ plotUnitsFlux,color='firebrick')

        ax.annotate(r'$\gamma$-ray background' "\n" '(Fermi)',
                    xy=(5e1, 3e-7), xycoords='data',
                    horizontalalignment='left', color='firebrick', rotation=0,fontsize=legendfontsize+2,fontweight='bold')


    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_xlabel(r'Particle energy [GeV]')
    ax.set_ylabel(r'$E^2 \Phi$ (all-flavor) [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]')

    ax.set_ylim(3e-11, 1e-6)
    ax.set_xlim(10, 4e11)


    plt.tight_layout()
    return fig, ax



# Make the figure as you would like it

if __name__=="__main__":


    fig, ax = get_E2_limit_figure(
                    show_ice_cube_EHE_limit=True,
                    show_ice_cube_HESE_fit=False,
                    show_ice_cube_HESE_data=True,
                    show_ice_cube_mu=True,
                    show_anita_limit=False,
                    show_auger_limit=False,
                    show_neutrino_best_fit=False,
                    show_neutrino_best_case=True,
                    show_neutrino_worst_case=True,
                    show_ara=False,
                    show_arianna=False,
                    show_grand_10k=False,
                    show_grand_200k=False,
                    show_radar=False,
                    show_fermi=True)


    name_plot = "MM_Figure_RNO_Concept_Paper.pdf"
    plt.savefig(name_plot)

    plt.show()
