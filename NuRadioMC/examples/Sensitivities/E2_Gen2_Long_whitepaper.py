import matplotlib
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rc('axes', labelsize=20)
legendfontsize = 13

import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.utilities import units
from NuRadioMC.utilities import fluxes
import os
from scipy.interpolate import interp1d

energyBinsPerDecade = 1.
plotUnitsEnergy = units.GeV
plotUnitsFlux = units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
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

GRAND_10k *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
#GRAND_10k /= 2 #halfdecade bins
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
#GRAND_200k /= 2 #halfdecade bins
GRAND_200k *= energyBinsPerDecade

#Update from Mauricio (see mail). 1:1:1 flavour ratio, 1 decade energy bins and 10 years

GRAND_200k_10 = np.loadtxt('sensitivity_grand.dat')
GRAND_200k_10[:,0] = 10**(GRAND_200k_10[:,0])*units.GeV
GRAND_200k_10[:,4] = 10**(GRAND_200k_10[:,4])*(units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)


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

ice_cube_limit[:, 0] = 10**ice_cube_limit[:, 0] * units.GeV
ice_cube_limit[:, 1] = 10**ice_cube_limit[:, 1] * (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
ice_cube_limit[:, 1] *= energyBinsPerDecade


# HESE 8 years
#log(E (GeV)); log(E^2 dN/dE (GeV cm^-2 s-1 sr-1)); format x y -dy +dy
ice_cube_hese = np.array(([
(4.78516,	-7.63256,		0.223256,	0.167442),
(5.10938,	-7.66977,		0.139535,	0.102326),
(5.42969,	-8.36744,		0.930233,	0.297674),
(5.75391,	-8.51628,		0.2,       	0.	),
(6.07813,	-8.38605,		0.604651,	0.288372),
(6.39844,	-8.35814,		0.455814,	0.334884),
(6.72266,	-9.0,    		0.2 ,    	0)
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




# Ice cube

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
auger_limit[:,1] *= 3
auger_limit[:, 1] /= 2  # half-decade binning
auger_limit[:, 1] *= energyBinsPerDecade


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
# level not fully defined yet)

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

# ARIANNA HRA 2019

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

def get_E2_limit_figure(
                        show_ice_cube_EHE_limit=True,
                        show_ice_cube_HESE_data=True,
                        show_ice_cube_HESE_fit=True,
                        show_ice_cube_mu=True,
                        show_anita_I_III_limit=True,
                        show_auger_limit=True,
                        show_ara=True,
                        show_arianna=True,
                        show_neutrino_best_fit=True,
                        show_neutrino_best_case=True,
                        show_neutrino_worst_case=True,
                        show_grand_10k=True,
                        show_grand_200k=False,
                        show_radar=False):

    # Limit E2 Plot
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Theoretical models

    # Diffuse Flux
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

    #Sources

    tde = np.loadtxt(os.path.join(os.path.dirname(__file__),"TDEneutrinos.txt"))
    tde[:,0] *= units.GeV
    tde[:,1] *= units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
    tde[:,1] *= 3 # all flavor
    tde_label = "TDE, Biehl et al. (1711.03555)"
    tde_int = interp1d(tde[:,0],tde[:,1])

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

    pulsars = np.loadtxt(os.path.join(os.path.dirname(__file__),"Pulsar_Fang+_2014.txt"))
    pulsars[:,0] *= units.GeV
    pulsars[:,1] *= units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
    pulsars[:,2] *= units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
    pulsars_label = "Pulsar, Fang et al. (1311.2044)"
    pulsars_int = interp1d(pulsars[:,0],np.average((pulsars[:,1],pulsars[:,2]),axis=0))

    clusters = np.loadtxt(os.path.join(os.path.dirname(__file__),"cluster_Fang_Murase_2018.txt"))
    clusters[:,0] *= units.GeV
    clusters[:,1] *= units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
    clusters_label = "Clusters, Fang & Murase, (1704.00015)"
    clusters_int = interp1d(clusters[:,0],clusters[:,1])

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

    data_ns_merger[:,0] *= units.GeV
    data_ns_merger[:,1] *= units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
    data_ns_merger_label = 'NS-NS merger, Fang & Metzger 1707.04263'
    data_ns_merger_int = interp1d(data_ns_merger[:,0],data_ns_merger[:,1])


    grb_afterglow = np.loadtxt(os.path.join(os.path.dirname(__file__),"grb_ag_wind_optimistic.dat"))
    grb_afterglow[:,0] = 10**grb_afterglow[:,0]*units.GeV
    grb_afterglow[:,1] = 10**grb_afterglow[:,1]*units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
    grb_afterglow_label = "GRB afterglow, Murase 0707.1140"
    grb_afterglow_int = interp1d(grb_afterglow[:,0],grb_afterglow[:,1])

    agn = np.loadtxt(os.path.join(os.path.dirname(__file__),"agn_murase.dat"),delimiter=',')
    agn[:,0] = agn[:,0]*units.GeV
    agn[:,1] = agn[:,1]*units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1
    #agn_label = "AGN, Murase et al. 1403.4089"
    agn_label = "From AGN"
    agn_int = interp1d(agn[:,0],agn[:,1])

    #plot_energy = np.arange(1e5,1e11,3e5)*units.GeV
    plot_energy = np.array([1e5,1.1e5,1.2e5,1.5e5,1.7e5,2e5,3e5,4e5,6e5,7e5,8e5,9e5,
                            1e6,1.1e6,1.2e6,1.5e6,1.7e6,2e6,3e6,4e6,6e6,7e6,8e6,9e6,
                            1e7,1.1e7,1.2e7,1.5e7,1.7e7,2e7,3e7,4e7,6e7,7e7,8e7,9e7,
                            1e8,1.1e8,1.2e8,1.5e8,1.7e8,2e8,3e8,4e8,6e8,7e8,8e8,9e8,
                            1e9,1.1e9,1.2e9,1.5e9,1.7e9,2e9,3e9,4e9,6e9,7e9,8e9,9e9,
                            1e10,1.1e10,1.2e10,1.5e10,1.7e10,2e10,3e10,4e10,6e10,7e10,8e10,9e10])*units.GeV

    max_all_models = np.maximum(vanVliet_max_int(plot_energy),Heinze_evo_int(plot_energy))
#
    max_plot = ax.fill_between(plot_energy/plotUnitsEnergy,max_all_models/plotUnitsFlux,1e-11, color='0.9',label='From UHECR, allowed region')

    best_fit_plot, = ax.plot(plot_energy/plotUnitsEnergy,Heinze_band_int(plot_energy)/plotUnitsFlux,c='0.4',linestyle='--',label= 'From UHECR, best fit')

    agn_plot, = ax.plot(plot_energy/plotUnitsEnergy,agn_int(plot_energy)/plotUnitsFlux,c='0.4',linestyle='-.',label=agn_label)
    #dashes=[10, 5, 20, 5]

    first_legend = plt.legend(handles=[agn_plot,best_fit_plot,max_plot], loc=4,fontsize=legendfontsize)

#
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
#         ax.plot(GRAND_energy / plotUnitsEnergy, GRAND_200k *3 /5. / plotUnitsFlux, linestyle=":", color='saddlebrown')
#         ax.annotate('GRAND 200k',
#                     xy=(5e9, 6e-10), xycoords='data',
#                     horizontalalignment='left', color='saddlebrown', rotation=40,fontsize=legendfontsize)
        # 10 years
        ax.plot(GRAND_200k_10[:,0] / plotUnitsEnergy, GRAND_200k_10[:,4] / plotUnitsFlux, linestyle=":", color='saddlebrown')
        ax.annotate('GRAND 200k',
                    xy=(6e9, 7e-10), xycoords='data',
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
            ax.annotate('IceCube',
                    xy=(4e6, 4e-8), xycoords='data',
                    horizontalalignment='center', color='dodgerblue', rotation=0,fontsize=legendfontsize, fontweight='semibold')

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


    if show_anita_I_III_limit:
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
        ax.plot(auger_limit[:, 0] / plotUnitsEnergy, auger_limit[:, 1] / plotUnitsFlux, color='forestgreen')
        if energyBinsPerDecade == 2:
            ax.annotate('Auger',
                        xy=(1.1e8, 2.1e-7), xycoords='data',
                        horizontalalignment='left', color='forestgreen', rotation=0,fontsize=legendfontsize)
        else:
            ax.annotate('Auger',
                        xy=(2e8, 4e-8), xycoords='data',
                        horizontalalignment='left', color='forestgreen', rotation=0,fontsize=legendfontsize,fontweight='semibold')

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

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.annotate(r'$\nu_e : \nu_{\mu} : \nu_\tau = 1 : 1 : 1$',
                        xy=(1.3e5, 8e-7), xycoords='data',
                        horizontalalignment='left', color='k', rotation=0,fontsize=legendfontsize-2, fontweight='bold')

    ax.set_xlabel(r'Neutrino energy [GeV]')
    ax.set_ylabel(r'$E^2 \Phi$ (all-flavor)[GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]')


    ax.set_ylim(1.e-11, 4e-6)
    ax.set_xlim(1e5, 1e11)


    plt.tight_layout()
    return fig, ax

def add_limit(ax, limit_labels, E, Veff, n_stations, label, livetime=3*units.year, linestyle='-',color='r',linewidth=3):
    E = np.array(E)
    Veff = np.array(Veff)

    if Veff.shape[0] != 2:

        limit = fluxes.get_limit_e2_flux(energy = E,
                                         veff_sr = Veff,
                                         livetime = livetime,
                                         signalEff = n_stations,
                                         energyBinsPerDecade=energyBinsPerDecade,
                                         upperLimOnEvents=2.44,
                                         nuCrsScn='ctw')


        _plt, = ax.plot(E/plotUnitsEnergy,limit/ plotUnitsFlux, linestyle=linestyle, color=color,
                        label="{1} ({0} years)".format(int(livetime/units.year),label),
                        linewidth=linewidth)

        limit_labels.append(_plt)
    else:

        limit_lower = fluxes.get_limit_e2_flux(energy = E,
                                         veff_sr = Veff[0],
                                         livetime = livetime,
                                         signalEff = n_stations,
                                         energyBinsPerDecade=energyBinsPerDecade,
                                         upperLimOnEvents=2.44,
                                         nuCrsScn='ctw')
        limit_upper = fluxes.get_limit_e2_flux(energy = E,
                                         veff_sr = Veff[1],
                                         livetime = livetime,
                                         signalEff = n_stations,
                                         energyBinsPerDecade=energyBinsPerDecade,
                                         upperLimOnEvents=2.44,
                                         nuCrsScn='ctw')

        plt1 = ax.fill_between(E/plotUnitsEnergy,limit_upper/ plotUnitsFlux,
        limit_lower/plotUnitsFlux,color=color,alpha=0.2)


    return limit_labels


# PyREx 1.5 sigma 60 m (trigger level, km3sr)
# En	Veff		Error
pyrex_60 = np.array((
#[15.5,	8.031E-3,	3.583E-3],
[16.0,	6.989E-2,	9.930E-3],
[16.5,	3.711E-1,	2.965E-2],
[17.0,	1.290E+0,	5.528E-2],
[17.5,	3.250E+0,	1.396E-1],
[18.0,	7.013E+0,	2.509E-1],
[18.5,	1.249E+1,	4.185E-1],
[19.0,	2.290E+1,	5.823E-1],
[19.5,	4.273E+1,	9.290E-1],
#[20.0,	6.479E+1,	1.144E+0]
))

pyrex_60[:,0] = 10**pyrex_60[:,0] *units.eV
pyrex_60[:,1] *= units.km**3
pyrex_60[:,2] *= units.km**3

pyrex_60_old = np.array((
#[15.5,	8.031E-3,	3.583E-3],
[16.0,	6.989E-2,	9.930E-3],
[16.5,	3.711E-1,	2.965E-2],
[17.0,	1.290E+0,	5.528E-2],
[17.5,	3.250E+0,	1.396E-1],
[18.0,	7.013E+0,	2.509E-1],
[18.5,	1.249E+1,	4.185E-1],
[19.0,	2.290E+1,	5.823E-1],
[19.5,	4.273E+1,	9.290E-1],
#[20.0,	6.479E+1,	1.144E+0]
))

pyrex_60_old[:,0] = 10**pyrex_60_old[:,0] *units.eV
pyrex_60_old[:,1] *= units.km**3
pyrex_60_old[:,2] *= units.km**3


# PyREx 1.5 sigma 100 m (trigger level, km3sr)
# En	Veff		Error
pyrex_100_old = np.array((
#[15.5,	9.033E-3,	4.249E-3], #OLD FOR RNO Proposal
[16.0,	6.816E-2,	9.530E-3],
[16.5,	5.060E-1,	3.370E-2],
[17.0,	1.717E+0,	6.563E-2],
[17.5,	4.931E+0,	1.719E-1],
[18.0,	1.056E+1,	3.163E-1],
[18.5,	1.971E+1,	5.258E-1],
[19.0,	3.530E+1,	7.036E-1],
[19.5,	6.863E+1,	1.177E+0],
#[20.0,	1.083E+2,	1.519E+0]
))

pyrex_100_old[:,0] = 10**pyrex_100_old[:,0] *units.eV
pyrex_100_old[:,1] *= units.km**3
pyrex_100_old[:,2] *= units.km**3

pyrex_100 = np.array((
[16.0,  	5.67397e-02,	2.60831e-02],
[16.5,  	4.13495e-01,	3.54847e-02],
[17.0,  	1.42276e+00,	5.65013e-02],
[17.5,  	3.71839e+00,	1.37013e-01],
[18.0,  	7.55479e+00,	2.60396e-01],
[18.5,  	1.47662e+01,	4.55059e-01],
[19.0,  	3.07369e+01,	6.56543e-01],
[19.5,  	5.79992e+01,	1.08224e+00],
[20.0,  	9.06216e+01,	1.35279e+00]
))

pyrex_100[:,0] = 10**pyrex_100[:,0] *units.eV
pyrex_100[:,1] *= units.km**3
pyrex_100[:,2] *= units.km**3

if __name__=="__main__":


    fig, ax = get_E2_limit_figure(
                    show_ice_cube_EHE_limit=True,
                    show_ice_cube_HESE_fit=False,
                    show_ice_cube_HESE_data=True,
                    show_ice_cube_mu=True,
                    show_anita_I_III_limit=True,
                    show_auger_limit=True,
                    show_neutrino_best_fit=False,
                    show_neutrino_best_case=True,
                    show_neutrino_worst_case=True,
                    show_ara=True,
                    show_grand_10k=False,
                    show_grand_200k=True,
                    show_radar=False)

    # Add predictions and limits to the figure

    labels =[]


    import json

    show_prediction_rno_g = False
    show_prediction_arianna_200 = False

    show_gen2_5_years = False
    show_gen2_10_years = True

    with open('Gen2_Radio_100m_new.json', 'r') as input:
        Gen2_100_full = json.load(input)

    with open('Gen2_Radio_60m.json', 'r') as input:
        Gen2_60_full = json.load(input)

    Gen2_100 =  Gen2_100_full['0.0']['simple_threshold']
    Gen2_60 =  Gen2_60_full['0.0']['simple_threshold']

    energies_Gen2_100 = np.array(Gen2_100['energies'])*units.eV
    V_eff_Gen2_100 = np.array(Gen2_100['Veff'])* units.m**3
    energies_Gen2_60 = np.array(Gen2_60['energies'])*units.eV
    V_eff_Gen2_60 = np.array(Gen2_60['Veff'])* units.m**3
    mask = np.where(energies_Gen2_100 > 1e16*units.eV)

    #DATA files from Daniel, Gen2 simulation, Aug 27th 2019

    #90% CL, UL, no background
    # Central energy of the bin [GeV]
    energies_Gen2_10years = np.array([316.22776601683796, 3162.2776601683795, 31622.776601683792, 316227.7660168379,
                                    3162277.6601683795, 31622776.60168379, 316227766.01683795, 3162277660.1683793,
                                    31622776601.683792]) * units.GeV
    # 90%CL upper limit flux [GeV cm^-2 s^-1 sr^-1]
    limit_Gen2_10_years = np.array([3.580680188634664e-08, 5.910247460729904e-11, 1.7720472758164112e-11,
                 2.0103976252003292e-11, 5.291928324771859e-11, 1.4091622570204057e-10, 1.0491152881916025e-10,
                 1.4292553332972088e-10, 2.902108529905861e-10]) * units.cm**-2 * units.GeV * units.s**-1 * units.sr**-1


    if show_gen2_5_years:

        gen2_100_limit_5 = fluxes.get_limit_e2_flux(energy = energies_Gen2_100[mask],
                                         veff_sr = V_eff_Gen2_100[mask],
                                         livetime = 5*units.year,
                                         signalEff = 200,
                                         energyBinsPerDecade=energyBinsPerDecade,
                                         upperLimOnEvents=2.44,
                                         nuCrsScn='ctw')

#THIS IS THE GENERIC PATHFINDER AT SOUTHPOLE, shaded region for uncertainties in trigger, design etc.

        _plt2a = ax.fill_between(energies_Gen2_100[mask]/plotUnitsEnergy,gen2_100_limit_5/plotUnitsFlux/0.8,gen2_100_limit_5/plotUnitsFlux/1.4, color='r',label='IceCube-Gen2 radio (5 years)',alpha=0.5)

        print("Gen2, 5 years")
        print(energies_Gen2_100[mask]/plotUnitsEnergy,gen2_100_limit_5/plotUnitsFlux)
        labels.append(_plt2a)

    if show_gen2_10_years:

        gen2_100_limit_10 = fluxes.get_limit_e2_flux(energy = energies_Gen2_100[mask],
                                         veff_sr = V_eff_Gen2_100[mask],
                                         livetime = 10*units.year,
                                         signalEff = 200,
                                         energyBinsPerDecade=energyBinsPerDecade,
                                         upperLimOnEvents=2.44,
                                         nuCrsScn='ctw')

        #THIS IS THE GENERIC PATHFINDER AT SOUTHPOLE, shaded region for uncertainties in trigger, design etc.

        _plt2b = ax.fill_between(energies_Gen2_100[mask]/plotUnitsEnergy,gen2_100_limit_10/plotUnitsFlux/0.8,gen2_100_limit_10/plotUnitsFlux/1.2, color='r',label='Gen2-Radio (10 years)',alpha=0.5)

        print("Gen2, 10 years")
        print(energies_Gen2_100[mask]/plotUnitsEnergy,gen2_100_limit_10/plotUnitsFlux)
        labels.append(_plt2b)


    # RNO-G simulations
    if show_prediction_rno_g:

        with open('Greenland_100m_veff_sr.json', 'r') as input:
            Greenland = json.load(input)


        energies_Greenland = np.array(Greenland['energies'])*units.eV
        V_eff_Greenland= np.array(Greenland['Veff'])* units.m**3

        mask = np.where(energies_Greenland > 1e16*units.eV)

        greenland_limit = fluxes.get_limit_e2_flux(energy = energies_Greenland[mask],
                                             veff_sr = V_eff_Greenland[mask],
                                             livetime = 5*units.year*0.7,
                                             signalEff = 35,
                                             energyBinsPerDecade=energyBinsPerDecade,
                                             upperLimOnEvents=2.44,
                                             nuCrsScn='ctw')

        _plt3, = ax.plot(energies_Greenland[mask]/plotUnitsEnergy, greenland_limit/plotUnitsFlux,color='r',linewidth=3,label='RNO-G (5 years)')
        labels.append(_plt3)
        print("RNO-G, 5 years, 0.7 efficiency uptime")
        print(energies_Greenland[mask]/plotUnitsEnergy, greenland_limit/plotUnitsFlux)


    if show_prediction_arianna_200:
        # 10 year sensitivity, scales to 5 years
        arianna_200 = np.loadtxt("expected_sensivity_ARIANNAA-200.txt")

        print("ARIANNA")
        print(arianna_200.shape)

        _plt4, = ax.plot(arianna_200[:,0],arianna_200[:,1]*2, label='ARIANNA-200 (5 years)',color='maroon',linewidth=2)


        labels.append(_plt4)


    plt.legend(handles=labels, loc=2,fontsize=legendfontsize)

#     ax.annotate("IceCube-Gen2",
#                         xy=(3.e5, 1.4e-11), xycoords='data',
#                         horizontalalignment='left', color='k', rotation=0,fontsize=legendfontsize-2, alpha=0.5)
    name_plot = "Sensitivities_Long_WhitePaper.pdf"
    plt.savefig(name_plot)

    plt.show()
