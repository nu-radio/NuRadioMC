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
from scipy.interpolate import interp1d

local_dir = os.path.abspath(os.path.dirname(__file__))

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
GRAND_10k_data = np.loadtxt(f"{local_dir}/experiments/grand10k_sensitivity.txt")
GRAND_10k_energy = GRAND_10k_data[:,0] 
GRAND_10k_energy *= units.GeV

GRAND_10k = GRAND_10k_data[:,1]
GRAND_10k *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
GRAND_10k /= GRAND_10k_energy
GRAND_10k *= energyBinsPerDecade
# The expected sensitivities for GRAND are given for 3 years, rescaling them to 10 years
GRAND_10k *= 3 / 10

GRAND_200k_data = np.loadtxt(f"{local_dir}/experiments/grand200k_sensitivity.txt")
GRAND_200k_energy = GRAND_200k_data[:,0] 
GRAND_200k_energy *= units.GeV

GRAND_200k = GRAND_200k_data[:,1]
GRAND_200k *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
GRAND_200k /= GRAND_200k_energy
GRAND_200k *= energyBinsPerDecade
# The expected sensitivities for GRAND are given for 3 years, rescaling them to 10 years
GRAND_200k *= 3 / 10

# RADAR proposed from https://arxiv.org/pdf/1710.02883.pdf

Radar = np.loadtxt(f"{local_dir}/experiments/radar_sensitivity.txt") 
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
ice_cube_limit_18 = np.loadtxt(f"{local_dir}/experiments/icecube_ehe18_limit.txt")
ice_cube_limit_18[:, 0] = 10 ** ice_cube_limit_18[:, 0] * units.GeV
ice_cube_limit_18[:, 1] = 10 ** ice_cube_limit_18[:, 1] * (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
ice_cube_limit_18[:, 1] *= energyBinsPerDecade

# Fig. 2 from PoS ICRC2017 (2018) 981
# IceCube preliminary
# E (GeV); E^2 dN/dE (GeV cm^-2 s-1 sr-1); yerror down; yerror up

# HESE 6 years
# ice_cube_hese = np.loadtxt(f"{local_dir}/experiments/icecube_hese_icrc17.txt")
# ice_cube_hese[:, 0] = ice_cube_hese[:, 0] * units.GeV
# ice_cube_hese[:, 1] = ice_cube_hese[:, 1] * (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
# ice_cube_hese[:, 1] *= 3
# ice_cube_hese[:, 2] = ice_cube_hese[:, 2] * (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
# ice_cube_hese[:, 2] *=  3
# ice_cube_hese[:, 3] = ice_cube_hese[:, 3] * (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
# ice_cube_hese[:, 3] *= 3

# HESE 8 years
# log(E (GeV)); log(E^2 dN/dE (GeV cm^-2 s-1 sr-1)); format x y -dy +dy
ice_cube_hese = np.loadtxt(f"{local_dir}/experiments/icecube_hese_8years.txt") 

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
nu_mu_data = np.loadtxt(f"{local_dir}/experiments/icecube_numu_9.5years.txt")
# convert energy to correct units
nu_mu_data[:, 0] = 10 ** nu_mu_data[:, 0] * units.GeV
nu_mu_data[:, 1:] = (10 ** nu_mu_data[:, 1:]) * 3 * (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)  # convert from single flavor to all flavor limit
nu_mu_data[:, 2] = np.abs(nu_mu_data[:, 1] - nu_mu_data[:, 2])
nu_mu_data[:, 3] = np.abs(nu_mu_data[:, 1] - nu_mu_data[:, 3])



# IceCube
# 2025 EHE paper. Data available at  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JHK49D
ice_cube_limit_25 = np.loadtxt(f"{local_dir}/experiments/icecube_ehe25_limit.txt")
ice_cube_limit_25[:, 0] = ice_cube_limit_25[:, 0] * units.GeV
ice_cube_limit_25[:, 1] = ice_cube_limit_25[:, 1] * (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
ice_cube_limit_25[:, 1] *= energyBinsPerDecade


# ApJ slope=-2.13, offset=0.9 (https://arxiv.org/pdf/1607.08006.pdf)
# ICR2017 slope=-2.19, offset=1.01 (https://pos.sissa.it/301/1005/)
# ICRC2019 slope=2.28, offset=1.44
# 9.5 years analysis 2.37+0.09−0.09, offset 1.44 + 0.25 - 0.26, Astrophysical normalization @ 100TeV: 1.44+0.25−0.26 × 10−18 GeV−1cm−2s−1 sr−1
nu_mu_slope = -2.37
nu_mu_slope_up = -(2.37 + 0.09)
nu_mu_slope_down = -(2.37 - 0.09)
nu_mu_offset = 1.44
nu_mu_offset_up = 1.44 + 0.25
nu_mu_offset_down = 1.44 - 0.26
nu_mu_show_data_points = True


## km3-230213A
# From https://www.nature.com/articles/s41586-024-08543-1

KM3_230231A_data = np.loadtxt(f"{local_dir}/experiments/km3net25_flux.txt")
KM3_230231A_E = KM3_230231A_data[0]  * units.PeV
KM3_230231A_E_err = KM3_230231A_data[1:3] * units.PeV
KM3_230231A_flux = KM3_230231A_data[3]  * (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr **-1)
KM3_230231A_flux_err = KM3_230231A_data[4:] * (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr **-1)



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


i3_glashow_data = np.genfromtxt(f"{local_dir}/experiments/icecube_glashow.csv",
    skip_header=2, delimiter=',', names=['E_min', 'E_max', 'y', 'y_lower', 'y_upper'])
i3_glashow_emin = i3_glashow_data['E_min'] * units.GeV / plotUnitsEnergy
i3_glashow_emax = i3_glashow_data['E_max'] * units.GeV / plotUnitsEnergy
i3_glashow_y = 3. * i3_glashow_data['y'] * (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1) * 1E-8 / plotUnitsFlux
i3_glashow_y_lower = 3. * i3_glashow_data['y_lower'] * (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1) * 1E-8 / plotUnitsFlux
i3_glashow_y_upper = 3. * i3_glashow_data['y_upper'] * (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1) * 1E-8 / plotUnitsFlux

# BEACON high frequency and low frequency design, 3 years, half decade, all flavor
BEACON_LF_100_data = np.loadtxt(f"{local_dir}/experiments/beacon_lf_100_sensitivity.txt")
BEACON_LF_1000_data = np.loadtxt(f"{local_dir}/experiments/beacon_lf_1000_sensitivity.txt")
BEACON_HF_100_data = np.loadtxt(f"{local_dir}/experiments/beacon_hf_100_sensitivity.txt")
BEACON_HF_1000_data = np.loadtxt(f"{local_dir}/experiments/beacon_hf_1000_sensitivity.txt")

BEACON_LF_100_energy = BEACON_LF_100_data[:,0] * units.GeV
BEACON_LF_100 = BEACON_LF_100_data[:,1] 

BEACON_LF_1000_energy = BEACON_LF_1000_data[:,0] * units.GeV
BEACON_LF_1000 = BEACON_LF_1000_data[:,1]

BEACON_HF_100_energy = BEACON_HF_100_data[:,0] * units.GeV
BEACON_HF_100 = BEACON_HF_100_data[:,1]

BEACON_HF_1000_energy = BEACON_HF_1000_data[:,0] * units.GeV
BEACON_HF_1000 = BEACON_HF_1000_data[:,1]

BEACON_LF_100 *= (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
BEACON_LF_100 /= 2  # half-decade energy bins
BEACON_LF_100 *= 3 * units.year / (10 * units.year)
BEACON_LF_1000 *= (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
BEACON_LF_1000 /= 2  # half-decade energy bins
BEACON_LF_1000 *= 3 * units.year / (10 * units.year)
BEACON_HF_100 *= (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
BEACON_HF_100 /= 2  # half-decade energy bins
BEACON_HF_100 *= 3 * units.year / (10 * units.year)
BEACON_HF_1000 *= (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
BEACON_HF_1000 /= 2 * energyBinsPerDecade  # half-decade energy bins
BEACON_HF_1000 *= 3 * units.year / (10 * units.year)

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
anita_limit = np.loadtxt(f"{local_dir}/experiments/anita_i_iii_limit.txt")
anita_limit[:, 0] *= units.eV
anita_limit[:, 1] *= (units.cm ** -2 * units.second ** -1 * units.sr ** -1)
anita_limit[:, 1] *= anita_limit[:, 0] # convert to E^2*dN/dE
anita_limit[:, 1] *= (4 / np.log(10))  # see discussion above about strange anita binning
anita_limit[:, 1] *= energyBinsPerDecade

# ANITA I - IV
# https://arxiv.org/abs/1902.04005
# Phys. Rev. D 99, 122001 (2019)
# NB: The ANITA I-IV is indeed weaker than the ANITA I-III limit (!!)
# The reason is not understood, but can be seen easily comparing the two limits side-by-side
anita_i_iv_limit = np.loadtxt(f"{local_dir}/experiments/anita_i_iv_limit.txt") 
anita_i_iv_limit[:, 0] *= units.eV
anita_i_iv_limit[:, 1] *= (units.eV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
anita_i_iv_limit[:, 1] *= (4 / np.log(10))  # see discussion above about strange anita binning
anita_i_iv_limit[:, 1] *= energyBinsPerDecade

# From 2020 PUEO whitepaper
# 30 day livetime
pueo_30 = np.loadtxt(f"{local_dir}/experiments/pueo30_sensitivity.txt")

PUEO30_energy = pueo_30[:, 0] * units.eV
PUEO30 = pueo_30[:, 1]
PUEO30 *= PUEO30_energy / units.GeV * (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
PUEO30 *= (4 / np.log(10))  # see discussion above about anita binning
PUEO30 *= 2.44  # convert from single event sensitivty to 90% confidence level
PUEO30 *= energyBinsPerDecade

# 100 day livetime
pueo_100 = np.loadtxt(f"{local_dir}/experiments/pueo100_sensitivity.txt") 

PUEO100_energy = pueo_100[:, 0] * units.eV
PUEO100 = pueo_100[:, 1]
PUEO100 *= PUEO100_energy / units.GeV * (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
PUEO100 *= (4 / np.log(10))  # see discussion above about anita binning
PUEO100 *= 2.44  # convert from single event sensitivty to 90% confidence level
PUEO100 *= energyBinsPerDecade

# TAROGE-M
# 10 stations, 5 year exposure, nutau only
# Log(Energy) GeV       Sensitivity*E^2 (GeV/cm^2 s sr)
taroge_m = np.loadtxt(f"{local_dir}/experiments/taroge_m_sensitivity.txt")
taroge_m_E = pow(10, taroge_m[::2]) * units.GeV
taroge_m_flux = taroge_m[1::2] * units.GeV * units.cm ** -2 * units.s ** -1
taroge_m_flux *= 3.0

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
auger_limit = np.loadtxt(f"{local_dir}/experiments/auger19_limit.txt") 
auger_limit[:, 0] *= units.eV
auger_limit[:, 1] *= (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
auger_limit[:, 1] /= 2  # half-decade binning
auger_limit[:, 1] *= 3  # correction for 3 flavors
auger_limit[:, 1] *= energyBinsPerDecade

# ARA Published 2sta x 1yr analysis level limit:
ara_1year = np.loadtxt(f"{local_dir}/experiments/ara_1year_limit.txt")
ara_1year[:, 0] *= units.eV
ara_1year[:, 1] *= (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
ara_1year[:, 1] /= 1  # binning is dLogE = 1
ara_1year[:, 1] *= energyBinsPerDecade

# Analysis from https://doi.org/10.1103/PhysRevD.102.043021  https://arxiv.org/abs/1912.00987
# 2 stations (A2 and A3), approx 1100 days of livetime per station
ara_4year_E, ara_4year_limit, t1, t2 = np.loadtxt(f"{local_dir}/experiments/limit_a23.txt", unpack=True)
ara_4year_E *= units.eV
ara_4year_limit *= units.eV * units.cm ** -2 * units.second ** -1 * units.sr ** -1
ara_4year_limit *= energyBinsPerDecade

# ARA phased array 6-month limit
ara_PA_6month_E, ara_PA_6month_limit, t1, t2 = np.loadtxt(f"{local_dir}/experiments/limit_ARA_PA_6months.txt", unpack=True)
ara_PA_6month_E *= units.eV
ara_PA_6month_limit *= units.eV * units.cm ** -2 * units.second ** -1 * units.sr ** -1
ara_PA_6month_limit *= energyBinsPerDecade

# ARIANNA HRA
ARIANNA_HRA = np.loadtxt(f"{local_dir}/experiments/arianna_hra_limit.txt") 
ARIANNA_HRA[:, 0] *= units.GeV
ARIANNA_HRA[:, 1] /= 1
ARIANNA_HRA[:, 1] *= (units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1)
ARIANNA_HRA[:, 1] *= energyBinsPerDecade



def get_TAGZK_flux(energy):
    """
    GZK neutrino flux from TA best fit from D. Bergmann privat communications
    """

    TA_data = np.loadtxt(f"{local_dir}/data/TA_combined_fit_m3.txt")
    E = TA_data[:, 0] * units.GeV
    f = TA_data[:, 1] * plotUnitsFlux / E ** 2
    get_TAGZK_flux = interp1d(E, f, bounds_error=False, fill_value="extrapolate")
    return get_TAGZK_flux(energy)


def get_TAGZK_flux_ICRC2021(energy):
    """
    GZK neutrino flux from TA best fit ICRC2021
    https://pos.sissa.it/395/338/
    """
    TA_data = np.loadtxt(f"{local_dir}/data/TA_GZKprediction_ICRC2021.txt")
    E = TA_data[:, 0] * units.GeV
    f = TA_data[:, 1] * plotUnitsFlux / E ** 2
    get_TAGZK_flux = interp1d(E, f, bounds_error=False, fill_value="extrapolate")
    return get_TAGZK_flux(energy)


def get_proton_10(energy):
    """
    10% proton flux at source for astrophysical parameters determined by Auger data, by van Vliet et al.
    """
    vanVliet_reas = np.loadtxt(f"{local_dir}/data/ReasonableNeutrinos1.txt")
    E = vanVliet_reas[0,:] * units.GeV
    f = vanVliet_reas[1,:] * plotUnitsFlux / E ** 2
    getE = interp1d(E, f, bounds_error=False, fill_value="extrapolate")
    return getE(energy)


def get_GZK_Auger_best_fit(energy):
    Heinze_band = np.loadtxt(f"{local_dir}/data/talys_neu_bands.out")
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
                        show_ara_PA=False,
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
                        show_prediction_arianna_200=False,
                        show_PUEO_100=False,
                        show_beacon=False,
                        show_ice_cube_EHE_limit_18=False,  # old IC limit,
                        show_KM3_230213A = True
                        ):

    # Limit E2 Plot
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Neutrino Models
    # Version for a diffuse flux and for a source dominated flux
    if diffuse:
        legends = []
        # TA combined fit
        if(show_TA):
            TA_data_low = np.loadtxt(f"{local_dir}/models/TA_combined_fit_low_exp_uncertainty.txt")
            TA_data_high = np.loadtxt(f"{local_dir}/models/TA_combined_fit_high_exp_uncertainty.txt")
            TA_m3 = ax.fill_between(TA_data_low[:, 0] * units.GeV / plotUnitsEnergy,
                                     TA_data_low[:, 1], TA_data_high[:, 1],
                              label=r'UHECRs TA combined fit (1$\sigma$), Bergman et al.', color='C0', alpha=0.5, zorder=-1)
            legends.append(TA_m3)
        if(show_TA_nominal):
            TA_data = np.loadtxt(f"{local_dir}/models/TA_combined_fit_m3.txt")
            E = TA_data[:, 0] * units.GeV
            f = TA_data[:, 1] * plotUnitsFlux
            TA_nominal, = ax.plot(E / plotUnitsEnergy, f / plotUnitsFlux, "k-.", label="UHECRs TA combined fit, Bergman et al.")
            legends.append(TA_nominal)
        if(show_TA_ICRC2021):
            TA_data = np.loadtxt(f"{local_dir}/models/TA_GZKprediction_ICRC2021.txt")
            E = TA_data[:, 0] * units.GeV
            f = TA_data[:, 1] * plotUnitsFlux
            TA_nominal, = ax.plot(E / plotUnitsEnergy, f / plotUnitsFlux, "k-.", label="UHECRs TA combined fit, Bergman et al.")
            legends.append(TA_nominal)
        if(shower_Auger):

            vanVliet_max_1 = np.loadtxt(f"{local_dir}/models/MaxNeutrinos1.txt")
            vanVliet_max_2 = np.loadtxt(f"{local_dir}/models/MaxNeutrinos2.txt")
            vanVliet_reas = np.loadtxt(f"{local_dir}/models/ReasonableNeutrinos1.txt")

            vanVliet_max = np.maximum(vanVliet_max_1[1,:], vanVliet_max_2[1,:])

            # prot10, = ax.plot(vanVliet_reas[0,:] * units.GeV / plotUnitsEnergy, vanVliet_reas[1,:],
                              # label=r'10% protons in UHECRs (Auger), m=3.4, van Vliet et al.', linestyle='--', color='k')
            # legends.append(prot10)

            prot = ax.fill_between(vanVliet_max_1[0,:] * units.GeV / plotUnitsEnergy, vanVliet_max,
                                   vanVliet_reas[1,:] / 50, color='0.9', label=r'allowed from UHECRs (Auger), van Vliet et al.', zorder=-2)
            legends.append(prot)

        if(show_Heinze):
            Heinze_band = np.loadtxt(f"{local_dir}/models/talys_neu_bands.out")
#             best_fit, = ax.plot(Heinze_band[:, 0] * units.GeV / plotUnitsEnergy, Heinze_band[:, 1] * Heinze_band[:, 0] ** 2, c='k',
#                                 label=r'UHECR (Auger) combined fit, Heinze et al.', linestyle='-.')

#             Auger_bestfit = ax.fill_between(Heinze_band[:, 0],
#                                      Heinze_band[:, 2] * Heinze_band[:, 0] ** 2, Heinze_band[:, 3] * Heinze_band[:, 0] ** 2,
#                               label=r'UHECRs Auger combined fit, Heinze et al.', color='C1', alpha=0.5, zorder=1)

            Heinze_evo = np.loadtxt(f"{local_dir}/models/talys_neu_evolutions.out")
            Auger_bestfit = ax.fill_between(Heinze_evo[:, 0] * units.GeV / plotUnitsEnergy,
                                     Heinze_evo[:, 3] * Heinze_band[:, 0] ** 2, Heinze_evo[:, 4] * Heinze_band[:, 0] ** 2,
                              label=r'UHECRs Auger combined fit (3$\sigma$), Heinze et al.', color='C1', alpha=0.5, zorder=1)

#             Heinze_evo = np.loadtxt(f"{local_dir}/talys_neu_evolutions.out")
#             best_fit_3s, = ax.plot(Heinze_evo[:, 0] * units.GeV / plotUnitsEnergy, Heinze_evo[:, 6] * Heinze_evo[:, 0] **
#                             2, color='0.5', label=r'UHECR (Auger) combined fit + 3$\sigma$, Heinze et al.', linestyle='-.')
            legends.append(Auger_bestfit)
#             legends.append(best_fit_3s)

        first_legend = plt.legend(handles=legends, loc=4, fontsize=legendfontsize, handlelength=4)

        plt.gca().add_artist(first_legend)
    else:
        tde = np.loadtxt(f"{local_dir}/models/TDEneutrinos.txt")
        ll_grb = np.loadtxt(f"{local_dir}/models/LLGRBneutrinos.txt")
        pulsars = np.loadtxt(f"{local_dir}/models/Pulsar_Fang+_2014.txt")
        clusters = np.loadtxt(f"{local_dir}/models/cluster_Fang_Murase_2018.txt")

        # Fang & Metzger
        data_ns_merger = np.loadtxt(f"{local_dir}/models/ns_merger_Fang_Metzger.txt")

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
        ax.plot(GRAND_10k_energy / plotUnitsEnergy, GRAND_10k / plotUnitsFlux, linestyle=":", color='saddlebrown')
        if energyBinsPerDecade == 2:
            ax.annotate('GRAND 10k',
                            xy=(0.9e10 * units.GeV / plotUnitsEnergy, 2e-8), xycoords='data',
                            horizontalalignment='left', color='saddlebrown', rotation=50, fontsize=legendfontsize)
        else:
            ax.annotate('GRAND 10k',
                xy=(1.5e19 * units.eV / plotUnitsEnergy, 3.4e-8), xycoords='data',
                horizontalalignment='left', va="bottom", color='saddlebrown', rotation=40, fontsize=legendfontsize)

    if show_grand_200k:
        ax.plot(GRAND_200k_energy / plotUnitsEnergy, GRAND_200k / plotUnitsFlux, linestyle=":", color='saddlebrown',
                lw=2)
        ax.annotate('GRAND 200k',
                    xy=(1e10 * units.GeV / plotUnitsEnergy, 5e-10), xycoords='data',
                    horizontalalignment='left', color='saddlebrown', rotation=35, fontsize=legendfontsize,
                    )
    if show_radar:
        ax.fill_between(Radar[:, 0] / plotUnitsEnergy, Radar[:, 1] / plotUnitsFlux,
                        Radar[:, 2] / plotUnitsFlux, facecolor='None', hatch='x', edgecolor='0.8')
        ax.annotate('Radar',
                    xy=(1e9 * units.GeV / plotUnitsEnergy, 4.5e-8), xycoords='data',
                    horizontalalignment='left', color='0.7', rotation=45, fontsize=legendfontsize)

    if show_ice_cube_EHE_limit_18:
        ax.plot(ice_cube_limit_18[2:, 0] / plotUnitsEnergy, ice_cube_limit_18[2:, 1] / plotUnitsFlux, color='dodgerblue')
        if energyBinsPerDecade == 2:
            ax.annotate('IceCube18',
                    xy=(0.6e7 * units.GeV / plotUnitsEnergy, 2e-8), xycoords='data',
                    horizontalalignment='center', color='dodgerblue', rotation=0, fontsize=legendfontsize)
        else:
            ax.annotate('IceCube18',
                    xy=(3e6 * units.GeV / plotUnitsEnergy, 3e-8), xycoords='data',
                    horizontalalignment='center', color='dodgerblue', rotation=0, fontsize=legendfontsize)
    if show_ice_cube_EHE_limit:
        ax.plot(ice_cube_limit_25[2:, 0] / plotUnitsEnergy, ice_cube_limit_25[2:, 1] / plotUnitsFlux, color='dodgerblue')
        if energyBinsPerDecade == 2:
            ax.annotate('IceCube25',
                    xy=(0.6e8 * units.GeV / plotUnitsEnergy, 1.5e-8), xycoords='data',
                    horizontalalignment='center', color='dodgerblue', rotation=0, fontsize=legendfontsize)
        else:
            ax.annotate('IceCube25',
                    xy=(8e7 * units.GeV / plotUnitsEnergy, 7e-9), xycoords='data',
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
        # energy_placeholder = np.array(([1e14, 1e19])) * units.eV
        # plt.plot(energy_placeholder / plotUnitsEnergy,
        #          ice_cube_nu_fit(energy_placeholder) * energy_placeholder ** 2 / plotUnitsFlux,
        #          color='dodgerblue', linestyle=':')

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
                        xy=(1e19 * units.eV / plotUnitsEnergy, 1.2e-6), xycoords='data',
                        horizontalalignment='left', color='darkorange', fontsize=legendfontsize)

    if show_auger_limit:
        ax.plot(auger_limit[:, 0] / plotUnitsEnergy, auger_limit[:, 1] / plotUnitsFlux, color='forestgreen')
        if energyBinsPerDecade == 2:
            ax.annotate('Auger',
                        xy=(8e16 * units.eV / plotUnitsEnergy, 2.1e-7), xycoords='data',
                        horizontalalignment='left', color='forestgreen', rotation=0, fontsize=legendfontsize)
        else:
            ax.annotate('Auger',
                        xy=(1.5e17 * units.eV / plotUnitsEnergy, 7e-8), xycoords='data',
                        horizontalalignment='right', color='forestgreen', rotation=-45, fontsize=legendfontsize)

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
                    xy=(1e18* units.eV / plotUnitsEnergy, 0.7e-6), xycoords='data',
                    horizontalalignment='left', color='indigo', rotation=0, fontsize=legendfontsize)
    if show_ara_PA:
        ax.plot(ara_PA_6month_E / plotUnitsEnergy, ara_PA_6month_limit / plotUnitsFlux, color='blue', linestyle='-')
        if energyBinsPerDecade == 2:
            ax.annotate('ARA PA',
                        xy=(2.95E1 * units.eV / plotUnitsEnergy, 6e-7), xycoords='data',
                        horizontalalignment='left', color='grey', rotation=0, fontsize=legendfontsize)
        else:
            ax.annotate('ARA PA',
                    xy=(2.9E17 * units.eV / plotUnitsEnergy, 5.2e-6), xycoords='data',
                    horizontalalignment='left', color='blue', rotation=0, fontsize=legendfontsize)

    if show_arianna:
        ax.plot(ARIANNA_HRA[:, 0] / plotUnitsEnergy, ARIANNA_HRA[:, 1] / plotUnitsFlux, color='red')
#         ax.plot(ara_4year[:,0]/plotUnitsEnergy,ara_4year[:,1]/ plotUnitsFlux,color='indigo',linestyle='--')
        if energyBinsPerDecade == 2:
            ax.annotate('ARIANNA',
                        xy=(5e8 * units.GeV / plotUnitsEnergy, 6e-7), xycoords='data',
                        horizontalalignment='left', color='red', rotation=0, fontsize=legendfontsize)
        else:
            ax.annotate('ARIANNA',
                    xy=(1e7 * units.GeV / plotUnitsEnergy, 2e-6), xycoords='data',
                    horizontalalignment='left', color='red', rotation=0, fontsize=legendfontsize)

    if show_IceCubeGen2_whitepaper:
        # flux limit for 5 years
        gen2_data = np.loadtxt(f"{local_dir}/data/icecube_gen2_sensitivity.txt")
        gen2_E = gen2_data[:,0] * units.GeV
        gen2_flux = gen2_data[:,0] * units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1
        ax.plot(gen2_E / plotUnitsEnergy, gen2_flux / 2 / plotUnitsFlux, color='purple', linestyle="--")
        ax.annotate('IceCube-Gen2 radio',
                    xy=(.8e8 * units.GeV / plotUnitsEnergy, 1.6e-10), xycoords='data',
                    horizontalalignment='left', color='purple', rotation=0, fontsize=legendfontsize)

    if show_IceCubeGen2_ICRC2021:
        # https://pos.sissa.it/395/1183/
        # flux limit for 10 years
        gen2_E, gen2_flux = np.loadtxt(f"{local_dir}/data/Gen2radio_sensitivity_ICRC2021.txt")
        gen2_E *= units.eV
        gen2_flux *= units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1
        ax.plot(gen2_E / plotUnitsEnergy, gen2_flux / plotUnitsFlux, color='purple', linestyle="--")
        ax.annotate('IceCube-Gen2 radio',
                    xy=(.8e8 * units.GeV / plotUnitsEnergy, 1.3e-10), xycoords='data',
                    horizontalalignment='left', color='purple', rotation=0, fontsize=legendfontsize)
    if show_RNOG:
        # flux limit for 5 years
        RNOG_data = np.loadtxt(f"{local_dir}/experiments/rnog_sensitivity.txt")
        RNOG_E = RNOG_data[:,0] * units.GeV
        RNOG_flux = RNOG_data[:,1] * units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1
        ax.plot(RNOG_E / plotUnitsEnergy, RNOG_flux / 0.7 / 2 / plotUnitsFlux, color='red', linestyle="-.")  # uses 70% uptime from RNO-G whitepaper and resacling to 10years
        ax.annotate('RNO-G',
                    xy=(8e18 * units.eV / plotUnitsEnergy, 1.5e-8), xycoords='data',
                    horizontalalignment='left', va="top", color='red', rotation=10, fontsize=legendfontsize)

    if show_prediction_arianna_200:
        # 10 year sensitivity
        arianna_200 = np.loadtxt(f"{local_dir}/experiments/expected_sensivity_ARIANNA-200.txt")
        arianna_200[:, 0] *= units.GeV
        arianna_200[:, 1] *= units.GeV * units.cm ** -2 * units.s ** -1
        print(arianna_200)

        _plt4, = ax.plot(arianna_200[:, 0] / plotUnitsEnergy, arianna_200[:, 1] / plotUnitsFlux, label='ARIANNA-200 (5 years)', color='blue', linestyle="--")
        ax.annotate('ARIANNA-200',
                    xy=(.9e19 * units.eV / plotUnitsEnergy, 3.15e-9), xycoords='data',
                    horizontalalignment='left', color='blue', rotation=30, fontsize=legendfontsize)

#         labels.append(_plt4)
    if show_PUEO_100:
        ax.annotate('PUEO (3 flights)', xy=(3e18 * units.eV / plotUnitsEnergy, 2.1e-8),
                    xycoords='data', horizontalalignment='left', color='#EA5A06', rotation=0, fontsize=legendfontsize)
        ax.plot(PUEO100_energy / plotUnitsEnergy, PUEO100 / plotUnitsFlux, linestyle=(0, (3, 1, 1, 1, 1, 1)), color='#EA5A06', label='PUEO (3 flights, 100 days)',
                lw=2)

    if show_beacon:
        beaconleg, = ax.plot(BEACON_LF_1000_energy / plotUnitsEnergy, BEACON_LF_1000 / plotUnitsFlux,
                             linestyle="-.", color='#F97807', label='BEACON 1k',
                             lw=2)
        ax.annotate('BEACON-1k',
                    xy=(7e18 * units.eV / plotUnitsEnergy, 9e-10), xycoords='data',
                    horizontalalignment='left', verticalalignment="bottom", color='#F97807', rotation=35, fontsize=legendfontsize)
        # second_legend.append(beaconleg)

    if show_KM3_230213A:
        ax.errorbar(KM3_230231A_E / plotUnitsEnergy, KM3_230231A_flux / plotUnitsFlux, yerr=KM3_230231A_flux_err.reshape(2, 1) / plotUnitsFlux, xerr=KM3_230231A_E_err.reshape(2, 1) / plotUnitsEnergy, color='deeppink')
        ax.scatter(KM3_230231A_E / plotUnitsEnergy, KM3_230231A_flux / plotUnitsFlux, color='deeppink')
        ax.annotate('KM3-230213A', xy = (KM3_230231A_E / plotUnitsEnergy*1.1, KM3_230231A_flux/plotUnitsFlux*1.1), color='deeppink', fontsize=legendfontsize)

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
                                         nuCrsScn='hedis_bgr18')
        limit_upper = fluxes.get_limit_e2_flux(energy=E,
                                         veff_sr=Veffsr[1],
                                         livetime=livetime,
                                         signalEff=n_stations,
                                         energyBinsPerDecade=energyBinsPerDecade,
                                         upperLimOnEvents=2.44,
                                         nuCrsScn='hedis_bgr18')

        plt1 = ax.fill_between(E / plotUnitsEnergy, limit_upper / plotUnitsFlux,
        limit_lower / plotUnitsFlux, color=color, alpha=0.2)

    else:

        limit = fluxes.get_limit_e2_flux(energy=E,
                                         veff_sr=Veffsr,
                                         livetime=livetime,
                                         signalEff=n_stations,
                                         energyBinsPerDecade=energyBinsPerDecade,
                                         upperLimOnEvents=2.44,
                                         nuCrsScn='hedis_bgr18')

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
    veff = np.loadtxt(f"{local_dir}/veffs/punch_deep_pa_50m_veff.txt")
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
