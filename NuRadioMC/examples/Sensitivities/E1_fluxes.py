import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.utilities import units
from NuRadioMC.utilities import fluxes
import os

local_dir = os.path.abspath(os.path.dirname(__file__))

# What to plot
save_figure_as = "E1_figure.pdf"
# ----------------------------------------
# Our simulations
show_strawman = False
N_strawman = [300]

show_strawman_pa = True
N_strawman_pa = [200]

show_punch = False
N_punch = [1000]

show_50_punch = True
N_50_punch = [50]


livetime = 5 * units.year

# -----------------------------------------
# Existing experimental limits
show_ice_cube_EHE_limit = True
show_ice_cube_HESE = True
show_ice_cube_mu = True
show_anita_I_III_limit = True
show_auger_limit = True

# Neutrino parameter space
show_neutrino_best_fit = True
show_neutrino_best_case = True
show_neutrino_worst_case = True

# ------------------------------------------
# Other planned experiments
show_grand_10k = True
show_grand_200k = True
show_radar = False

#--------------------------------------
show_veff = False

energyBinsPerDecade = 2.
plotUnitsEnergy = units.GeV
plotUnitsFlux = units.cm**-2 * units.second**-1 * units.sr**-1

# Input
# --------------------------------------------------------
# --------------------------------------------------------
# Add here your simulations:
# Form: [Energy, Veff]
# Multiply by appropriate units (see NuRadioRec utilities)
# --------------------------------------------------------


# NuRadioMC Simulations 2018-10-30
# shallow (+ PA@15m)
strawman_veff_pa = np.loadtxt(f"{local_dir}/veffs/strawman_shallow_pa_15m_veff.txt")

strawman_veff_pa[:,0] *= units.eV
strawman_veff_pa[:,1] *= units.km**3 * units.sr

strawman_pa_label = 'Strawman + PA (15m)'
# NuRadioMC Simulations 2018-10-30
# shallow (no PA)

strawman_veff = np.loadtxt(f"{local_dir}/veffs/strawman_shallow_no_pa_veff.txt")
strawman_veff[:,0] *= units.eV
strawman_veff[:,1] *= units.km**3 * units.sr
strawman_label = "shallow (no PA)"

# NuRadioMC Simulations 2018-10-30
# punch deep 90m PA (+ 3x > 3$\\sigma$)

punch_veff = np.loadtxt(f"{local_dir}/veffs/punch_deep_pa_90m_veff.txt")
punch_veff[:,0] *= units.eV
punch_veff[:,1] *= units.km**3 * units.sr
punch_label = 'Punch 90m PA'


# NuRadioMC Simulations 2018-10-30
# deep 50m PA (+ 3x > 3$\\sigma$)

punch_50_veff = np.loadtxt(f"{local_dir}/veffs/punch_deep_pa_50m_veff.txt")
punch_50_veff[:,0] *= units.eV
punch_50_veff[:,1] *= units.km**3 * units.sr
punch_50_label = 'Punch 50m PA'

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

GRAND_200k_data = np.loadtxt(f"{local_dir}/experiments/grand200k_sensitivity.txt")
GRAND_200k_energy = GRAND_200k_data[:,0] 
GRAND_200k_energy *= units.GeV

GRAND_200k = GRAND_200k_data[:,1]
GRAND_200k *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
GRAND_200k /= GRAND_200k_energy

# RADAR proposed from https://arxiv.org/pdf/1710.02883.pdf
Radar = np.loadtxt(f"{local_dir}/experiments/radar_sensitivity.txt") 
Radar[:,0] = 10**Radar[:,0]*units.eV
Radar[:,1] *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
Radar[:,1] /= Radar[:,0]
Radar[:,2] *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
Radar[:,2] /= Radar[:,0]
# --------------------------------------------------------------------
# Published data and limits

# IceCube
# log (E^2 * Phi [GeV cm^02 s^-1 sr^-1]) : log (E [Gev])
# Phys Rev D 98 062003 (2018)
# Numbers private correspondence Shigeru Yoshida
ice_cube_limit = np.loadtxt(f"{local_dir}/experiments/icecube_ehe18_limit.txt")
ice_cube_limit[:,0] = 10**ice_cube_limit[:,0] * units.GeV
ice_cube_limit[:,1] = 10**ice_cube_limit[:,1] * (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
ice_cube_limit[:,1] *= energyBinsPerDecade
ice_cube_limit[:,1] /= ice_cube_limit[:,0]

# Fig. 2 from PoS ICRC2017 (2018) 981
# IceCube preliminary
# E (GeV); E^2 dN/dE (GeV cm^-2 s-1 sr-1); yerror down; yerror up
ice_cube_hese = np.loadtxt(f"{local_dir}/experiments/icecube_hese_icrc17.txt")

ice_cube_hese[:,0] = ice_cube_hese[:,0]* units.GeV
ice_cube_hese[:,1] = ice_cube_hese[:,1] * (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
ice_cube_hese[:,1] *= energyBinsPerDecade * 3 # single flavor
ice_cube_hese[:,1] /= ice_cube_hese[:,0]
ice_cube_hese[:,2] = ice_cube_hese[:,2] * (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
ice_cube_hese[:,2] *= energyBinsPerDecade * 3
ice_cube_hese[:,2] /= ice_cube_hese[:,0]
ice_cube_hese[:,3] = ice_cube_hese[:,3] * (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
ice_cube_hese[:,3] *= energyBinsPerDecade * 3
ice_cube_hese[:,3] /= ice_cube_hese[:,0]

# Ice cube
def ice_cube_nu_fit(energy,slope=-2.13,offset=0.9):
    flux = 3 * offset * (energy/(100*units.TeV))**slope *1e-18 * (units.GeV**-1 * units.cm**-2 * units.second**-1 * units.sr**-1)
    return flux

def ice_cube_mu_range():
    energy = np.arange(1e5,5e6,1e5)*units.GeV
    upper = np.maximum(ice_cube_nu_fit(energy,offset=0.9,slope=-2.),ice_cube_nu_fit(energy,offset=1.2,slope=-2.13))
    upper *= energy
    lower = np.minimum(ice_cube_nu_fit(energy,offset=0.9,slope=-2.26),ice_cube_nu_fit(energy,offset=0.63,slope=-2.13))
    lower *= energy
    return energy, upper, lower

def ice_cube_hese_range():
    energy = np.arange(1e5,5e6,1e5)*units.GeV
    upper = np.maximum(ice_cube_nu_fit(energy,offset=2.46,slope=-2.63),ice_cube_nu_fit(energy,offset=2.76,slope=-2.92))
    upper *= energy
    lower = np.minimum(ice_cube_nu_fit(energy,offset=2.46,slope=-3.25),ice_cube_nu_fit(energy,offset=2.16,slope=-2.92))
    lower *= energy
    return energy, upper, lower


#ANITA I - III
#Phys. Rev. D 98, 022001 (2018)
anita_limit = np.loadtxt(f"{local_dir}/experiments/anita_i_iii_limit.txt")
anita_limit[:,0] *= units.eV
anita_limit[:,1] *= (units.cm**-2 * units.second**-1 * units.sr**-1)
anita_limit[:, 1] *= anita_limit[:, 0] # convert to E^2*dN/dE
anita_limit[:,1] /= 2
anita_limit[:,1] *= energyBinsPerDecade
anita_limit[:,1] /= anita_limit[:,0]


# Auger neutrino limit
auger_limit = np.loadtxt(f"{local_dir}/experiments/auger_9year_limit.txt") 
auger_limit[:,0] = 10** auger_limit[:,0] * units.eV
auger_limit[:,1] *= (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1)
auger_limit[:,1] /= 2 #half-decade binning
auger_limit[:,1] *= energyBinsPerDecade
auger_limit[:,1] /= auger_limit[:,0]

# ===========================================================================
# Plotting

# Veff
# ---------------------------------------------------------------------------
if show_veff:
    plt.figure()
    plt.plot(strawman_veff_pa[:,0],strawman_veff_pa[:,1]/(units.km**3 * units.sr),label=strawman_pa_label)
    plt.plot(strawman_veff[:,0],strawman_veff[:,1]/(units.km**3 * units.sr),label=strawman_label)
    plt.plot(punch_veff[:,0],punch_veff[:,1]/(units.km**3 * units.sr),label=punch_label)
    plt.plot(punch_50_veff[:,0],punch_50_veff[:,1]/(units.km**3 * units.sr),label=punch_50_label)

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Energy [eV]")
    plt.ylabel(r'Effective Volume [km$^3$ sr]')
    plt.legend()
    plt.tight_layout()


# Limit E2 Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1,1,figsize=(7,8))
# fig, ax = plt.subplots(1,1,figsize=(7,4))

# Neutrino Models

Heinze_band = np.loadtxt(f"{local_dir}/models/talys_neu_bands.out")
best_fit, = ax.plot(Heinze_band[:,0], Heinze_band[:,1]*Heinze_band[:,0],c='k',label=r'Best fit UHECR ($\pm$ 3$\sigma$), Heinze et al.',linestyle='-.')

Heinze_evo = np.loadtxt(f"{local_dir}/models/talys_neu_evolutions.out")
ax.fill_between(Heinze_evo[:,0],Heinze_evo[:,5]*Heinze_evo[:,0],Heinze_evo[:,6]*Heinze_evo[:,0],color='0.8')

vanVliet_max_1 = np.loadtxt(f"{local_dir}/models/MaxNeutrinos1.txt")
vanVliet_max_2 = np.loadtxt(f"{local_dir}/models/MaxNeutrinos2.txt")
vanVliet_reas = np.loadtxt(f"{local_dir}/models/ReasonableNeutrinos1.txt")

vanVliet_max = np.maximum(vanVliet_max_1[1,:]/vanVliet_max_1[0,:],vanVliet_max_2[1,:]/vanVliet_max_2[0,:])

prot10, = ax.plot(vanVliet_reas[0,:],vanVliet_reas[1,:]/vanVliet_reas[0,:],label=r'10% protons in UHECRs, van Vliet et al.',linestyle=':',color='darkmagenta')

prot = ax.fill_between(vanVliet_max_1[0,:],vanVliet_max,vanVliet_reas[1,:]/vanVliet_reas[0,:], color='thistle',alpha=0.5,label=r'not excluded from UHECRs')

first_legend = plt.legend(handles=[best_fit,prot,prot10], loc=4)

plt.gca().add_artist(first_legend)
#-----------------------------------------------------------------------

if show_grand_10k:
    ax.plot(GRAND_10k_energy/plotUnitsEnergy,GRAND_10k/plotUnitsFlux,linestyle="--",color='saddlebrown')
    ax.annotate('GRAND 10k',
            xy=(1e10, 3.5e-18), xycoords='data',
            horizontalalignment='left',color='saddlebrown',rotation=-5 )

if show_grand_200k:
    ax.plot(GRAND_200k_energy/plotUnitsEnergy,GRAND_200k/plotUnitsFlux,linestyle="--",color='saddlebrown')
    ax.annotate('GRAND 200k',
            xy=(1e10, 1.8e-19), xycoords='data',
            horizontalalignment='left',color='saddlebrown' ,rotation=-5)
if show_radar:
    ax.fill_between(Radar[:,0]/plotUnitsEnergy,Radar[:,1]/plotUnitsFlux,Radar[:,2]/plotUnitsFlux, facecolor='None',hatch='x',edgecolor='0.8')
    ax.annotate('Radar',
            xy=(3e10, 8e-8), xycoords='data',
            horizontalalignment='left',color='0.7' ,rotation=45)

if show_ice_cube_EHE_limit:
    ax.plot(ice_cube_limit[:,0]/plotUnitsEnergy,ice_cube_limit[:,1]/plotUnitsFlux,color='dodgerblue')
    ax.annotate('IceCube',
            xy=(2e6, 4e-14), xycoords='data',
            horizontalalignment='center',color='dodgerblue',rotation=0 )

if show_ice_cube_HESE:
    # data points
    uplimit = np.copy(ice_cube_hese[:,3])
    uplimit[np.where(ice_cube_hese[:,3]==0)] = 1
    uplimit[np.where(ice_cube_hese[:,3]!=0.)] = 0

    ax.errorbar(ice_cube_hese[:,0]/plotUnitsEnergy,ice_cube_hese[:,1]/plotUnitsFlux, yerr=ice_cube_hese[:,2:].T/plotUnitsFlux,uplims=uplimit,color='dodgerblue',marker='o',ecolor='dodgerblue',linestyle='None')

    # hese fit
    ice_cube_hese_range = ice_cube_hese_range()
    ax.fill_between(ice_cube_hese_range[0]/plotUnitsEnergy, ice_cube_hese_range[1]/plotUnitsFlux,ice_cube_hese_range[2]/plotUnitsFlux,hatch='//',edgecolor='dodgerblue',facecolor='azure')
    plt.plot(ice_cube_hese_range[0]/plotUnitsEnergy, ice_cube_nu_fit(ice_cube_hese_range[0],offset=2.46,slope=-2.92)*ice_cube_hese_range[0]/plotUnitsFlux, color='dodgerblue')

if show_ice_cube_mu:
    # mu fit
    ice_cube_mu_range = ice_cube_mu_range()
    ax.fill_between(ice_cube_mu_range[0]/plotUnitsEnergy, ice_cube_mu_range[1]/plotUnitsFlux,ice_cube_mu_range[2]/plotUnitsFlux,hatch='\\',edgecolor='dodgerblue',facecolor='azure')
    plt.plot(ice_cube_mu_range[0]/plotUnitsEnergy, ice_cube_nu_fit(ice_cube_mu_range[0],offset=0.9,slope=-2.13)*ice_cube_mu_range[0]/plotUnitsFlux,color='dodgerblue')


if show_anita_I_III_limit:
    ax.plot(anita_limit[:,0]/plotUnitsEnergy,anita_limit[:,1]/plotUnitsFlux,color='darkorange')
    ax.annotate('ANITA I - III',
            xy=(3e9, 1e-14), xycoords='data',
            horizontalalignment='left',color='darkorange' )

if show_auger_limit:
    ax.plot(auger_limit[:,0]/plotUnitsEnergy,auger_limit[:,1]/plotUnitsFlux,color='forestgreen')
    ax.annotate('Auger',
            xy=(1.1e8, 2.1e-15), xycoords='data',
            horizontalalignment='left',color='forestgreen',rotation=0 )



# Own limits
limit_labels = []

if show_strawman_pa:
    for N in N_strawman_pa:
        strawman_limit_pa = fluxes.get_limit_e1_flux(energy = strawman_veff_pa[:,0],
                                            veff_sr = strawman_veff_pa[:,1],
                                            livetime = livetime,
                                            signalEff = N,
                                            energyBinsPerDecade=energyBinsPerDecade,
                                            upperLimOnEvents=2.300,
                                            nuCrsScn='hedis_bgr18')

        str_plt_pa, = ax.plot(strawman_veff_pa[:,0]/plotUnitsEnergy,strawman_limit_pa/ plotUnitsFlux,label="{2}: {0} stations, {1} years".format(N,int(livetime/units.year),strawman_pa_label),color='red',linewidth=3)
        limit_labels.append(str_plt_pa)

if show_strawman:
    for N in N_strawman:
        strawman_limit = fluxes.get_limit_e1_flux(energy = strawman_veff[:,0],
                                            veff_sr = strawman_veff[:,1],
                                            livetime = livetime,
                                            signalEff = N,
                                            energyBinsPerDecade=energyBinsPerDecade,
                                            upperLimOnEvents=2.300,
                                            nuCrsScn='hedis_bgr18')

        str_plt, = ax.plot(strawman_veff[:,0]/plotUnitsEnergy,strawman_limit/ plotUnitsFlux,label="{2}: {0} stations, {1} years".format(N,int(livetime/units.year),strawman_label),color='darkmagenta',linewidth=3)
        limit_labels.append(str_plt)


if show_punch:
    for N in N_punch:
        punch_limit = fluxes.get_limit_e1_flux(energy = punch_veff[:,0],
                                            veff_sr = punch_veff[:,1],
                                            livetime = livetime,
                                            signalEff = N,
                                            energyBinsPerDecade=energyBinsPerDecade,
                                            upperLimOnEvents=2.300,
                                            nuCrsScn='hedis_bgr18')

        punch_plt, = ax.plot(punch_veff[:,0]/plotUnitsEnergy,punch_limit/ plotUnitsFlux,label="{2}: {0} stations, {1} years".format(N,int(livetime/units.year),punch_label),color='firebrick',linewidth=3)
        limit_labels.append(punch_plt)


if show_50_punch:
    for N in N_50_punch:
        punch_50_limit = fluxes.get_limit_e1_flux(energy = punch_50_veff[:,0],
                                            veff_sr = punch_50_veff[:,1],
                                            livetime = livetime,
                                            signalEff = N,
                                            energyBinsPerDecade=energyBinsPerDecade,
                                            upperLimOnEvents=2.300,
                                            nuCrsScn='hedis_bgr18')

        punch_50_plt, = ax.plot(punch_50_veff[:,0]/plotUnitsEnergy,punch_50_limit/ plotUnitsFlux,label="{2}: {0} stations, {1} years".format(N,int(livetime/units.year),punch_50_label),color='deeppink',linewidth=3)
        limit_labels.append(punch_50_plt)

plt.legend(handles=limit_labels, loc=2)

ax.set_yscale('log')
ax.set_xscale('log')

ax.set_xlabel(r'Neutrino Energy [GeV]')
ax.set_ylabel(r'$E\Phi$ [cm$^{-2}$ s$^{-1}$ sr$^{-1}$]')


ax.set_ylim(1e-20,2e-12)
ax.set_xlim(1e5,1e11)

plt.tight_layout()
plt.savefig(save_figure_as)
plt.show()
