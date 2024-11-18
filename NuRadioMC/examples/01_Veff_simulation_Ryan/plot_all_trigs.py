import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import h5py
import glob
from scipy import interpolate
import json
import os
import sys

# Setup logging
#from NuRadioReco.utilities.logging import setup_logger
#logger = setup_logger(name="")

from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
from NuRadioMC.utilities import fluxes
from NuRadioMC.utilities.Veff import get_Veff_Aeff, get_Veff_Aeff_array, get_index, get_Veff_water_equivalent
from NuRadioMC.examples.Sensitivities import E2_fluxes3 as limits
plt.switch_backend('agg')

if __name__ == "__main__":

    path = '.'
    if(len(sys.argv) == 1):
        print("no path specified, assuming that hdf5 files are in directory 'output'")
    else:
        path = sys.argv[1]
    fig_res, (ax_main,ax_res)=plt.subplots(2,1,figsize=(6,6),sharex=True,gridspec_kw={'height_ratios':[3,1]})
    
    trig_names={'triggers':['high_low','PA_fir']}
    data = get_Veff_Aeff(path,station=11)
    Veffs, energies, energies_low, energies_up, zenith_bins, utrigger_names = get_Veff_Aeff_array(data)
    V_comp=np.average(Veffs[:,:,get_index('high_low',utrigger_names),0],axis=1)
    comp=get_Veff_water_equivalent(V_comp)*4*np.pi
    for i in trig_names['triggers']:
        #print(i)
        #print(energies)
        #print(Veffs[:,:,get_index(i,utrigger_names),0]/units.km**3/units.sr)
        Veff=np.average(Veffs[:,:,get_index(i,utrigger_names),0],axis=1)
        #print('b4',Veff/units.km**3/units.sr)
        Veff=get_Veff_water_equivalent(Veff)*4*np.pi
        #print('after',Veff/units.km**3/units.sr/4/np.pi)
        Veff_error=Veff/np.sum(Veffs[:,:,get_index(i,utrigger_names),2],axis=1)**0.5
        label=i
        ax_main.errorbar(energies/units.eV,Veff/units.km**3/units.sr,yerr=Veff_error/units.km**3/units.sr,fmt='d-',label=label)
        ax_res.plot(energies/units.eV,Veff/comp,label=label)
        print(i)
        print(energies,Veff/units.km**3/units.sr)
    
    ax_main.semilogx(True)
    ax_main.semilogy(True)
    ax_res.semilogx(True)
    #ax_res.semilogy(True)
    ax_res.set_xlabel('Neutrino Energy [eV]')
    ax_res.set_ylabel('Ratio to H-L')
    ax_main.set_ylabel('Effective Volume [km$^3$ sr]')
    ax_main.set_ylim(top=1e3,bottom=1e-2)
    ax_main.legend()
    fig_res.tight_layout()
    fig_res.savefig('lin_ratio_to_highlow.png')
    exit()

    # plot expected limit
    fig, ax = limits.get_E2_limit_figure(diffuse=True, show_grand_10k=True, show_grand_200k=False)
    labels = []
    labels = limits.add_limit(ax, labels, energies, Veff,
                              100, 'NuRadioMC example', livetime=3 * units.year, linestyle='-', color='blue', linewidth=3)
    leg = plt.legend(handles=labels, loc=2)
    fig.savefig("limits.pdf")
    #plt.show()
