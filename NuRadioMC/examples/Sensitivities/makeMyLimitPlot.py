import numpy as np
from matplotlib import pyplot as plt
from NuRadioMC.utilities import fluxes
from NuRadioMC.examples.Sensitivities import E2_fluxes2 as limit
from NuRadioReco.utilities import units
import json


# import latest Veff from NuRadioMC
with open("NuRadioMC_20181113_Veff.json", 'r') as fin:
    data = json.load(fin)

    # producen nice outreach plot
    fig, ax = limit.get_E2_limit_figure()
    labels = []
    labels = limit.add_limit(ax, labels,
                             data['shallow + PA@15m@2s']['energies'],
                             data['shallow + PA@15m@2s']['Veff'], 200, 'GIANT', fmt='-C1')
    labels = limit.add_limit(ax, labels,
                             data['shallow + PA@15m@2s']['energies'],
                             data['shallow + PA@15m@2s']['Veff'], 2000, 'GIANT', fmt='-C3')
    plt.legend(handles=labels, loc=2)
    ax.set_title("raw triggered volumes")
    fig.tight_layout()
    fig.savefig("limits_outreach.pdf")


    fig, ax = limit.get_E2_limit_figure()
    labels = []
    labels = limit.add_limit(ax, labels,
                             data['shallow + PA@15m@2s']['energies'],
                             data['shallow + PA@15m@2s']['Veff'], 270, 'shallow + PA@15m', fmt='-C0')
    labels = limit.add_limit(ax, labels, data['shallow + PA@50m@2s']['energies'],
                             data['shallow + PA@50m@2s']['Veff'], 130, 'shallow + deep 60m', fmt='--C0')
    labels = limit.add_limit(ax, labels, data['shallow + PA@15m@2s']['energies'],
                             data['shallow + PA@15m@2s']['Veff'], 110, 'shallow + PA@15m', fmt='-C1')
    labels = limit.add_limit(ax, labels, data['shallow + PA@50m@2s']['energies'],
                             data['shallow + PA@50m@2s']['Veff'], 55, 'shallow + deep 60m', fmt='--C1')
    plt.legend(handles=labels, loc=2)
    ax.set_title("raw triggered volumes")
    fig.tight_layout()
    fig.savefig("limits_shallow_deep60m_raw.pdf")

    fig, ax = limit.get_E2_limit_figure()
    labels = []
    labels = limit.add_limit(ax, labels, data['shallow + PA@15m@2s + 3x3s']['energies'],
                             data['shallow + PA@15m@2s + 3x3s']['Veff'], 270,
                             'shallow + PA@15m', fmt='-C0')
    labels = limit.add_limit(ax, labels, data['shallow + PA@50m@2s + 3x3s']['energies'],
                             data['shallow + PA@50m@2s + 3x3s']['Veff'], 130, 'shallow + deep 60m', fmt='--C0')
    labels = limit.add_limit(ax, labels, data['shallow + PA@15m@2s + 3x3s']['energies'],
                             data['shallow + PA@15m@2s + 3x3s']['Veff'], 110, 'shallow + PA@15m', fmt='-C1')
    labels = limit.add_limit(ax, labels, data['shallow + PA@50m@2s + 3x3s']['energies'],
                             data['shallow + PA@50m@2s + 3x3s']['Veff'], 55, 'shallow + deep 60m', fmt='--C1')
    plt.legend(handles=labels, loc=2)
    ax.set_title("with 3x 3sigma cut")
    fig.tight_layout()
    fig.savefig("limits_shallow_deep60m_reco.pdf")

    ## calculate number of neutrinos from icecube (nu mu) flux
    def print_N_neutrinos(trigger_name, n_stations):
        print("{} for 3 years and {} stations with 100% uptime".format(trigger_name, n_stations))
        E = np.array(data[trigger_name]['energies'])
        Nnu = fluxes.get_number_of_events_for_flux(E, limit.ice_cube_nu_fit(E),
                                                   data[trigger_name]['Veff'], 3 * units.year * n_stations)
        print("{:>10}: {}".format("energy [eV]", "N_nu"))
        for iE in range(len(E)):
            print("{:>10.1g}: {:.1f}".format(E[iE], Nnu[iE]))
        print("--------------------")
        print("{:>10}: {:.1f}".format("total", np.sum(Nnu)))

    print("number of neutrinos for icecube (numu) flux")
    print_N_neutrinos('shallow + PA@15m@2s', 270)
    print_N_neutrinos('shallow + PA@15m@2s', 110)
    print_N_neutrinos('shallow + PA@50m@2s', 130)
    print_N_neutrinos('shallow + PA@50m@2s', 55)

    plt.show()
