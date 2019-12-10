import numpy as np
from matplotlib import pyplot as plt
from NuRadioMC.utilities import fluxes
from NuRadioMC.examples.Sensitivities import E2_fluxes3 as limit
from NuRadioReco.utilities import units
import json
legendfontsize = 11

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


# PyREx 1.5 sigma 100 m (trigger level, km3sr)
# En	Veff		Error
pyrex_100 = np.array((
#[15.5,	9.033E-3,	4.249E-3],
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

pyrex_100[:,0] = 10**pyrex_100[:,0] *units.eV
pyrex_100[:,1] *= units.km**3
pyrex_100[:,2] *= units.km**3

#

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
labels = []
labels = limit.add_limit(ax, labels,
                         pyrex_60[:,0],pyrex_60[:,1]
                         , 61, 'RNO (trigger)', livetime=5*units.year, linestyle='-', color='r',linewidth=3)

labels = limit.add_limit(ax, labels,
                         pyrex_100[:,0],pyrex_100[:,1]
                         , 61, 'RNO 100m (trigger)', livetime=5*units.year, linestyle='--', color='r',linewidth=3)


plt.legend(handles=labels, loc=2,fontsize= legendfontsize)
ax.set_aspect('equal')
fig.tight_layout()

fig.savefig("Diffuse_proposal.pdf")

plt.show()
