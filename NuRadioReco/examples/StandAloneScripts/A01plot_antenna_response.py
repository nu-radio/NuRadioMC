import NuRadioReco.detector.antennapattern
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
import logging
import numpy as np

from NuRadioReco.modules.base import module
logger = module.setup_logger(level=logging.DEBUG)

provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()

theta = 90 * units.deg
ff = np.linspace(50 * units.MHz, 1 * units.GHz, 1000)
bicone = provider.load_antenna_pattern("bicone_v8_inf_n1.78")
bicone_n14 = provider.load_antenna_pattern("bicone_v8_inf_n1.4")
bicone_air = provider.load_antenna_pattern("bicone_v8_InfAir")

bicone_XFDTD = provider.load_antenna_pattern("XFDTD_Vpol_CrossFeed_150mmHole_n1.78")

VELs = bicone.get_antenna_response_vectorized(ff, 90 * units.deg, np.deg2rad(0),
                                              np.deg2rad(180), 0, np.deg2rad(90), np.deg2rad(0))
fig, (ax) = plt.subplots(1, 1, sharey=True)
ax.plot(ff / units.MHz, np.abs(VELs['theta']), label='eTheta bicone n=1.78 (WIPL-D)')
# ax.plot(ff / units.MHz, np.abs(VELs['phi']), label='ePhi bicone (WIPL-D)')

VELs = bicone_XFDTD.get_antenna_response_vectorized(
    ff,
    90 * units.deg,
    np.deg2rad(0),
    np.deg2rad(180),
    0,
    np.deg2rad(90),
    np.deg2rad(0)
)
ax.plot(ff / units.MHz, np.abs(VELs['theta']), '--', label='eTheta bicone n=1.78 (XFDTD)')
# ax.plot(ff / units.MHz, np.abs(VELs['phi']), '--', label='ePhi bicone (old ARA)')

VELs = bicone_air.get_antenna_response_vectorized(ff, 90 * units.deg, np.deg2rad(0),
                                                  np.deg2rad(180), 0, np.deg2rad(90), np.deg2rad(0))
ax.plot(ff / units.MHz, np.abs(VELs['theta']), '--', label='eTheta bicone (air)')
ax.plot(ff / units.MHz, np.abs(VELs['phi']), '--', label='ePhi bicone (air)')

ax.set_title('NS ant down, signal from East at 90deg')
ax.legend()
ax.set_ylabel("Heff [m]")
ax.set_xlabel("frequency [MHz]")


plt.show()
