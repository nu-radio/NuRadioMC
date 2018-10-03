from NuRadioReco.detector.antennapattern import *
import matplotlib.pyplot as plt
import logging
import copy

logging.basicConfig(level=logging.DEBUG)

provider = AntennaPatternProvider()

theta = 90 * units.deg
ff = np.linspace(50 * units.MHz, 1 * units.GHz, 1000)
bicone = provider.load_antenna_pattern("bicone_v8_InfFirn")
bicone_air = provider.load_antenna_pattern("bicone_v8_InfAir")
bicone_ARA = provider.load_antenna_pattern("ARA_bicone_InfFirn")

VELs = bicone.get_antenna_response_vectorized(ff, 90 * units.deg, np.deg2rad(0),
                                              np.deg2rad(180), 0, np.deg2rad(90), np.deg2rad(0))
fig, (ax) = plt.subplots(1, 1, sharey=True)
ax.plot(ff /units.MHz, np.abs(VELs['theta']), label='eTheta bicone (WIPL-D)')
ax.plot(ff / units.MHz, np.abs(VELs['phi']), label='ePhi bicone (WIPL-D)')

VELs = bicone_ARA.get_antenna_response_vectorized(ff, 90 * units.deg, np.deg2rad(0),
                                                  np.deg2rad(180), 0, np.deg2rad(90), np.deg2rad(0))
ax.plot(ff / units.MHz, np.abs(VELs['theta']), '--', label='eTheta bicone (old ARA)')
ax.plot(ff / units.MHz, np.abs(VELs['phi']), '--', label='ePhi bicone (old ARA)')

VELs = bicone_air.get_antenna_response_vectorized(ff, 90 * units.deg, np.deg2rad(0),
                                                  np.deg2rad(180), 0, np.deg2rad(90), np.deg2rad(0))
ax.plot(ff / units.MHz, np.abs(VELs['theta']), '--', label='eTheta bicone (air)')
ax.plot(ff / units.MHz, np.abs(VELs['phi']), '--', label='ePhi bicone (air)')

ax.set_title('NS ant down, signal from East at 90deg')
ax.legend()


plt.show()
