from NuRadioReco.detector import antennapattern
from NuRadioReco.utilities import units

from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    # name of the antenna model
    # see ./antenna_models_hash.json for available antenna models
    antenna_model = "createLPDA_100MHz_InfAir"

    # ploting range
    min_freq, max_freq = 0 * units.MHz, 1000 * units.MHz
    df = 1 * units.MHz
    ff = np.arange(min_freq, max_freq, df)

    # signal income direction
    zenith, azimuth = np.deg2rad(0), np.deg2rad(0)

    # antenna orientation (0, 0, 90, 0 : Upwards looking, northwards pointing lpda)
    zen_boresight, azi_boresight, zen_ori, azi_ori = np.deg2rad(0), np.deg2rad(0), np.deg2rad(90), np.deg2rad(0)

    provider = antennapattern.AntennaPatternProvider()
    antenna = provider.load_antenna_pattern(antenna_model)

    VEL = antenna.get_antenna_response_vectorized(ff, zenith, azimuth, zen_boresight, azi_boresight, zen_ori, azi_ori)

    fig, ax = plt.subplots(1, 1)
    ax.plot(ff / units.MHz, np.abs(VEL['theta']), '.-', label='eTheta')
    ax.plot(ff / units.MHz, np.abs(VEL['phi']), '.-', label='ePhi')
    ax.set_xlabel("frequency / MHz")
    ax.set_ylabel("vector effective length / m")
    fig.tight_layout()
    ax.legend()
    plt.savefig("VEL_%s_%d_%d.png" % (antenna_model, np.deg2rad(zenith), np.deg2rad(azimuth)))
