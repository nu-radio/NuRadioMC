""" Mapping of PDG particle ids to particle names. """
import numpy as np
import logging
logger = logging.getLogger('NuRadioReco.particle_names')

# list from NuRadioMC/pages/Manuals/event_generation.rst
particle_names = \
    {0: "Gamma (photon)",
     11: "Electron",
     -11: "Positron",
     12: "Electron neutrino",
     -12: "Electron antineutrino",
     13: "Muon (negative)",
     -13: "Antimuon (positive muon)",
     14: "Muon neutrino",
     -14: "Muon antineutrino",
     15: "Tau (negative)",
     -15: "Antitau (or positive tau)",
     16: "Tau neutrino",
     -16: "Tau antineutrino",
     81: "Bremsstrahlung photon",
     82: "Ionised electron",
     83: "Electron-positron pair",
     84: "Hadron blundle",
     85: "Nuclear interaction products",
     86: "Hadronic Decay bundle",
     87: "Muon pair",
     88: "Continuous loss",
     89: "Weak interaction",
     90: "Compton",
     111: "Pion (neutral)",
     211: "Pion (positive)",
     -211: "Pion (negative)",
     311: "Kaon (neutral)",
     321: "Kaon (positive)",
     -321: "Kaon (negative)",
     2212: "Proton",
     -2212: "Antiproton"}

def particle_name(id):
    if not isinstance(id, (int, np.int_)):
        logger.error("This function only takes integers.")
        raise TypeError("This function only takes integers.")
    
    if not id in particle_names.keys():
        logger.error("Particle id: {:d} unknown".format(id))
        raise ValueError("Particle id: {:d} unknown".format(id))
            
    return particle_names[id]
