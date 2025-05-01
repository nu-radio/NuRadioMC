""" Mapping of PDG particle ids to particle names. """
import numpy as np
import logging
logger = logging.getLogger('NuRadioReco.particle_names')

# list from documentation/source/NuRadioMC/pages/Manuals/event_generation.rst
particle_names = \
    {    0: "Gamma (photon)",
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
        # IDs 81 - 100 are reserved for generator-specific pseudoparticles and concepts
        80: "Particle",  # demanded by proposal
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
        91: "Decay",
       111: "Pion (neutral)",
       211: "Pion (positive)",
      -211: "Pion (negative)",
       311: "Kaon (neutral)",
       321: "Kaon (positive)",
      -321: "Kaon (negative)",
      2212: "Proton",
     -2212: "Antiproton"}

particle_ids = {}
for key, value in particle_names.items():
    particle_ids[value] = key
    
em_primary_names = ['Gamma (photon)', 'Electron', 'Positron', 'Bremsstrahlung photon',
                    'Ionised electron', 'Electron-positron pair', 'Weak interaction', 'Compton']

had_primary_names = ['Hadron blundle', 'Nuclear interaction products', 'Hadronic Decay bundle', "Pion (neutral)", "Pion (positive)",
                     "Pion (negative)", "Kaon (neutral)", "Kaon (positive)", "Kaon (negative)", "Proton", "Antiproton"]

primary_names = em_primary_names + had_primary_names


def particle_name(id):
    if not isinstance(id, (int, np.int_)):
        logger.error("This function only takes integers.")
        raise TypeError("This function only takes integers.")
    
    if not id in particle_names.keys():
        logger.error("Particle id: {:d} unknown".format(id))
        raise ValueError("Particle id: {:d} unknown".format(id))
            
    return particle_names[id]


def particle_id(name):
    if not isinstance(name, str):
        logger.error("This function only takes strings.")
        raise TypeError("This function only takes strings.")

    if not name in particle_ids.keys():
        logger.error("Particle name: {} unknown".format(name))
        print("The following names are implemented: ",
              ", ".join(particle_ids.keys()))
        raise ValueError("Particle name: {} unknown".format(name))

    return particle_ids[name]


if __name__ == "__main__":
    print("The following particles are implemented: ")
    for key, value in particle_names.items():
        print("\t {:5d} : {}".format(key, value))