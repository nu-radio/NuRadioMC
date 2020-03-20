import pyPROPOSAL as pp
import numpy as np
from NuRadioReco.utilities import units
import os

"""
This module takes care of the PROPOSAL implementation. Some important things
should be considered.
Units: PROPOSAL used a fixed system of units that differs from that of NuRadioMC
and conversion between them must be done carefully. The definition of PROPOSAL
units can be found in this file. The most important are the energy unit (MeV)
and the distance unit (cm).
When a muon or a tau is propagated using PROPOSAL and its secondaries are obtained,
most of the secondaries belong to a DynamicData type. The ones we should care
about are the following:
- Brems: a bremsstrahlung photon
- DeltaE: an ionized electron
- EPair: an electron/positron pair
- Hadrons: a set of unspecified hadrons
- NuclInt: the products of a nuclear interaction
The last secondaries obtained via the propagation belong to the Particle DynamicData
type, and represent the products of the decay. They are standard particles with
a PDG code.
"""

# Units definition in PROPOSAL
pp_eV  = 1.e-6
pp_keV = 1.e-3
pp_MeV = 1.e0
pp_GeV = 1.e3
pp_TeV = 1.e6
pp_PeV = 1.e9
pp_EeV = 1.e12
pp_ZeV = 1.e15

pp_m = 1.e2
pp_km = 1.e5

class SecondaryProperties:
    """
    This class stores the properties from secondary particles that are
    relevant for NuRadioMC, namely:
    - distance, the distance to the first interaction vertex
    - energy, the particle energy
    - shower_type, whether the shower they induce is hadronic or electromagnetic
    - name, its name according to the particle_name dictionary on this module

    Distance and energy are expected to be in NuRadioMC units
    """
    def __init__(self,
                 distance,
                 energy,
                 shower_type,
                 code,
                 name):
        self.distance = distance
        self.energy = energy
        self.shower_type = shower_type
        self.code = code
        self.name = name

    def __str__(self):
        s  = "Particle and code: {:} ({:})\n".format(self.name, self.code)
        s += "Energy in PeV: {:}\n".format(self.energy/units.PeV)
        s += "Distance from vertex in km: {:}\n".format(self.distance/units.km)
        s += "Shower type: {:}\n".format(self.shower_type)
        return s

def particle_code (particle):
    """
    If a particle object from PROPOSAL is passed as input, it returns the
    corresponding PDG particle code. DynamicData objects are not considered
    particles internally in PROPOSAL, so they are handled differently.

    Parameters
    ----------
    particle: particle object from PROPOSAL

    Returns
    -------
    integer with the particle code. None if the argument is not a particle
    """
    if particle.particle_def == pp.particle.GammaDef.get()      : return    0
    elif particle.particle_def == pp.particle.EMinusDef.get()   : return   11
    elif particle.particle_def == pp.particle.EPlusDef.get()    : return  -11
    elif particle.particle_def == pp.particle.NuEDef.get()      : return   12
    elif particle.particle_def == pp.particle.NuEBarDef.get()   : return  -12
    elif particle.particle_def == pp.particle.MuMinusDef.get()  : return   13
    elif particle.particle_def == pp.particle.MuPlusDef.get()   : return  -13
    elif particle.particle_def == pp.particle.NuMuDef.get()     : return   14
    elif particle.particle_def == pp.particle.NuMuBarDef.get()  : return  -14
    elif particle.particle_def == pp.particle.TauMinusDef.get() : return   15
    elif particle.particle_def == pp.particle.TauPlusDef.get()  : return  -15
    elif particle.particle_def == pp.particle.NuTauDef.get()    : return   16
    elif particle.particle_def == pp.particle.NuTauBarDef.get() : return  -16
    elif particle.particle_def == pp.particle.Pi0Def.get()      : return  111
    elif particle.particle_def == pp.particle.PiPlusDef.get()   : return  211
    elif particle.particle_def == pp.particle.PiMinusDef.get()  : return -211
    elif particle.particle_def == pp.particle.K0Def.get()       : return  130
    elif particle.particle_def == pp.particle.KPlusDef.get()    : return  310
    elif particle.particle_def == pp.particle.KMinusDef.get()   : return -310
    elif particle.particle_def == pp.particle.PPlusDef.get()    : return 2212
    elif particle.particle_def == pp.particle.PMinusDef.get()   : return-2212
    else: return None

def is_em_primary (particle):
    """
    Given a PROPOSAL particle object as an input, returns True if the particle
    can be an electromagnetic shower primary and False otherwise
    """
    if particle.particle_def == pp.particle.EMinusDef.get(): return True
    elif particle.particle_def == pp.particle.EPlusDef.get(): return True
    elif particle.particle_def == pp.particle.GammaDef.get(): return True
    else:
       return False

def is_had_primary(particle):
    """
    Given a PROPOSAL particle object as an input, returns True if the particle
    can be a hadronic shower primary and Fasle otherwise
    """
    if particle.particle_def == pp.particle.PMinusDef.get(): return True
    elif particle.particle_def == pp.particle.PPlusDef.get(): return True
    elif particle.particle_def == pp.particle.Pi0Def.get(): return True
    elif particle.particle_def == pp.particle.PiMinusDef.get(): return True
    elif particle.particle_def == pp.particle.PiPlusDef.get(): return True
    elif particle.particle_def == pp.particle.K0Def.get(): return True
    elif particle.particle_def == pp.particle.KMinusDef.get(): return True
    elif particle.particle_def == pp.particle.KPlusDef.get(): return True
    else: return False

def is_shower_primary(particle):
    """
    Given a PROPOSAL particle object, returns True if the particle can be
    a shower primary and False otherwise
    """
    return is_em_primary(particle) or is_had_primary(particle)

"""
Codes for the DynamicData class from PROPOSAL. These represent interactions
calculated by PROPOSAL, and although most of them correspond to actual particles -
Brems is a bremsstrahlung photon, DeltaE is an ionised electron, and EPair is an
electron-positron pair, it is useful to treat them as separate entities so that
we know they come from an interaction. PROPOSAL returns particles as decay products
only, in our case.

We have followed the PDG recommendation and used numbers between 80 and 89 for
our own-defined particles.
"""
datatype_code = {
    'Data.Brems'   : 81,
    'Data.DeltaE'  : 82,
    'Data.EPair'   : 83,
    'Data.Hadrons' : 84,
    'Data.NuclInt' : 85
}

em_datatypes = [
    'Data.Brems',
    'Data.DeltaE',
    'Data.EPair'
]

hadrons_datatypes = [
    'Data.Hadrons',
    'Data.NuclInt'
]

datatype_primaries = em_datatypes + hadrons_datatypes

# NuRadioMC internal particle names organised using the PDG codes as keys
particle_name = {
        0 : 'gamma',
       11 : 'e-',
      -11 : 'e+',
       12 : 'nu_e',
      -12 : 'nu_e_bar',
       13 : 'mu-',
      -13 : 'mu+',
       14 : 'nu_mu',
      -14 : 'nu_mu_bar',
       15 : 'tau-',
      -15 : 'tau+',
       16 : 'nu_tau',
      -16 : 'nu_tau_bar',
       81 : 'brems',
       82 : 'ionized_e',
       83 : 'e_pair',
       84 : 'hadrons',
       85 : 'nucl_int',
       86 : 'decay_bundle',
      111 : 'pi0',
      211 : 'pi+',
     -211 : 'pi-',
      130 : 'K0',
      310 : 'K+',
     -310 : 'K-',
     2212 : 'p+',
    -2212 : 'p-'
}

class ProposalFunctions:
    """
    This class serves as a container for PROPOSAL functions. The functions that
    start with double underscore take PROPOSAL units as an argument and should
    not be used from the outside to avoid mismatching units.
    """

    def __create_propagator(self,
                            low=0.1*pp_PeV,
                            particle_code=13,
                            config_file='SouthPole'):
        """
        Creates a PROPOSAL propagator for muons or taus

        Parameters
        ----------
        low: float
            Minimum energy that a particle can have. If this energy is attained,
            propagation stops. In PROPOSAL units (MeV)
        particle_code: integer
            Particle code for the muon- (13), muon+ (-13), tau- (15), or tau+ (-15)
        config_file: string or path
            The user can specify the path to their own config file or choose among
            the three available options:
            -'SouthPole', a config file for the South Pole (spherical Earth). It
            consists of a 2.7 km deep layer of ice, bedrock below and air above.
            -'MooresBay', a config file for Moore's Bay (spherical Earth). It
            consists of a 576 m deep ice layer with a 2234 m deep water layer below,
            and bedrock below that.
            -'InfIce', a config file with a medium of infinite ice
            -'Greenland', a config file for Summit Station, Greenland (spherical Earth),
            same as SouthPole but with a 3 km deep ice layer.
            IMPORTANT: If these options are used, the code is more efficient if the
            user requests their own "path_to_tables" and "path_to_tables_readonly",
            pointing them to a writable directory

        Returns
        -------
        propagator: PROPOSAL propagator
            Propagator that can be used to calculate the interactions of a muon or tau
        """
        mu_def_builder = pp.particle.ParticleDefBuilder()
        if (particle_code == 13):
            mu_def_builder.SetParticleDef(pp.particle.MuMinusDef.get())
        elif (particle_code == -13):
            mu_def_builder.SetParticleDef(pp.particle.MuPlusDef.get())
        elif (particle_code == 15):
            mu_def_builder.SetParticleDef(pp.particle.TauMinusDef.get())
        elif (particle_code == -15):
            mu_def_builder.SetParticleDef(pp.particle.TauPlusDef.get())
        else:
            error_str = "The propagation of this particle via PROPOSAL is not currently supported.\n"
            error_str += "Please choose between -/+muon (13/-13) and -/+tau (15/-15)"
            raise NotImplementedError(error_str)

        mu_def_builder.SetLow(low)
        mu_def = mu_def_builder.build()

        if (config_file == 'SouthPole'):
            config_file_full_path = os.path.join(os.path.dirname(__file__), 'config_PROPOSAL.json')
        elif (config_file == 'MooresBay'):
            config_file_full_path = os.path.join(os.path.dirname(__file__), 'config_PROPOSAL_mooresbay.json')
        elif (config_file == 'InfIce'):
            config_file_full_path = os.path.join(os.path.dirname(__file__), 'config_PROPOSAL_infice.json')
        elif (config_file == 'Greenland'):
            config_file_full_path = os.path.join(os.path.dirname(__file__), 'config_PROPOSAL_greenland.json')
        elif (os.path.exists(config_file)):
            config_file_full_path = config_file
        else:
            raise ValueError("Proposal config file is not valid. Please provide a valid option.")

        if not os.path.exists(config_file_full_path):
            error_message  = "Proposal config file does not exist.\n"
            error_message += "Please provide valid paths for the interpolation tables "
            error_message += "in file {}.sample ".format(config_file_full_path)
            error_message += "and copy the file to {}.".format(os.path.basename(config_file_full_path))
            raise ValueError(error_message)

        propagator = pp.Propagator(particle_def=mu_def, config_file=config_file_full_path)

        return propagator

    def __get_compact_sub_pev_losses(self,
                                     energy_arr,
                                     distance_arr,
                                     compact_dist,
                                     min_energy_loss):
        r""" return biggest compact loss if above min_energy_cut

        This function groups energy losses along a path and groups them into a
        single shower. The effect is only seen for < 10 PeV energy bins and it's
        at least one order of magnitude lower than non-grouped losses, so it does
        not influence that much.

        Parameters
        ----------
        energy_arr: array-like
            energy of the energy losses below min_energy_loss, in PROPOSAL units (MeV)
        distance_arr: array_like
            distances of the energy losses below min_energy_loss, in PROPOSAL units (cm)
        compact_dist: float
            distance in centimeters (PROPOSAL units): how compact the energy losses should be
        min_energy_loss: float
            min energy for the sensitivity (here a PeV), in PROPOSAL units (MeV)
        """
        len_bins = np.arange(distance_arr[0], distance_arr[-1] + 1e-3, 100)
        # We have used 100 to create a bin length of 1 m
        len_indices = np.digitize(distance_arr, len_bins)
        bincount = np.bincount(len_indices, energy_arr)

        if len(bincount) <= compact_dist:
            sum_bins = np.sum(bincount)
            if sum_bins > min_energy_loss:
                return [sum_bins]
            else:
                return []

        # We transform the compact_dist into meters, since the above histogram
        # has a bin length of 1 m
        convolved_comp_arr = np.convolve(np.ones(int(compact_dist/pp_m)), bincount, mode='valid')
        if np.any(convolved_comp_arr > min_energy_loss):
            return [np.max(convolved_10m_arr)]
        else:
            return []

    def __produces_shower(self,
                          particle,
                          min_energy_loss=1*pp_PeV):
        """
        Returns True if the input particle or interaction can be a shower primary
        and its energy is above min_energy_loss

        Parameters
        ----------
        particle: PROPOSAL particle or DynamicData (interaction)
            Input particle
        min_energy_loss: float
            Threshold above which a particle shower is considered detectable
            or relevant, in PROPOSAL units (MeV)

        Returns
        -------
        bool
            True if particle produces shower, False otherwise
        """
        energy_threshold = particle.energy > min_energy_loss
        if not energy_threshold:
            return False

        if particle.id == pp.particle.Data.Particle:
            shower_inducing = is_shower_primary(particle)
        elif str(particle.id) in datatype_primaries:
            shower_inducing = True
        else:
            return False

        return shower_inducing

    def __shower_properties(self, particle):
        """
        Calculates shower properties for the shower created by the input particle

        Parameters
        ----------
        particle: PROPOSAL particle or DynamicData

        Returns
        -------
        shower_type: string
            'em' for EM showers and 'had' for hadronic showers
        code: integer
            Particle code for the shower primary
        name: string
            Name of the shower primary
        """
        if particle.id == pp.particle.Data.Particle:
            if is_em_primary(particle):
                shower_type = 'em'
            elif is_had_primary(particle):
                shower_type = 'had'
            else:
                return None, None, None

            code = particle_code(particle)

        elif str(particle.id) in em_datatypes:
            shower_type = 'em'
            code = datatype_code[str(particle.id)]

        elif str(particle.id) in hadrons_datatypes:
            shower_type = 'had'
            code = datatype_code[str(particle.id)]

        name = particle_name[code]

        return shower_type, code, name

    def __propagate_particle(self,
                             energy_lepton,
                             lepton_code,
                             lepton_position,
                             lepton_direction,
                             propagation_length,
                             propagators):
        """
        Calculates secondary particles using a PROPOSAL propagator. It needs to
        be given a propagators dictionary with particle codes as key

        Parameters
        ----------
        energy_lepton: float
            Energy of the input lepton, in PROPOSAL units (MeV)
        lepton_code: integer
            Input lepton code
        lepton_position: (float, float, float) tuple
            Position of the input lepton, in PROPOSAL units (cm)
        lepton_direction: (float, float, float) tuple
            Lepton direction vector, normalised to 1
        propagation_length: float
            Maximum length the particle is propagated, in PROPOSAL units (cm)

        Returns
        -------
        secondaries: array of PROPOSAL particles
            Secondaries created by the propagation
        """
        x, y, z = lepton_position
        px, py, pz = lepton_direction
        propagators[lepton_code].particle.position = pp.Vector3D(x, y, z)
        propagators[lepton_code].particle.direction = pp.Vector3D(px, py, pz)
        propagators[lepton_code].particle.propagated_distance = 0
        propagators[lepton_code].particle.energy = energy_lepton

        secondaries = propagators[lepton_code].propagate(propagation_length)

        return secondaries

    def __filter_secondaries(self,
                             secondaries,
                             min_energy_loss,
                             lepton_position):
        """
        Takes an input secondary particles array and returns an array with the
        SecondaryProperties of those particles that create a shower above a threshold

        Parameters
        ----------
        secondaries: array of PROPOSAL particles
            Array with secondary particles
        min_energy_loss: float
            Threshold for shower production, in PROPOSAL units (MeV)
        lepton_position: tuple (float, float, float)
            Initial position of the primary lepton, in PROPOSAL units (cm)

        Returns
        -------
        shower_inducing_prods: list
            List containing the secondary properties of every shower-inducing
            secondary particle, in NuRadioMC units
        """

        shower_inducing_prods = []

        for sec in secondaries:

            # Decays contain only one shower-inducing particle
            # Muons and neutrinos resulting from decays are ignored
            if self.__produces_shower(sec, min_energy_loss):

                distance  = ( (sec.position.x - lepton_position[0]) * units.cm )**2
                distance += ( (sec.position.y - lepton_position[1]) * units.cm )**2
                distance += ( (sec.position.z - lepton_position[2]) * units.cm )**2
                distance  = np.sqrt(distance)
                energy = sec.energy * units.MeV

                shower_type, code, name = self.__shower_properties(sec)

                shower_inducing_prods.append( SecondaryProperties(distance, energy, shower_type, code, name) )

        return shower_inducing_prods

    def get_secondaries_array(self,
                              energy_leptons_nu,
                              lepton_codes,
                              lepton_positions_nu=None,
                              lepton_directions=None,
                              config_file='SouthPole',
                              low_nu=0.1*units.PeV,
                              propagation_length_nu=1000*units.km,
                              min_energy_loss_nu=1*units.PeV,
                              propagate_decay_muons=True):
        """
        Propagates a set of leptons and returns a list with the properties for
        all the properties of the shower-inducing secondary particles

        Parameters
        ----------
        energy_leptons_nu: array of floats
            Array with the energies of the input leptons, in NuRadioMC units (eV)
        lepton_codes: array of integers
            Array with the PDG lepton codes
        lepton_positions_nu: array of (float, float, float) tuples
            Array containing the lepton positions in NuRadioMC units (m)
        lepton_directions: array of (float, float, float) tuples
            Array containing the lepton directions, normalised to 1
        config_file: string or path
            The user can specify the path to their own config file or choose among
            the three available options:
            -'SouthPole', a config file for the South Pole (spherical Earth)
            -'MooresBay', a config file for Moore's Bay (spherical Earth)
            -'InfIce', a config file with a medium of infinite ice
            -'Greenland', a config file for Summit Station, Greenland (spherical Earth)
            IMPORTANT: If these options are used, the code is more efficient if the
            user requests their own "path_to_tables" and "path_to_tables_readonly",
            pointing them to a writable directory
        low_nu: float
            Low energy limit for the propagating particle in NuRadioMC units (eV)
        propagation_length_nu: float
            Maximum propagation length in NuRadioMC units (m)
        min_energy_loss_nu: float
            Minimum energy for a selected secondary-induced shower (eV)
        propagate_decay_muons: bool
            If True, muons created by tau decay are propagated and their induced
            showers are stored

        Returns
        -------
        secondaries_array: 2D-list containing SecondaryProperties objects
            List containing the information on the shower-inducing secondaries. The
            first dimension indicates the primary lepton and the second dimension
            navigates through the secondaries produced by that primary.

            The SecondaryProperties objects are expressed in NuRadioMC units.
        """

        # Converting to PROPOSAL units
        low = low_nu * pp_eV
        propagation_length = propagation_length_nu * pp_m
        min_energy_loss = min_energy_loss_nu * pp_eV
        energy_leptons = np.array(energy_leptons_nu) * pp_eV

        if lepton_positions_nu is None:
            lepton_positions = [(0, 0, 0)] * len(energy_leptons)
        else:
            lepton_positions = [ np.array(lepton_position_nu) * pp_m for lepton_position_nu in lepton_positions_nu ]

        if lepton_directions is None:
            lepton_directions = [(0, 0, -1)] * len(energy_leptons)

        propagators = {}
        for lepton_code in np.unique(lepton_codes):
            if lepton_code not in propagators:
                propagators[lepton_code] = self.__create_propagator(low=low, particle_code=lepton_code,
                                                                    config_file=config_file)

        if propagate_decay_muons:
            # We create another muon propagator dictionary to try to avoid a segmentation
            # fault happening in some installations of Proposal
            mu_propagators = {}
            for muon_code in [13, -13]:
                mu_propagators[muon_code] = self.__create_propagator(low=low, particle_code=muon_code,
                                                                     config_file=config_file)
            decay_muons_array = []

        secondaries_array = []

        for energy_lepton, lepton_code, lepton_position, lepton_direction in zip(energy_leptons,
            lepton_codes, lepton_positions, lepton_directions):

            secondaries = self.__propagate_particle(energy_lepton, lepton_code,
                                                    lepton_position, lepton_direction,
                                                    propagation_length, propagators)

            shower_inducing_prods = self.__filter_secondaries(secondaries, min_energy_loss, lepton_position)

            # Checking if there is a muon in the products
            if propagate_decay_muons:

                decay_muons_array.append([None, None, None, None])

                for sec in secondaries:

                    if (sec.id != pp.particle.Data.Particle):
                        continue

                    if (sec.particle_def == pp.particle.MuMinusDef.get()) or (sec.particle_def == pp.particle.MuPlusDef.get()):

                        if sec.particle_def == pp.particle.MuMinusDef.get():
                            mu_code = 13
                        elif sec.particle_def == pp.particle.MuPlusDef.get():
                            mu_code = -13

                        mu_energy = sec.energy
                        if (mu_energy <= low):
                            continue
                        mu_position = (sec.position.x, sec.position.y, sec.position.z)
                        mu_direction = lepton_direction # We reuse the primary lepton direction
                                                        # At these energies the muon direction is the same
                        decay_muons_array[-1] = [mu_energy, mu_code, mu_position, mu_direction]
                        # I store the properties of each muon in an array because they cannot be
                        # propagated while we are looping the secondaries array. Doing that can
                        # create a segmentation fault because the results of the new propagation
                        # are written into the secondaries array (!)

            # group shower-inducing decay products so that they create a single shower
            min_distance = 0.1 * units.m
            while( len(shower_inducing_prods) > 1 and
                   np.abs(shower_inducing_prods[-1].distance - shower_inducing_prods[-2].distance) < min_distance):

                last_decay_prod = shower_inducing_prods.pop(-1)
                shower_inducing_prods[-1].energy += last_decay_prod.energy
                shower_inducing_prods[-1].code = 86
                shower_inducing_prods[-1].name = particle_name[86]

            secondaries_array.append(shower_inducing_prods)

        # Propagating the decay muons
        if propagate_decay_muons:

            for shower_inducing_prods, decay_muon, lepton_position in zip(secondaries_array,
                decay_muons_array, lepton_positions):

                if decay_muon[0] is None:
                    continue
                mu_energy, mu_code, mu_position, mu_direction = decay_muon
                mu_secondaries = self.__propagate_particle(mu_energy, mu_code, mu_position, mu_direction,
                                                           propagation_length, mu_propagators)

                mu_shower_inducing_prods = self.__filter_secondaries(mu_secondaries, min_energy_loss, lepton_position)

                shower_inducing_prods += mu_shower_inducing_prods

        return secondaries_array

    def get_decays(self,
                   energy_leptons_nu,
                   lepton_codes,
                   lepton_positions_nu = None,
                   lepton_directions = None,
                   config_file='InfIce',
                   low_nu=0.1*units.PeV,
                   propagation_length_nu=1000*units.km):
        """
        Propagates a set of leptons and returns a list with the properties of
        the decay particles.

        Parameters
        ----------
        energy_leptons_nu: array of floats
            Array with the energies of the input leptons, in NuRadioMC units (eV)
        lepton_codes: array of integers
            Array with the PDG lepton codes
        lepton_positions_nu: array of (float, float, float) tuples
            Array containing the lepton positions in NuRadioMC units (m)
        lepton_directions: array of (float, float, float) tuples
            Array containing the lepton directions, normalised to 1
        config_file: string or path
            The user can specify the path to their own config file or choose among
            the three available options:
            -'SouthPole', a config file for the South Pole (spherical Earth)
            -'MooresBay', a config file for Moore's Bay (spherical Earth)
            -'InfIce', a config file with a medium of infinite ice
            -'Greenland', a config file for Summit Station, Greenland (spherical Earth)
            IMPORTANT: If these options are used, the code is more efficient if the
            user requests their own "path_to_tables" and "path_to_tables_readonly",
            pointing them to a writable directory
        low_nu: float
            Low energy limit for the propagating particle in NuRadioMC units (eV)
        propagation_length_nu: float
            Maximum propagation length in NuRadioMC units (m)

        Returns
        -------
        decays_array: array of (float, float) tuples
            The first element of the tuple contains the decay distance in m
            The second element contains the decay energy in eV (NuRadioMC units)
        """

        # Converting to PROPOSAL units
        low = low_nu * pp_eV
        propagation_length = propagation_length_nu * pp_m
        energy_leptons = np.array(energy_leptons_nu) * pp_eV

        if lepton_positions_nu is None:
            lepton_positions = [(0, 0, 0)] * len(energy_leptons)
        else:
            lepton_positions = [ np.array(lepton_position_nu) * pp_m for lepton_position_nu in lepton_positions_nu ]

        if lepton_directions is None:
            lepton_directions = [(0, 0, 1)] * len(energy_leptons)

        propagators = {}
        for lepton_code in lepton_codes:
            if lepton_code not in propagators:
                propagators[lepton_code] = self.__create_propagator(low=low, particle_code=lepton_code,
                                                                    config_file=config_file)

        decays_array = []

        for energy_lepton, lepton_code, lepton_position, lepton_direction in zip(energy_leptons,
            lepton_codes, lepton_positions, lepton_directions):

            decay_prop = (None, None)

            while( decay_prop == (None,None) ):

                secondaries = self.__propagate_particle(energy_lepton, lepton_code, lepton_position, lepton_direction,
                                                        propagation_length, propagators)

                decay_particles = np.array([p for p in secondaries if p.id == pp.particle.Data.Particle])
                decay_energies = np.array([p.energy for p in decay_particles])
                decay_energy = np.sum(decay_energies) * units.MeV

                try:
                    # If Proposal fails and there is no decay (sometimes it happens),
                    # the particle is propagated again
                    decay_distance  = ( (decay_particles[0].position.x - lepton_position[0]) * units.cm )**2
                    decay_distance += ( (decay_particles[0].position.y - lepton_position[1]) * units.cm )**2
                    decay_distance += ( (decay_particles[0].position.z - lepton_position[2]) * units.cm )**2
                    decay_distance  = np.sqrt(decay_distance)

                    decay_prop = (decay_distance, decay_energy)
                    decays_array.append(decay_prop)
                except:
                    decay_prop = (None, None)

        return np.array(decays_array)
