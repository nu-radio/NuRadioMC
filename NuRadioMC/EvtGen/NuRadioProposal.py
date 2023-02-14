import proposal as pp
import numpy as np
from NuRadioReco.utilities import units, particle_names
import NuRadioReco.utilities.metaclasses
import os
import six
import json
import logging
from glob import glob

"""
This module takes care of the PROPOSAL implementation. Some important things
should be considered.
Units: PROPOSAL used a fixed system of units that differs from that of NuRadioMC
and conversion between them must be done carefully. The definition of PROPOSAL
units can be found in this file. The most important are the energy unit (MeV)
and the distance unit (cm).
When a muon or a tau is propagated using PROPOSAL and its secondaries are obtained,
most of the secondaries have an InteractionType associated. The more important for
us are the following:
- Brems: a bremsstrahlung photon
- DeltaE: an ionized electron
- EPair: an electron/positron pair
- Hadrons: a set of unspecified hadrons
- NuclInt: the products of a nuclear interaction
- MuPair: a muon/antimuon pair
- WeakInt: a weak interaction
- Compton: Compton effect
"""

# Units definition in PROPOSAL
pp_eV = 1.e-6
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
    - distance      : Distance to the first interaction vertex
    - energy        : Particle energy
    - shower_type   : Whether the shower they induce is hadronic or electromagnetic
    - name          : Name according to NuRadioReco.utilities.particle_names
    - parent_energy : Energy of the parent particle

    Distance and energy are in NuRadioMC units
    """

    def __init__(self,
                 distance,
                 energy,
                 shower_type,
                 code,
                 name,
                 parent_energy):
        self.distance = distance
        self.energy = energy
        self.shower_type = shower_type
        self.code = code
        self.name = name
        self.parent_energy = parent_energy

    def __str__(self):
        s =  "Particle and code    : {:} ({:})\n".format(self.name, self.code)
        s += "Energy               : {:} PeV\n".format(self.energy / units.PeV)
        s += "Distance from vertex : {:} km\n".format(self.distance / units.km)
        s += "Shower type          : {:}\n".format(self.shower_type)
        s += "Parent energy        : {:} PeV".format(self.parent_energy / units.PeV)
        return s

"""
Codes for the InteractionType class from PROPOSAL. These represent interactions
calculated by PROPOSAL, and although most of them correspond to actual particles -
Brems is a bremsstrahlung photon, DeltaE is an ionised electron, and EPair is an
electron-positron pair, it is useful to treat them as separate entities so that
we know they come from an interaction.

We have followed the PDG recommendation and used numbers between 80 and 89 for
our own-defined particles, although we have needed also 90 and 91 (they are
very rarely used).
"""

# For dicussion why we cast to int instead of using
# e.g. pp.particle.Interaction_Type.particle.value, see https://github.com/nu-radio/NuRadioMC/pull/458
proposal_interaction_names = { int(pp.particle.Interaction_Type.particle): 'particle',
                               int(pp.particle.Interaction_Type.brems): 'brems',
                               int(pp.particle.Interaction_Type.ioniz): 'ionized_e',
                               int(pp.particle.Interaction_Type.epair): 'e_pair',
                               int(pp.particle.Interaction_Type.photonuclear): 'nucl_int',
                               int(pp.particle.Interaction_Type.mupair): 'mu_pair',
                               int(pp.particle.Interaction_Type.hadrons): 'hadrons',
                               int(pp.particle.Interaction_Type.continuousenergyloss): 'cont_loss',
                               int(pp.particle.Interaction_Type.weakint): 'weak_int',
                               int(pp.particle.Interaction_Type.compton): 'compton',
                               int(pp.particle.Interaction_Type.decay): 'decay' }

proposal_interaction_codes = { int(pp.particle.Interaction_Type.particle): 80,
                               int(pp.particle.Interaction_Type.brems): 81,
                               int(pp.particle.Interaction_Type.ioniz): 82,
                               int(pp.particle.Interaction_Type.epair): 83,
                               int(pp.particle.Interaction_Type.photonuclear): 85,
                               int(pp.particle.Interaction_Type.mupair): 87,
                               int(pp.particle.Interaction_Type.hadrons): 84,
                               int(pp.particle.Interaction_Type.continuousenergyloss): 88,
                               int(pp.particle.Interaction_Type.weakint): 89,
                               int(pp.particle.Interaction_Type.compton): 90,
                               int(pp.particle.Interaction_Type.decay): 91 }


def particle_code(pp_type):
    """
    For an integer, corresponding to a proposal.particle.Interaction_Type or 
    proposal.particle.Particle_Type, it returns the corresponding PDG 
    particle code.

    Parameters
    ----------
    pp_type: int, corresponding to a proposal.particle.Interaction_Type or 
             proposal.particle.Particle_Type

    Returns
    -------
    integer with the PDG particle code. None if the argument is not a particle
    """
    if pp_type in proposal_interaction_codes:
        return proposal_interaction_codes[pp_type]
    elif pp_type in particle_names.particle_names:
        return pp_type
    else:
        print(pp_type)
        return None


def is_em_primary (pp_type):
    """
    For an integer corresponding to a particle type or interaction type of 
    proposal, returns True if the object can be an electromagnetic shower primary 
    and False otherwise

    Parameters
    ----------
    pp_type: int, corresponding to a pp.proposal.Interaction_Type or pp.proposal.Particle_Type

    Returns
    -------
    bool, True if the particle can be an electromagnetic shower primary and False otherwise
    """
    code = particle_code(pp_type)
    name = particle_names.particle_name(code)
    
    if name in particle_names.em_primary_names:
        return True
    else:
        return False


def is_had_primary(pp_type):
    """
    Given an integer corresponding to a particle type or interaction type of 
    proposal, returns True if the object can be a hadronic shower primary 
    and False otherwise

    Parameters
    ----------
    pp_type: int, corresponding to a pp.proposal.Interaction_Type or pp.proposal.Particle_Type

    Returns
    -------
    bool, True if the particle can be a hadronic shower primary and False otherwise
    """
    code = particle_code(pp_type)
    name = particle_names.particle_name(code)

    if name in particle_names.had_primary_names:
        return True
    else:
        return False


def is_shower_primary(pp_type):
    """
    Given an integer corresponding to a particle type or interaction type of 
    proposal, returns True if the object can be a shower primary and False otherwise

    Parameters
    ----------
    pp_type: int, corresponding to a pp.proposal.Interaction_Type or pp.proposal.Particle_Type

    Returns
    -------
    bool, True if the particle can be a shower primary and False otherwise
    """
    code = particle_code(pp_type)
    name = particle_names.particle_name(code)
    
    if name in particle_names.primary_names:
        return True
    else:
        return False


@six.add_metaclass(NuRadioReco.utilities.metaclasses.Singleton)
class ProposalFunctions(object):
    """
    This class serves as a container for PROPOSAL functions. The functions that
    start with double underscore take PROPOSAL units as an argument and should
    not be used from the outside to avoid mismatching units.
    """

    def __init__(self, config_file='SouthPole', log_level=logging.INFO, tables_path=None, seed=12, upper_energy_limit=1e14*units.MeV):
        """
        Parameters
        ----------
        config_file: string or path
            The user can specify the path to their own config file or choose among
            the available options:
            -'SouthPole', a config file for the South Pole (spherical Earth). It
            consists of a 2.7 km deep layer of ice, bedrock below and air above.
            -'MooresBay', a config file for Moore's Bay (spherical Earth). It
            consists of a 576 m deep ice layer with a 2234 m deep water layer below,
            and bedrock below that.
            -'InfIce', a config file with a medium of infinite ice
            -'Greenland', a config file for Summit Station, Greenland (spherical Earth),
            same as SouthPole but with a 3 km deep ice layer.
        log_level: logging level
        tables_path: path
            Path to PROPOSAL tables. Should be set to a path where PROPOSAL tables exist, or
            where PROPOSAL tables can be saved. This avoids that PROPOSAL has to regenerate
            tables (which can take several minutes) every time the script is executed.
            Default: None -> create directory "proposal_tables" at the location of this file.
        seed: int
            Seed to be used by PROPOSAL
        upper_energy_limit: float
            upper_energy_limit of tables that will be created by PROPOSAL, in NuRadioMC units (eV).
            There will be an error if primaries with energies above this energy will be injected.
            Note that PROPOSAL will have to regenerate tables for a new values of upper_energy_limit
        create_new: bool (default:False)
            Can be used to force the creation of a new ProposalFunctions object.
            By default, the __init__ will only create a new object if none already exists.
            For more details, check the documentation for the
            :class:`Singleton metaclass <NuRadioReco.utilities.metaclasses.Singleton>`.
        """
        self.__logger = logging.getLogger("proposal")
        self.__logger.setLevel(log_level)
        self.__logger.info("initializing proposal interface class")

        pp.RandomGenerator.get().set_seed(seed) # set global seed for PROPOSAL

        default_configs = {
            'SouthPole':  'config_PROPOSAL.json', 'MooresBay': 'config_PROPOSAL_mooresbay.json',
            'InfIce': 'config_PROPOSAL_infice.json', 'Greenland': 'config_PROPOSAL_greenland.json'}

        if tables_path is None:
            tables_path = os.path.join(os.path.dirname(__file__), "proposal_tables")
        
        if config_file in default_configs:
            # default configurations should append their identifier to the tables path
            if not config_file == os.path.dirname(tables_path).split("/")[-1]:
                    tables_path = os.path.join(tables_path, config_file)

        if not os.path.exists(tables_path):
            self.__logger.info(f"Create directory {tables_path} to store proposal tables")
            os.makedirs(tables_path)

        if config_file in default_configs:
            config_file_full_path = os.path.join(os.path.dirname(__file__), default_configs[config_file])

        elif os.path.exists(config_file):
            config_file_full_path = config_file
        else:
            raise ValueError("Proposal config file is not valid. Please provide a valid option.")

        if not os.path.exists(config_file_full_path):
            raise ValueError("Unable to find proposal config file.\n"
                "Make sure that json configuration file under "
                "path {} exists.".format(config_file_full_path))


        self.__propagators = {}
        self.__config_file = config_file
        self.__config_file_full_path = config_file_full_path
        self.__tables_path = tables_path
        self.__upper_energy_limit = upper_energy_limit * pp_eV # convert to PROPOSAL units

        self.__download_tables = False
        if self.__config_file in default_configs:
            if len(glob(self.__tables_path + "/*.dat")) == 0:
                self.__download_tables = True

    @staticmethod
    def __calculate_distance(pp_position, pos_arr):
        """ 
        Calculates distance between secondary and lepton position (both in proposal units).
        
        Paramters
        ---------
        
        pp_position: ParticleState.position
            Position of a secondary particle (in proposal units)
            
        pos_arr: np.array(3,)
            Init. position of the lepton (in proposal units)
        
        Returns
        -------
        
        Distance between both coordinates in NuRadioMC units
        """
        return np.linalg.norm(pp_position.cartesian_coordinates - pos_arr) * units.cm


    def __get_propagator(self, particle_code=13):
        """
        Returns a PROPOSAL propagator for muons or taus. If it does not exist yet it is being generated.

        Parameters
        ----------
        particle_code: integer
            Particle code for the muon- (13), muon+ (-13), tau- (15), or tau+ (-15)

        Returns
        -------
        propagator: PROPOSAL propagator
            Propagator that can be used to calculate the interactions of a muon or tau
        """
        if particle_code not in self.__propagators:
            self.__logger.info(f"initializing propagator for particle code {particle_code}")

            pp.InterpolationSettings.tables_path = self.__tables_path
            # download pre-calculated tables for default configs, but not more than once
            if self.__download_tables:
                from NuRadioMC.EvtGen.proposal_table_manager import download_proposal_tables

                try:
                    download_proposal_tables(self.__config_file, tables_path=self.__tables_path)
                except:
                    self.__logger.warning("requested pre-calculated proposal tables could not be downloaded, calculating manually")
                    pass
                self.__download_tables = False

            # upper energy lim for proposal tables, in PROPOSAL units (MeV)
            pp.InterpolationSettings.upper_energy_lim = self.__upper_energy_limit

            try:
                p_def = pp.particle.get_ParticleDef_for_type(particle_code)
            except:
                error_str = "The propagation of this particle via PROPOSAL is not currently supported.\n" + \
                    "Please choose between -/+muon (13/-13) and -/+tau (15/-15)"
                raise NotImplementedError(error_str)

            self.__propagators[particle_code] = pp.Propagator(particle_def=p_def, path_to_config_file=self.__config_file_full_path)

        return self.__propagators[particle_code]

    def __produces_shower(self,
                          pp_type,
                          energy,
                          min_energy_loss=1 * pp_PeV):
        """
        Returns True if the input particle or interaction can be a shower primary
        and its energy is above min_energy_loss

        Parameters
        ----------
        pp_type: int
            int corresponding to a pp.proposal.Interaction_Type or pp.proposal.Particle_Type
        energy: float
            Energy of particle or interaction, in PROPOSAL units (MeV)
        min_energy_loss: float
            Threshold above which a particle shower is considered detectable
            or relevant, in PROPOSAL units (MeV)

        Returns
        -------
        bool
            True if particle produces shower, False otherwise
        """
        if (energy < min_energy_loss):
            return False

        return is_shower_primary(pp_type)


    def __shower_properties(self, pp_type):
        """
        Calculates shower properties for the shower created by the input particle

        Parameters
        ----------
        pp_type: int
            int corresponding to a pp.proposal.Interaction_Type or pp.proposal.Particle_Type

        Returns
        -------
        shower_type: string
            'em' for EM showers and 'had' for hadronic showers
        code: integer
            Particle code for the shower primary
        name: string
            Name of the shower primary
        """
        if not is_shower_primary(pp_type):
            return None, None, None

        code = particle_code(pp_type)

        if is_em_primary(pp_type):
            shower_type = 'em'

        if is_had_primary(pp_type):
            shower_type = 'had'

        name = particle_names.particle_name(code)

        return shower_type, code, name

    def __propagate_particle(self,
                             energy_lepton,
                             lepton_code,
                             lepton_position,
                             lepton_direction,
                             propagation_length,
                             low=1 * pp_PeV):
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
        low: float
            Low energy limit for the propagating particle in Proposal units (MeV)

        Returns
        -------
        secondaries: proposal.particle.Secondaries
            object containing all information about secondary particles produced 
            during propagation
        """
        x, y, z = lepton_position
        px, py, pz = lepton_direction

        if (energy_lepton > self.__upper_energy_limit):
            raise ValueError("Initial lepton energy higher than upper_energy_limit of PROPOSAL. Adjust upper_energy_limit when"
                             " initialzing EvtGen.NuRadioProposal.ProposalFunctions.")

        initial_condition = pp.particle.ParticleState()
        initial_condition.type = lepton_code
        initial_condition.position = pp.Cartesian3D(x, y, z)
        initial_condition.direction = pp.Cartesian3D(px, py, pz)
        initial_condition.energy = energy_lepton
        initial_condition.propagated_distance = 0

        initial_condition.direction.normalize() # ensure that direction is normalized

        secondaries = self.__get_propagator(lepton_code).propagate(initial_condition,
                                                                  max_distance = propagation_length,
                                                                  min_energy=low)
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
        secondaries: array of proposal.particle.StochasticLoss objects
            Array of all stochastic losses produced by PROPOSAL 
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

            if self.__produces_shower(sec.type, sec.energy, min_energy_loss):

                distance = ProposalFunctions.__calculate_distance(sec.position, lepton_position)

                energy = sec.energy / pp_MeV * units.MeV

                shower_type, code, name = self.__shower_properties(sec.type)
                
                shower_inducing_prods.append(
                    SecondaryProperties(distance, energy, shower_type, code, name, 
                                        sec.parent_particle_energy / pp_MeV * units.MeV))

        return shower_inducing_prods

    def __group_decay_products(self,
                               decay_products,
                               min_energy_loss,
                               lepton_position,
                               decay_energy):
        """
        group remaining shower-inducing decay products so that they create a single shower
        
        Parameters
        ----------
        decay_products: list of proposal.particle.ParticleState
            List of decay particles that we want to group
        min_energy_loss: float
            Threshold for shower production, in PROPOSAL units (MeV)
        lepton_position: (float, float, float) tuple
            Original lepton positions in proposal units (cm)
        decay_energy: float
            Energy of the lepton before decaying (in NuRadioMC units)

        Returns
        -------
        None or SecondaryProperties
            If energy of grouped decay particles is above min_energy_loss, SecondaryProperties object,
            containing information about the grouped decay particles is returned.
            Otherwise, None is returned.
        """

        # TODO: At the moment, all decay_products in primary_names are grouped and identified
        #       as a 'Hadronic Decay bundle', even electromagnetic decay products such as
        #       electrons. Is that ok? Should they be negleceted? Or grouped separately?

        sum_decay_particle_energy = 0

        for decay_particle in decay_products:
            if is_shower_primary(decay_particle.type):
                sum_decay_particle_energy += decay_particle.energy
        
        if sum_decay_particle_energy > min_energy_loss:
            # all decay_particles have the same position, so we can just look at the first in list
            distance = ProposalFunctions.__calculate_distance(decay_products[0].position, lepton_position)

            return SecondaryProperties(distance, sum_decay_particle_energy / pp_MeV * units.MeV,
                                       'had', 86, particle_names.particle_name(86), 
                                       decay_energy)
        return None


    def get_secondaries_array(self,
                              energy_leptons_nu,
                              lepton_codes,
                              lepton_positions_nu=None,
                              lepton_directions=None,
                              low_nu=0.5 * units.PeV,
                              propagation_length_nu=1000 * units.km,
                              min_energy_loss_nu=0.5 * units.PeV,
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
        low_nu: float
            Low energy limit for the propagating particle in NuRadioMC units (eV)
            controls the minimum energy of the particle. Below this energy, the propagated particle will be discarded
        propagation_length_nu: float
            Maximum propagation length in NuRadioMC units (m)
        min_energy_loss_nu: float
            Minimum energy for a selected secondary-induced shower (eV)
            controls the minimum energy a secondary shower must have to be returned and saved in an event file
        propagate_decay_muons: bool
            If True, muons created by tau decay are propagated and their induced
            showers are stored

        Returns
        -------
        secondaries_array: 2D-list containing SecondaryProperties objects
            For each primary a list containing the information on the shower-inducing secondaries is
            returned. The first dimension indicates the (index of the) primary lepton and the second dimension
            navigates through the secondaries produced by that primary (time-ordered). The SecondaryProperties
            properties are in NuRadioMC units.
        """

        # Converting to PROPOSAL units
        low = low_nu * pp_eV
        propagation_length = propagation_length_nu * pp_m
        min_energy_loss = min_energy_loss_nu * pp_eV
        energy_leptons = np.array(energy_leptons_nu) * pp_eV

        if lepton_positions_nu is None:
            lepton_positions = [(0, 0, 0)] * len(energy_leptons)
        else:
            lepton_positions = [np.array(lepton_position_nu) * pp_m
                                    for lepton_position_nu in lepton_positions_nu]
            
        if lepton_directions is None:
            lepton_directions = [(0, 0, -1)] * len(energy_leptons)

        secondaries_array = []

        for energy_lepton, lepton_code, lepton_position, lepton_direction in zip(energy_leptons,
            lepton_codes, lepton_positions, lepton_directions):

            secondaries = self.__propagate_particle(energy_lepton, lepton_code,
                                                    lepton_position, lepton_direction,
                                                    propagation_length, low=low)

            shower_inducing_prods = self.__filter_secondaries(secondaries.stochastic_losses(), 
                                                              min_energy_loss, lepton_position)

            decay_products = secondaries.decay_products() # array of decay particles
            # Checking if there is a muon in the decay products
            if propagate_decay_muons:

                for decay_particle in list(decay_products):

                    if abs(decay_particle.type) == 13:
                        mu_energy = decay_particle.energy
                        if mu_energy <= low:
                            continue
                        
                        mu_position = (decay_particle.position.x, decay_particle.position.y, decay_particle.position.z)
                        mu_direction = (decay_particle.direction.x, decay_particle.direction.y, decay_particle.direction.z)

                        muon_secondaries = self.__propagate_particle(mu_energy, decay_particle.type,
                                                                     mu_position, mu_direction,
                                                                     propagation_length, low=low)

                        shower_inducing_muon_secondaries = self.__filter_secondaries(muon_secondaries.stochastic_losses(),
                                                                                     min_energy_loss, lepton_position)

                        shower_inducing_prods.extend(shower_inducing_muon_secondaries)

                        # We have already handled the muon, remove it to avoid double counting.
                        decay_products.remove(decay_particle)

            decay_energy = secondaries.final_state().energy / pp_MeV * units.MeV  # energy of the lepton before decay
            grouped_decay_products = self.__group_decay_products(decay_products, min_energy_loss, lepton_position, decay_energy)

            if grouped_decay_products is not None:
                shower_inducing_prods.append(grouped_decay_products)

            secondaries_array.append(shower_inducing_prods)

        return secondaries_array

    def get_decays(self,
                   energy_leptons_nu,
                   lepton_codes,
                   lepton_positions_nu=None,
                   lepton_directions=None,
                   low_nu=0.1 * units.PeV,
                   propagation_length_nu=1000 * units.km):
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
            lepton_positions = [np.array(lepton_position_nu) * pp_m
                                    for lepton_position_nu in lepton_positions_nu]

        if lepton_directions is None:
            lepton_directions = [(0, 0, 1)] * len(energy_leptons)

        decays_array = []

        for energy_lepton, lepton_code, lepton_position, lepton_direction in zip(energy_leptons,
            lepton_codes, lepton_positions, lepton_directions):

            decay_prop = (None, None)

            while(decay_prop == (None, None)):

                secondaries = self.__propagate_particle(energy_lepton, lepton_code, lepton_position, lepton_direction,
                                                        propagation_length, low=low)


                decay_particles = secondaries.decay_products()
                decay_energies = np.array([p.energy for p in decay_particles])
                decay_energy = np.sum(decay_energies) / pp_MeV * units.MeV

                # TODO: Is it physical to repeat the propagation until a decay (before the energy of low) happened?
                if (len(decay_particles) == 0):
                    decay_prop = (None, None)
                    continue

                # all decay particles have the same position, so we can just use the position of the first one
                decay_distance = ProposalFunctions.__calculate_distance(decay_particles[0].position, lepton_position)

                # TODO: Note that this includes ALL decay particles, including invisible ones like neutrinos
                decay_prop = (decay_distance, decay_energy)
                decays_array.append(decay_prop)

        return np.array(decays_array)
