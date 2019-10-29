import pyPROPOSAL as pp
import numpy as np
from NuRadioMC.utilities import units
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
pp_MeV = 1.e0
pp_GeV = 1.e3
pp_TeV = 1.e6
pp_PeV = 1.e9
pp_EeV = 1.e12
pp_ZeV = 1.e15

pp_m = 1.e2
pp_km = 1.e5

class SecondaryProperties:

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
    if particle.particle_def == pp.particle.EMinusDef.get(): return True
    elif particle.particle_def == pp.particle.EPlusDef.get(): return True
    elif particle.particle_def == pp.particle.GammaDef.get(): return True
    else:
       return False

def is_had_primary(particle):
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
    return is_em_primary(particle) or is_had_primary(particle)

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

def filter_particle(secondaries, particle):
    prods = [p for p in secondaries if p.id == pp.particle.Data.Particle]
    E = [p.energy for p in prods if p.particle_def == particle]
    return sum(E)

def create_propagator(low=0.1*pp_PeV, particle_code=13, ecut=100*pp_TeV,
                      config_file='SouthPole'):

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
    elif (os.path.exists(config_file)):
        config_file_full_path = config_file
    else:
        raise ValueError("Proposal config file is not valid. Please provide a valid option.")


    propagator = pp.Propagator(particle_def=mu_def, config_file=config_file_full_path)

    return propagator

def get_compact_sub_pev_losses(energy_arr, distance_arr, compact_dist, min_energy_loss):
    r""" return biggest compact loss if above min_energy_cut
    Parameters
    ----------
    energy_arr: array-like
        energy of the energy losses below min_energy_loss
    distance_arr: array_like
        distances of the energy losses below min_energy_loss
    compact_dist: float
        distance in centimeters (PROPOSAL units): how compact the energy losses should be
    min_energy_loss: float
        min energy for the sensitivity (here a PeV)
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

def produces_shower(particle, min_energy_loss=1*pp_PeV):

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

def shower_properties(particle):

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

def propagate_particle(energy_lepton, lepton_code, lepton_position, lepton_direction,
                       propagation_length, propagators):

    x, y, z = lepton_position
    px, py, pz = lepton_direction
    propagators[lepton_code].particle.position = pp.Vector3D(x, y, z)
    propagators[lepton_code].particle.direction = pp.Vector3D(px, py, pz)
    propagators[lepton_code].particle.propagated_distance = 0
    propagators[lepton_code].particle.energy = energy_lepton / units.MeV # Proposal's energy unit is the MeV

    secondaries = propagators[lepton_code].propagate(propagation_length)

    return secondaries

def filter_secondaries(secondaries, min_energy_loss, lepton_position):

    shower_inducing_prods = []

    for sec in secondaries:

        # Decays contain only one shower-inducing particle
        # Muons and neutrinos resulting from decays are ignored
        if produces_shower(sec, min_energy_loss):

            distance  = ( (sec.position.x - lepton_position[0]) * units.cm )**2
            distance += ( (sec.position.y - lepton_position[1]) * units.cm )**2
            distance += ( (sec.position.z - lepton_position[2]) * units.cm )**2
            distance  = np.sqrt(distance)
            energy = sec.energy * units.MeV

            shower_type, code, name = shower_properties(sec)

            shower_inducing_prods.append( SecondaryProperties(distance, energy, shower_type, code, name) )

    return shower_inducing_prods

def get_secondaries(energy_lepton, lepton_code, random_seed=None, config_file='SouthPole'):

    low = 0.1*pp_PeV # Low energy limit for the propagating particle
    propagation_length = 1000*pp_km # Maximum propagation length
    compact_dist = 10*pp_m # Maximum distance for the compact losses

    if random_seed == None:
        random_seed = int( np.random.uniform(0,1e8) )

    pp.RandomGenerator.get().set_seed( random_seed )

    prop = create_propagator(low=low, particle_code=lepton_code, config_file=config_file)
    prop.particle.position = pp.Vector3D(0, 0, 0)
    prop.particle.direction = pp.Vector3D(0, 0, 1)
    prop.particle.propagated_distance = 0
    prop.particle.energy = energy_lepton / units.MeV # Proposal's energy unit is the MeV

    secondaries = prop.propagate(propagation_length)

    return secondaries

def get_secondaries_array(energy_leptons, lepton_codes, lepton_positions = None, lepton_directions = None,
                        random_seed=None, config_file='SouthPole'):

    low = 0.1*pp_PeV # Low energy limit for the propagating particle
    propagation_length = 1000*pp_km # Maximum propagation length
    compact_dist = 10*pp_m # Maximum distance for the compact losses
    min_energy_loss = 1*pp_PeV # Minimal energy for a selected secondary-induced shower

    if random_seed == None:
        random_seed = int( np.random.uniform(0,1e8) )

    propagators = {}
    for lepton_code in np.unique(lepton_codes):
        if lepton_code not in propagators:
            propagators[lepton_code] = create_propagator(low=low, particle_code=lepton_code,
                                                         config_file=config_file)

    secondaries_array = []

    if lepton_positions is None:
        lepton_positions = [(0, 0, 0)] * len(energy_leptons)
    if lepton_directions is None:
        lepton_directions = [(0, 0, -1)] * len(energy_leptons)

    for energy_lepton, lepton_code, lepton_position, lepton_direction in zip(energy_leptons,
        lepton_codes, lepton_positions, lepton_directions):

        secondaries = propagate_particle(energy_lepton, lepton_code, lepton_position, lepton_direction,
                                         propagation_length, propagators)

        shower_inducing_prods = filter_secondaries(secondaries, min_energy_loss, lepton_position)

        # Checking if there is a muon and propagating to know if it creates
        # particle showers.
        for sec in secondaries:

            if (sec.id != pp.particle.Data.Particle):
                continue

            if (sec.particle_def == pp.particle.MuMinusDef.get()) or (sec.particle_def == pp.particle.MuPlusDef.get()):

                if sec.particle_def == pp.particle.MuMinusDef.get():
                    mu_code = 13
                elif sec.particle_def == pp.particle.MuPlusDef.get():
                    mu_code = -13

                if mu_code not in propagators:
                    propagators[mu_code] = create_propagator(low=low, particle_code=mu_code,
                                                             config_file=config_file)

                mu_energy = sec.energy
                mu_position = (sec.position.x, sec.position.y, sec.position.z)
                mu_direction = lepton_direction

                mu_secondaries = propagate_particle(mu_energy, mu_code, mu_position, mu_direction,
                                                    propagation_length, propagators)

                mu_shower_inducing_prods = filter_secondaries(mu_secondaries, min_energy_loss, lepton_position)

                shower_inducing_prods += mu_shower_inducing_prods

        # group shower-inducing decay products so that they create a single shower
        min_distance = 0.1 * units.m
        while( len(shower_inducing_prods) > 1 and
               np.abs(shower_inducing_prods[-1].distance - shower_inducing_prods[-2].distance) < min_distance):

            last_decay_prod = shower_inducing_prods.pop(-1)
            shower_inducing_prods[-1].energy += last_decay_prod.energy
            shower_inducing_prods[-1].code = 86
            shower_inducing_prods[-1].name = particle_name[86]

        secondaries_array.append(shower_inducing_prods)

    return secondaries_array

def get_decays(energy_leptons, lepton_codes, random_seed=None, config_file='InfIce'):

    low = 0.1*pp_PeV # Low energy limit for the propagating particle
    propagation_length = 1000*pp_km # Maximum propagation length
    compact_dist = 10*pp_m # Maximum distance for the compact losses
    min_energy_loss = 1*pp_PeV # Minimal energy for a selected secondary-induced shower

    if random_seed == None:
        random_seed = int( np.random.uniform(0,1e8) )

    propagators = {}
    for lepton_code in lepton_codes:
        if lepton_code not in propagators:
            propagators[lepton_code] = create_propagator(low=low, particle_code=lepton_code,
                                                         config_file=config_file)

    decays_array = []

    for energy_lepton, lepton_code in zip(energy_leptons, lepton_codes):

        decay_prop = (None, None)

        while( decay_prop == (None,None) ):

            propagators[lepton_code].particle.position = pp.Vector3D(0, 0, 0)
            propagators[lepton_code].particle.direction = pp.Vector3D(0, 0, 1)
            propagators[lepton_code].particle.propagated_distance = 0
            propagators[lepton_code].particle.energy = energy_lepton / units.MeV # Proposal's energy unit is the MeV

            secondaries = propagators[lepton_code].propagate(propagation_length)

            decay_particles = np.array([p for p in secondaries if p.id == pp.particle.Data.Particle])
            decay_energies = np.array([p.energy for p in decay_particles])
            decay_energy = np.sum(decay_energies) * units.MeV

            try:
                decay_distance = decay_particles[0].position.z * units.cm
                decay_prop = (decay_distance, decay_energy)
                decays_array.append(decay_prop)
            except:
                decay_prop = (None, None)

    return np.array(decays_array)

def get_decay(energy_lepton, lepton_code, random_seed=None):

    secondaries = get_secondaries(energy_lepton, lepton_code)

    decay_particles = np.array([p for p in secondaries if p.id == pp.particle.Data.Particle])
    decay_energies = np.array([p.energy for p in decay_particles])
    decay_energy = np.sum(decay_energies) * units.MeV
    try:
        decay_distance = decay_particles[0].position.z * units.cm
    except:
        return (None, None)

    return (decay_distance, decay_energy)

def get_prods_array(energy_leptons, lepton_codes, aggregated_showers=False, random_seed=None):

    min_energy_loss = 1*pp_PeV # Minimal energy for a selected secondary-induced shower

    secondaries_array = get_secondaries_array(energy_leptons, lepton_codes, random_seed)

    shower_inducing_array = []

    for secondaries in secondaries_array:

        shower_inducing_prods = []

        if aggregated_showers:
            pass
        else:
            for sec in secondaries:
                # Decays contain only one shower-inducing particle
                # Muons and neutrinos resulting from decays are ignored
                if produces_shower(sec, min_energy_loss):

                    distance = sec.position.z * units.cm
                    energy = sec.energy * units.MeV

                    shower_type, code, name = shower_properties(sec)

                    shower_inducing_prods.append( SecondaryProperties(distance, energy, shower_type, code, name) )

        shower_inducing_array.append(shower_inducing_prods)

    return shower_inducing_array

def get_prods(energy_lepton, lepton_code, aggregated_showers=False, random_seed=None):

    min_energy_loss = 1*pp_PeV # Minimal energy for a selected secondary-induced shower

    secondaries = get_secondaries(energy_lepton, lepton_code, random_seed)

    shower_inducing_prods = []

    if aggregated_showers:
        pass
    else:
        for sec in secondaries:
            # Decays contain only one shower-inducing particle
            # Muons and neutrinos resulting from decays are ignored
            if produces_shower(sec, min_energy_loss):

                distance = sec.position.z * units.cm
                energy = sec.energy * units.MeV

                shower_type, code, name = shower_properties(sec)

                shower_inducing_prods.append( SecondaryProperties(distance, energy, shower_type, code, name) )

    return shower_inducing_prods
