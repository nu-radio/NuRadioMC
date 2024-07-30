"""
Provides an interface to store simulated and reconstructed quantities

The parameters module provides access to store and read simulated or
reconstructed quantities in the different custom classes used in NuRadioMC.

"""

from aenum import Enum


class stationParameters(Enum):
    nu_zenith = 1  #: the zenith angle of the incoming neutrino direction
    nu_azimuth = 2  #: the azimuth angle of the incoming neutrino direction
    nu_energy = 3  #: the energy of the neutrino
    nu_flavor = 4  #: the flavor of the neutrino
    ccnc = 5  #: neutral current of charged current interaction
    nu_vertex = 6  #: the neutrino vertex position
    inelasticity = 7  #: inelasticity ot neutrino interaction
    triggered = 8  #: flag if station was triggered or not
    cr_energy = 9  #: the cosmic-ray energy
    cr_zenith = 10  #: zenith angle of the cosmic-ray incoming direction
    cr_azimuth = 11  #: azimuth angle of the cosmic-ray incoming direction
    channels_max_amplitude = 12  #: the maximum amplitude of all channels (considered in the trigger module)
    zenith = 13  #: the zenith angle of the incoming signal direction (WARNING: this parameter is not well defined as the incoming signal direction might be different for different channels)
    azimuth = 14  #: the azimuth angle of the incoming signal direction (WARNING: this parameter is not well defined as the incoming signal direction might be different for different channels)
    zenith_cr_templatefit = 15
    zenith_nu_templatefit = 16
    cr_xcorrelations = 19  #: dict of result of crosscorrelations with cr templates
    nu_xcorrelations = 20  #: dict of result of crosscorrelations with nu templates
    station_time = 21
    cr_energy_em = 24  #: the electromagnetic shower energy (the cosmic ray energy that ends up in electrons, positrons and gammas)
    nu_inttype = 25  #: interaction type, e.g., cc, nc, tau_em, tau_had
    chi2_efield_time_direction_fit = 26  #: the chi2 of the direction fitter that used the maximum pulse times of the efields
    ndf_efield_time_direction_fit = 27  #: the number of degrees of freedom of the direction fitter that used the maximum pulse times of the efields
    cr_xmax = 28  #: Depth of shower maximum of the air shower
    vertex_2D_fit = 29  #: horizontal distance and z coordinate of the reconstructed vertex of the neutrino
    distance_correlations = 30
    shower_energy = 31 #: the energy of the shower
    viewing_angles = 32 #: reconstructed viewing angles. A nested map structure. First key is channel id, second key is ray tracing solution id. Value is a float
    flagged_channels = 60  #: a set of flagged channel ids (calculated by readLOFARData and adjusted by stationRFIFilter)
    cr_dominant_polarisation = 61  #: the channel orientation containing the dominant cosmic ray signal (calculated by stationPulseFinder)
    dirty_fft_channels = 62  #: a list of FFT channels flagged as RFI (calculated by stationRFIFilter)

class channelParameters(Enum):
    zenith = 1  #: zenith angle of the incoming signal direction
    azimuth = 2  #: azimuth angle of the incoming signal direction
    maximum_amplitude = 4  #: the maximum ampliude of the magnitude of the trace
    SNR = 5  #: an dictionary of various signal-to-noise ratio definitions
    maximum_amplitude_envelope = 6  #: the maximum ampliude of the hilbert envelope of the trace
    P2P_amplitude = 7  #: the peak to peak amplitude
    cr_xcorrelations = 8  #: dict of result of crosscorrelations with cr templates
    nu_xcorrelations = 9  #: dict of result of crosscorrelations with nu templates
    signal_time = 10  #: the time of the maximum amplitude of the envelope
    noise_rms = 11  #: the root mean square of the noise
    signal_regions = 12     #: list of start and end times of regions that likely contain a signal
    noise_regions = 13      #: list of start and end times of regions that likel do not contain any signals
    signal_time_offset = 14     #: the relative timing differences of the signal arrival times between channels
    signal_receiving_zenith = 15    #: the zenith angle of direction at which the radio signal arrived at the antenna
    signal_ray_type = 16        #: type of the ray propagation path of the signal received by this channel. Options are direct, reflected and refracted
    signal_receiving_azimuth = 17   #: the azimuth angle of direction at which the radio signal arrived at the antenna
    block_offsets = 18 #: 'block' or pedestal offsets. See `NuRadioReco.modules.RNO_G.channelBlockOffsetFitter`


class electricFieldParameters(Enum):
    ray_path_type = 1  #: the type of the ray tracing solution ('direct', 'refracted' or 'reflected')
    polarization_angle = 2  #: electric field polarization in onsky-coordinates. 0 corresponds to polarization in e_theta, 90deg is polarization in e_phi
    polarization_angle_expectation = 3  #: expected polarization based on shower geometry. Defined analogous to polarization_angle
    signal_energy_fluence = 4  #: Energy/area in the radio signal
    cr_spectrum_slope = 5  #: Slope of the radio signal's spectrum as reconstructed by the voltageToAnalyticEfieldConverter
    zenith = 7  #: zenith angle of the signal. Note that refraction at the air/ice boundary is not taken into account
    azimuth = 8  #: azimuth angle of the signal. Note that refraction at the air/ice boundary is not taken into account
    signal_time = 9
    nu_vertex_distance = 10  #: the distance along the ray path from the vertex to the channel
    nu_viewing_angle = 11  #: the angle between shower axis and launch vector
    max_amp_antenna = 12  #: the maximum amplitude of the signal after convolution with the antenna response pattern, dict with channelid as key
    max_amp_antenna_envelope = 13  #: the maximum amplitude of the signal envelope after convolution with the antenna response pattern, dict with channelid as key
    reflection_coefficient_theta = 14  #: for reflected rays: the complex Fresnel reflection coefficient of the eTheta component
    reflection_coefficient_phi = 15  #: for reflected rays: the complex Fresnel reflection coefficient of the ePhi component
    cr_spectrum_quadratic_term = 16  #: result of the second order correction to the spectrum fitted by the voltageToAnalyticEfieldConverter
    energy_fluence_ratios = 17   #: Ratios of the energy fluences in different passbands


class ARIANNAParameters(Enum):  #: this class stores parameters specific to the ARIANNA data taking
    seq_start_time = 1  #: the start time of a sequence
    seq_stop_time = 2  #: the stop time of a sequence
    seq_num = 3  #: the sequence number of the current event
    comm_period = 4  #: length of data taking window
    comm_duration = 5  #: maximum diration of communication window
    trigger_thresholds = 6  #: trigger thresholds converted to voltage
    l1_supression_value = 7  #: This provieds the L1 supression value for given event
    internal_clock_time = 8  #: time since last trigger with ms precision


class showerParameters(Enum):
    zenith = 1  #: zenith angle of the shower axis pointing towards xmax
    azimuth = 2  #: azimuth angle of the shower axis pointing towards xmax
    core = 3  #: position of the intersection between shower axis and an observer plane
    energy = 4  #: total energy of the primary particle, or shower energy for in-ice particle showers
    electromagnetic_energy = 5  #: energy of the electromagnetic shower component
    radiation_energy = 6  #: totally emitted radiation energy
    electromagnetic_radiation_energy = 7  #: radiation energy originated from the electromagnetic emission
    primary_particle = 8  #: particle id of the primary particle
    shower_maximum = 9  #: position of shower maximum in slant depth, e.g., Xmax
    distance_shower_maximum_geometric = 10  #: distance to xmax in meter
    distance_shower_maximum_grammage = 11  #: distance to xmax in g / cm^2
    parent_id = 12 #: id of parent in sim particles

    #: dedicated parameter for sim showers
    refractive_index_at_ground = 100  #: refractivity at sea level
    atmospheric_model = 101  #: atmospheric model used in simulation
    #: offset between magnetic field and north in reconstruction corrdinatesystem
    magnetic_field_rotation = 102
    magnetic_field_vector = 103  #: magnetic field used in simulation in local coordinate system
    observation_level = 104  #: altitude a.s.l where the particles are stored

    charge_excess_profile_id = 105  #: the id of the charge-excess profile used in the ARZ Askaryan calculation
    type = 106  #: for neutrino induces showers in ice: can be "HAD" or "EM"
    vertex = 107  #: the interaction vertex (for air showers this corresponds to the point of X0)
    vertex_time = 108  #: the propagation time relative to the first interactions
    interaction_type = 109  #: the interaction type, e.g. cc or nc
    k_L = 110  #: the k_L parameter of the Alvarez2009 parameter that controls the longitudional width of the charge excess profile
    flavor = 111  #: the flavor of the particle initiating the shower

    interferometric_shower_maximum = 120  #: depth of the maximum of the longitudinal profile of the beam-formed signal
    interferometric_shower_axis = 121  #: shower axis (direction) derived from beam-formed signal
    interferometric_core = 122  #: core (intersection of shower axis with obs plane) derived from beam-formed signal


class emitterParameters(Enum):
    position = 1  #: the interaction vertex (for air showers this corresponds to the point of X0)
    model = 2  #: the emitter model used to simulate the emission (as defined in NuRadioMC/SignalGen/emitter.py)
    amplitude = 3  #: the amplitude of the signal
    polarization = 4  #: the polarization of the signal
    half_width = 5  #: the width of square and tone_burst signal
    frequency = 6  #: the frequency of a signal (for cw and tone_burst model)
    orientation_phi = 7  #: the orientation of the emiting antenna, defined via two vectors that are defined with two angles each
    orientation_theta = 8  #: the orientation of the emiting antenna, defined via two vectors that are defined with two angles each
    rotation_phi = 9  #: the orientation of the emiting antenna, defined via two vectors that are defined with two angles each
    rotation_theta = 10  #: the orientation of the emiting antenna, defined via two vectors that are defined with two angles each
    realization_id = 11  #: the id of the measurement of the emitted electric field


class particleParameters(Enum):
    parent_id = 1 #: the entry number of the parent particle, None if primary.
    zenith = 2  #: the zenith angle of the incoming neutrino direction
    azimuth = 3  #: the azimuth angle of the incoming neutrino direction
    energy = 4  #: the energy of the neutrino
    flavor = 5  #: the flavor of the neutrino, more generally the PDG code
    vertex = 6  #: the neutrino vertex position (x,y,z)
    vertex_time = 9
    weight = 10
    inelasticity = 11  #: inelasticity ot neutrino interaction
    interaction_type = 12  #: interaction type, e.g., cc, nc
    n_interaction = 13 #: number of interaction

    cr_energy = 101  #: the cosmic-ray energy
    cr_zenith = 102  #: zenith angle of the cosmic-ray incoming direction
    cr_azimuth = 103  #: azimuth angle of the cosmic-ray incoming direction
    cr_energy_em = 104  #: the electromagnetic shower energy (the cosmic ray energy that ends up in electrons, positrons and gammas)

class generatorAttributes(Enum):
    Emax = 1 #: maximum simulated energy
    Emin = 2 #: minimum simulated energy

    deposited = 3 #: deposited energies or neutrino energies?

    fiducial_rmin = 4 #: fiducial volume parameter (if cylindrical footprint used)
    fiducial_rmax = 5 #: fiducial volume parameter (if cylindrical footprint used)

    fiducial_xmin = 6 #: fiducial volume parameter (if rectangular footprint used)
    fiducial_xmax = 7 #: fiducial volume parameter (if rectangular footprint used)
    fiducial_ymin = 8 #: fiducial volume parameter (if rectangular footprint used)
    fiducial_ymax = 9 #: fiducial volume parameter (if rectangular footprint used)

    fiducial_zmin = 10
    fiducial_zmax = 11

    rmin = 12 #: volume parameter (if cylindrical)
    rmax = 13 #: volume parameter (if cylindrical)

    xmin = 14 #: volume parameter (if rectangular)
    xmax = 15 #: volume parameter (if rectangular)
    ymin = 16 #: volume parameter (if rectangular)
    ymax = 17 #: volume parameter (if rectangular)

    zmin = 18
    zmax = 19

    # volume calculated from the (z r) min max or (x y z) min max parameters
    volume = 20
    area = 21

    phimax = 22 #: simulated space angle range
    phimin = 23 #: simulated space angle range
    thetamax = 24 #: simulated space angle range
    thetamin = 25 #: simulated space angle range

    flavors = 26 #: list of simulated event flavours
    dt = 27 #: inverse of sampling rate used in the simulation

    # simulated statistics
    n_events = 100
    n_samples = 101
    start_event_id = 102
    total_number_of_events = 103

    # version numbers
    NuRadioMC_EvtGen_version = 200
    NuRadioMC_EvtGen_version_hash = 201
    NuRadioMC_version = 202
    NuRadioMC_version_hash = 203

class eventParameters(Enum):
    sim_config = 1 #: contents of the config file that the NuRadioMC simulation was run with
    hash_NuRadioReco = 2 #: deprecated, since NuRadioReco is no longer its own repository
    hash_NuRadioMC = 3 #: git hash of the NuRadioMC commit that the file was created with
