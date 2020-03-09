from aenum import Enum


class stationParameters(Enum):
    nu_zenith = 1  # the zenith angle of the incoming neutrino direction
    nu_azimuth = 2  # the azimuth angle of the incoming neutrino direction
    nu_energy = 3  # the energy of the neutrino
    nu_flavor = 4  # the flavor of the neutrino
    ccnc = 5  # neutral current of charged current interaction
    nu_vertex = 6  # the neutrino vertex position
    inelasticity = 7  # inelasticity ot neutrino interaction
    triggered = 8  # flag if station was triggered or not
    cr_energy = 9  # the cosmic-ray energy
    cr_zenith = 10  # zenith angle of the cosmic-ray incoming direction
    cr_azimuth = 11  # azimuth angle of the cosmic-ray incoming direction
    channels_max_amplitude = 12  # the maximum amplitude of all channels (considered in the trigger module)
    zenith = 13  # the zenith angle of the incoming signal direction (WARNING: this parameter is not well defined as the incoming signal direction might be different for different channels)
    azimuth = 14  # the azimuth angle of the incoming signal direction (WARNING: this parameter is not well defined as the incoming signal direction might be different for different channels)
    zenith_cr_templatefit = 15
    zenith_nu_templatefit = 16
    cr_xcorrelations = 19  # dict of result of crosscorrelations with cr templates
    nu_xcorrelations = 20  # dict of result of crosscorrelations with nu templates
    station_time = 21
    cr_energy_em = 24  # the electromagnetic shower energy (the cosmic ray energy that ends up in electrons, positrons and gammas)
    nu_inttype = 25  # interaction type, e.g., cc, nc, tau_em, tau_had
    chi2_efield_time_direction_fit = 26  # the chi2 of the direction fitter that used the maximum pulse times of the efields
    ndf_efield_time_direction_fit = 27  # the number of degrees of freedom of the direction fitter that used the maximum pulse times of the efields
    cr_xmax = 28  # Depth of shower maximum of the air shower


class channelParameters(Enum):
    zenith = 1  # zenith angle of the incoming signal direction
    azimuth = 2  # azimuth angle of the incoming signal direction
    maximum_amplitude = 4  # the maximum ampliude of the magnitude of the trace
    SNR = 5  # an dictionary of various signal-to-noise ratio definitions
    maximum_amplitude_envelope = 6  # the maximum ampliude of the hilbert envelope of the trace
    P2P_amplitude = 7  # the peak to peak amplitude
    cr_xcorrelations = 8  # dict of result of crosscorrelations with cr templates
    nu_xcorrelations = 9  # dict of result of crosscorrelations with nu templates
    signal_time = 10  # the time of the maximum amplitude of the envelope


class electricFieldParameters(Enum):
    ray_path_type = 1  # the type of the ray tracing solution ('direct', 'refracted' or 'reflected')
    polarization_angle = 2  # electric field polarization in onsky-coordinates. 0 corresponds to polarization in e_theta, 90deg is polarization in e_phi
    polarization_angle_expectation = 3  # expected polarization based on shower geometry. Defined analogous to polarization_angle
    signal_energy_fluence = 4  # Energy/area in the radio signal
    cr_spectrum_slope = 5  # Slope of the radio signal's spectrum as reconstructed by the voltageToAnalyticEfieldConverter
    zenith = 7  # zenith angle of the signal. Note that refraction at the air/ice boundary is not taken into account
    azimuth = 8  # azimuth angle of the signal. Note that refraction at the air/ice boundary is not taken into account
    signal_time = 9
    nu_vertex_distance = 10  # the distance along the ray path from the vertex to the channel
    nu_viewing_angle = 11  # the angle between shower axis and launch vector
    max_amp_antenna = 12  # the maximum amplitude of the signal after convolution with the antenna response pattern, dict with channelid as key
    max_amp_antenna_envelope = 13  # the maximum amplitude of the signal envelope after convolution with the antenna response pattern, dict with channelid as key
    reflection_coefficient_theta = 14  # for reflected rays: the complex Fresnel reflection coefficient of the eTheta component
    reflection_coefficient_phi = 15  # for reflected rays: the complex Fresnel reflection coefficient of the ePhi component
    cr_spectrum_quadratic_term = 16  # result of the second order correction to the spectrum fitted by the voltageToAnalyticEfieldConverter


class ARIANNAParameters(Enum):  # this class stores parameters specific to the ARIANNA data taking
    seq_start_time = 1  # the start time of a sequence
    seq_stop_time = 2  # the stop time of a sequence
    seq_num = 3  # the sequence number of the current event
    comm_period = 4  # length of data taking window
    comm_duration = 5  # maximum diration of communication window
    trigger_thresholds = 6  # trigger thresholds converted to voltage
    l1_supression_value = 7  # This provieds the L1 supression value for given event
    internal_clock_time = 8  # time since last trigger with ms precision


class showerParameters(Enum):
    zenith = 1  # zenith angle of the shower axis pointing towards xmax
    azimuth = 2  # azimuth angle of the shower axis pointing towards xmax
    core = 3  # position of the intersection between shower axis and an observer plane
    energy = 4  # total energy of the primary particle
    electromagnetic_energy = 5  # energy of the electromagnetic shower component
    radiation_energy = 6  # totally emitted radiation energy
    electromagnetic_radiation_energy = 7  # radiation energy originated from the electromagnetic emission
    primary_particle = 8  # particle id of the primary particle
    shower_maximum = 9  # position of shower maximum in slant depth, e.g., Xmax
    distance_shower_maximum_geometric = 10  # distance to xmax in meter
    distance_shower_maximum_grammage = 11  # distance to xmax in g / cm^2

    # dedicated parameter for sim showers
    refractive_index_at_ground = 100  # refractivity at sea level
    atmospheric_model = 101  # atmospheric model used in simulation
    # offset between magnetic field and north in reconstruction corrdinatesystem
    magnetic_field_rotation = 102
    magnetic_field_vector = 103  # magnetic field used in simulation in local coordinate system
    observation_level = 104  # altitude a.s.l where the particles are stored

    charge_excess_profile_id = 105  # the id of the charge-excess profile used in the ARZ Askaryan calculation
    type = 106  # for neutrino induces showers in ice: can be "HAD" or "EM"


class eventParameters(Enum):
    sim_config = 1  # contents of the config file that the NuRadioMC simulation was run with
