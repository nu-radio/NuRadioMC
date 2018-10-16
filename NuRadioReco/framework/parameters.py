from enum import Enum


class stationParameters(Enum):
    nu_zenith = 1  # the zenith angle of the incoming neutrino direction
    nu_azimuth = 2  # the azimuth angle of the incoming neutrino direction
    nu_energy = 3  # the energy of the neutrino
    nu_flavor = 4  # the flavor of the neutrino
    ccnc = 5  # neutral current of charged current interaction
    nu_vertex = 6  # the neutrino vertex position
    inelasticity = 7  # inelasiticy ot neutrino interaction
    triggered = 8  # flag if station was triggered or not
    cr_energy = 9  # the cosmic-ray energy
    cr_zenith = 10  # zenith angle of the cosmic-ray incoming direction
    cr_azimuth = 11  # azimuth angle of the cosmic-ray incoming direction
    channels_max_amplitude = 12  # the maximum amplitude of all channels (considered in the trigger module)
    zenith = 13  # the zenith angle of the incoming signal direction (WARNING: this parameter is not well defined as the incoming signal direction might be different for different channels)
    azimuth = 14  # the azimuth angle of the incoming signal direction (WARNING: this parameter is not well defined as the incoming signal direction might be different for different channels)
    zenith_cr_templatefit = 15
    zenith_nu_templatefit = 16
    polarization_angle = 17
    polarization_angle_expectation = 18
    cr_xcorrelations = 19 # dict of result of crosscorrelations with cr templates
    nu_xcorrelations = 20 #  dict of result of crosscorrelations with nu templates
    signal_energy_fluence = 21
    polarization_angle = 22
    signal_time = 23
    efield_vector = 24
    efield_vector_polarization = 25

class channelParameters(Enum):
    zenith = 1  # zenith angle of the incoming signal direction
    azimuth = 2  # azimuth angle of the incoming signal direction
    ray_path_type = 3  # the type of the ray tracing solution ('direct', 'refracted' or 'reflected')
    maximum_amplitude = 4  # the maximum ampliude of the magnitude of the trace
    SNR = 5  # an dictionary of various signal-to-noise ratio definitions
    maximum_amplitude_envelope = 6  # the maximum ampliude of the hilbert envelope of the trace
    P2P_amplitude = 7  # the peak to peak amplitude
    cr_xcorrelations = 8 # dict of result of crosscorrelations with cr templates
    nu_xcorrelations = 9 #  dict of result of crosscorrelations with nu templates
