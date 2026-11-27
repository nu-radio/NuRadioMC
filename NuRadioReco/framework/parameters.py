"""
Provides an interface to store simulated and reconstructed quantities

The parameters module provides access to store and read simulated or
reconstructed quantities in the different custom classes used in NuRadioMC.

"""

import typing
from typing import Any, Dict, List, TypedDict

import astropy.time
import numpy as np
from aenum import Enum


# Values of the ``*_xcorrelations`` parameters (written by `NuRadioReco.modules.channelTemplateCorrelation`).
# ``total=False``: keys may be missing; the channel keys ``*_all``, ``*_max`` and ``*_template`` exist only
# if several templates were used. Stored as arrow structs (see `NuRadioReco.utilities.parquet_utilities`).
class StationCRXcorrelations(TypedDict, total=False):
    number_of_templates: int
    cr_max_xcorr: float
    cr_max_xcorr_crchannels: float
    cr_avg_xcorr_crchannels: float
    cr_avg_xcorr_parallel_crchannels: float


class StationNuXcorrelations(TypedDict, total=False):
    """As `StationCRXcorrelations`, for neutrino templates."""

    number_of_templates: int
    nu_max_xcorr: float
    nu_max_xcorr_nuchannels: float
    nu_avg_xcorr_nuchannels: float
    nu_avg_xcorr_parallel_nuchannels: float


class ChannelCRXcorrelations(TypedDict, total=False):
    cr_ref_xcorr: float  #: correlation (mean over templates if several)
    cr_ref_xcorr_time: float  #: time of the correlation maximum (relative to the trace start)
    cr_ref_xcorr_all: np.ndarray  #: absolute value of the correlation per template
    cr_ref_xcorr_max: float  #: absolute value of the correlation of the best template
    cr_ref_xcorr_template: int  #: key of the best template


class ChannelNuXcorrelations(TypedDict, total=False):
    """As `ChannelCRXcorrelations`, for neutrino templates."""

    nu_ref_xcorr: float
    nu_ref_xcorr_time: float
    nu_ref_xcorr_all: np.ndarray
    nu_ref_xcorr_max: float
    nu_ref_xcorr_template: int


# Value of `electricFieldParameters.raytracing_solution`: the dictionary returned by
# `get_raytracing_output(i_solution)` of the propagators in `NuRadioMC.SignalProp` (each sets a subset of the keys).
class RaytracingSolution(TypedDict, total=False):
    ray_tracing_C0: float  #: analytic ray tracer: parameter C_0 of the ray path
    ray_tracing_C1: float  #: analytic ray tracer: parameter C_1 of the ray path
    ray_tracing_reflection: int  #: number of bottom reflections (reflective layer in the ice)
    ray_tracing_reflection_case: int  #: 1: ray starts upwards, 2: downwards (only relevant for bottom reflections)
    ray_tracing_solution_type: int  #: 1: direct, 2: refracted, 3: reflected (off the surface)
    focusing_factor: float
    launch_vector: np.ndarray  #: radiopropa ray tracer
    sphere_sizes: np.ndarray  #: radiopropa ray tracer


class TypedEnum(Enum):
    """
    Enum whose members are declared as ``name = id, type``.

    ``value`` remains the integer id (so ``Parameters(id)`` works), the expected
    type of the stored parameter is available as ``dtype``. The parquet writer derives the
    column type from it (``Any`` or a bare ``dict``/``list``: no column; see
    ``parquet_format.rst``); setting a parameter does not check it.
    """

    _init_ = 'value dtype'


class OneOf:
    """
    Declared type of a parameter whose value has one of several types, e.g. one threshold for all channels
    or a dict channel id -> threshold: ``OneOf(value=float, per_channel=dict[int, float])``.

    The names label the alternatives (parquet stores each in its own field of a struct column, the others are
    null). Alternatives are matched in the given order, so a more specific type goes before a more general one.
    """

    def __init__(self, **alternatives):
        self.alternatives = alternatives

    def select(self, value):
        """
        The name and type of the first alternative matching `value` (None if none matches).

        Only the container kind is checked: a dict matches a ``dict[...]`` alternative whose value type is a dict
        if and only if its first value is a dict; a list, tuple or array matches ``list[...]`` / `numpy.ndarray`;
        anything else matches the remaining (scalar) alternatives.
        """
        for name, dtype in self.alternatives.items():
            if _kind(dtype) == _kind_of_value(value):
                return name, dtype

        return None


def _kind(dtype):
    """Container kind of a declared type: "dict of dict", "dict", "sequence" or "scalar"."""
    origin = typing.get_origin(dtype)
    if origin is dict:
        value_type = typing.get_args(dtype)[1]
        return "dict of dict" if typing.get_origin(value_type) is dict else "dict"

    if origin in (list, tuple) or dtype is np.ndarray:
        return "sequence"

    return "scalar"


def _kind_of_value(value):
    """Container kind of a value (see `_kind`)."""
    if isinstance(value, dict):
        first = next(iter(value.values()), None)
        return "dict of dict" if isinstance(first, dict) else "dict"

    if isinstance(value, (list, tuple, np.ndarray)):
        return "sequence"

    return "scalar"

    def __repr__(self):
        return "OneOf({})".format(", ".join(f"{name}={dtype!r}" for name, dtype in self.alternatives.items()))


class stationParameters(TypedEnum):
    nu_zenith = 1, float  #: the zenith angle of the incoming neutrino direction
    nu_azimuth = 2, float  #: the azimuth angle of the incoming neutrino direction
    nu_energy = 3, float  #: the energy of the neutrino
    nu_flavor = 4, int  #: the flavor of the neutrino
    ccnc = 5, str  #: neutral current of charged current interaction
    nu_vertex = 6, np.ndarray  #: the neutrino vertex position
    inelasticity = 7, float  #: inelasticity ot neutrino interaction
    triggered = 8, bool  #: flag if station was triggered or not
    cr_energy = 9, float  #: the cosmic-ray energy
    cr_zenith = 10, float  #: zenith angle of the cosmic-ray incoming direction
    cr_azimuth = 11, float  #: azimuth angle of the cosmic-ray incoming direction
    channels_max_amplitude = 12, float  #: the maximum amplitude of all channels (considered in the trigger module)
    zenith = (
        13,
        float,
    )  #: the zenith angle of the incoming signal direction (WARNING: this parameter is not well defined as the incoming signal direction might be different for different channels)
    azimuth = (
        14,
        float,
    )  #: the azimuth angle of the incoming signal direction (WARNING: this parameter is not well defined as the incoming signal direction might be different for different channels)
    zenith_cr_templatefit = 15, float
    zenith_nu_templatefit = 16, float
    cr_xcorrelations = 19, StationCRXcorrelations  #: dict of result of crosscorrelations with cr templates
    nu_xcorrelations = 20, StationNuXcorrelations  #: dict of result of crosscorrelations with nu templates
    station_time = (
        21,
        astropy.time.Time,
    )  #: the station time. Not a parameter column in parquet: stored as ``station_time_jd1/_jd2/_scale``
    cr_energy_em = (
        24,
        float,
    )  #: the electromagnetic shower energy (the cosmic ray energy that ends up in electrons, positrons and gammas)
    nu_inttype = 25, str  #: interaction type, e.g., cc, nc, tau_em, tau_had
    chi2_efield_time_direction_fit = (
        26,
        float,
    )  #: the chi2 of the direction fitter that used the maximum pulse times of the efields
    ndf_efield_time_direction_fit = (
        27,
        int,
    )  #: the number of degrees of freedom of the direction fitter that used the maximum pulse times of the efields
    cr_xmax = 28, float  #: Depth of shower maximum of the air shower
    vertex_2D_fit = 29, np.ndarray  #: horizontal distance and z coordinate of the reconstructed vertex of the neutrino
    distance_correlations = (
        30,
        np.ndarray,
    )  #: maximum correlation per tested horizontal distance (neutrino3DVertexReconstructor, no longer set)
    shower_energy = 31, float  #: the energy of the shower
    viewing_angles = (
        32,
        Dict[int, Dict[int, float]],
    )  #: reconstructed viewing angles. A nested map structure. First key is channel id, second key is ray tracing solution id. Value is a float
    flagged_channels = (
        60,
        Dict[int, List[str]],
    )  #: a defaultdict of flagged NRR channel ids with as value a list of the reason(s) for flagging (used in readLOFARData, stationRFIFilter)
    cr_dominant_polarisation = (
        61,
        List[float],
    )  #: the channel orientation containing the dominant cosmic ray signal (calculated by stationPulseFinder)
    dirty_fft_channels = 62, List[int]  #: a list of FFT channels flagged as RFI (calculated by stationRFIFilter)
    channels_max_amplitude_norm = 63, float  #: maximum std-normalised peak to peak amplitude of all chosen channels


class channelParameters(TypedEnum):
    zenith = 1, float  #: zenith angle of the incoming signal direction
    azimuth = 2, float  #: azimuth angle of the incoming signal direction
    maximum_amplitude = 4, float  #: the maximum ampliude of the magnitude of the trace
    SNR = 5, Dict[str, float]  #: a dictionary with the following signal-to-noise ratio definitions:
    # 'integrated_power':
    # Difference of the sum of the squared amplitudes in the signal window and in the noise window
    # SNR = sum_sig(V_i^2) - sum_noise(V_i^2)
    # 'peak_amplitude':
    # Maximum amplitude of the absolute signal trace divided by the rms of the noise window
    # SNR = max(abs(V_sig))/V_rms_noise
    # 'peak_2_peak_amplitude':
    # Difference between max and min of the signal trace divided by twice the rms value in the noise window
    # SNR = (max(V_sig)-min(V_sig))/2*V_rms_noise
    # 'peak_2_peak_amplitude_split_noise_rms':
    # Peak to peak amplitude in the trace divided by twice the noise rms value, where the latter is calculated by splitting the trace into segments and taking the mean of the lowest few segment rms values
    # SNR = V_p2p/2*V_rms_noise
    maximum_amplitude_envelope = 6, float  #: the maximum ampliude of the hilbert envelope of the trace
    P2P_amplitude = 7, float  #: the peak to peak amplitude
    cr_xcorrelations = 8, ChannelCRXcorrelations  #: dict of result of crosscorrelations with cr templates
    nu_xcorrelations = 9, ChannelNuXcorrelations  #: dict of result of crosscorrelations with nu templates
    signal_time = 10, float  #: the time of the maximum amplitude of the envelope
    noise_rms = 11, float  #: the root mean square of the noise
    signal_regions = 12, List[int]  #: list of start and end times of regions that likely contain a signal
    noise_regions = 13, List[int]  #: list of start and end times of regions that likel do not contain any signals
    signal_time_offset = 14, float  #: the relative timing differences of the signal arrival times between channels
    signal_receiving_zenith = 15, float  #: the zenith angle of direction at which the radio signal arrived at the antenna
    signal_ray_type = (
        16,
        str,
    )  #: type of the ray propagation path of the signal received by this channel. Options are direct, reflected and refracted
    signal_receiving_azimuth = 17, float  #: the azimuth angle of direction at which the radio signal arrived at the antenna
    block_offsets = 18, np.ndarray  #: 'block' or pedestal offsets. See `NuRadioReco.modules.RNO_G.channelBlockOffsetFitter`
    Vrms_NuRadioMC_simulation = 19, float  #: the noise rms used in the MC simulation
    bandwidth_NuRadioMC_simulation = (
        20,
        float,
    )  #: the integrated channel response (=bandwidth for signal chains without amplification) used in the MC simulation
    Vrms_trigger_NuRadioMC_simulation = (
        21,
        float,
    )  #: the noise rms of the trigger channels (optional) used in the MC simulation
    root_power_ratio = 22, float  #: the root power ratio (float)
    impulsivity = 23, float  #: average of the CDF about the peak of the coherently summed waveform
    entropy = 24, float  #: Shannon entropy of a waveform
    kurtosis = 25, float  #: kurtosis of a waveform


class channelParametersRNOG(TypedEnum):
    # RNO-G specific channel parameters
    # FS: I did not start with a negative parameter on the 1, hence I chose 100
    glitch = 100, bool  #: True if channel is likely to have a glitch. See 'NuRadioReco.modules.RNO_G.channelGlitchDetector'
    glitch_test_statistic = (
        101,
        float,
    )  #: Numerical value delivered by the glitch detector. Positive values indicate a likely glitch.


class stationParametersRNOG(TypedEnum):
    # RNO-G specific station parameters
    coherent_snr = (
        1,
        float,
    )  #: Signal to Noise Ratio of the coherently summed waveform using the SNR definition of #63 avg_ch_snr
    coherent_impulsivity = 2, float  #: impulsivity of the coherently summed waveform
    coherent_entropy = 3, float  #: Shannon entropy of the coherently summed waveform
    coherent_kurtosis = 4, float  #: kurtosis of the coherently summed waveform


class electricFieldParameters(TypedEnum):
    ray_path_type = 1, str  #: the type of the ray tracing solution ('direct', 'refracted' or 'reflected')
    polarization_angle = (
        2,
        float,
    )  #: electric field polarization in onsky-coordinates. 0 corresponds to polarization in e_theta, 90deg is polarization in e_phi
    polarization_angle_expectation = (
        3,
        float,
    )  #: expected polarization based on shower geometry. Defined analogous to polarization_angle
    signal_energy_fluence = 4, float  #: Energy/area in the radio signal
    cr_spectrum_slope = (
        5,
        float,
    )  #: Slope of the radio signal's spectrum as reconstructed by the voltageToAnalyticEfieldConverter
    zenith = 7, float  #: zenith angle of the signal. Note that refraction at the air/ice boundary is not taken into account
    azimuth = 8, float  #: azimuth angle of the signal. Note that refraction at the air/ice boundary is not taken into account
    signal_time = 9, float
    nu_vertex_distance = 10, float  #: the distance along the ray path from the vertex to the channel
    nu_viewing_angle = 11, float  #: the angle between shower axis and launch vector
    max_amp_antenna = (
        12,
        Dict[int, float],
    )  #: the maximum amplitude of the signal after convolution with the antenna response pattern, dict with channelid as key
    max_amp_antenna_envelope = (
        13,
        Dict[int, float],
    )  #: the maximum amplitude of the signal envelope after convolution with the antenna response pattern, dict with channelid as key
    reflection_coefficient_theta = (
        14,
        complex,
    )  #: for reflected rays: the complex Fresnel reflection coefficient of the eTheta component
    reflection_coefficient_phi = (
        15,
        complex,
    )  #: for reflected rays: the complex Fresnel reflection coefficient of the ePhi component
    cr_spectrum_quadratic_term = (
        16,
        float,
    )  #: result of the second order correction to the spectrum fitted by the voltageToAnalyticEfieldConverter
    energy_fluence_ratios = 17, Dict[str, float]  #: Ratios of the energy fluences in different passbands
    nu_vertex_propagation_time = 18, float  #: the time it takes for the signal to propagate from the vertex to the channel
    raytracing_solution = 19, RaytracingSolution  #: the ray tracing solution (see `RaytracingSolution`)
    launch_vector = (
        20,
        np.ndarray,
    )  #: the launch vector of the ray from which this efield originates (only available for in-ice simulations)


class ARIANNAParameters(TypedEnum):  #: this class stores parameters specific to the ARIANNA data taking
    seq_start_time = 1, Any  #: the start time of a sequence
    seq_stop_time = 2, Any  #: the stop time of a sequence
    seq_num = 3, int  #: the sequence number of the current event
    comm_period = 4, Any  #: length of data taking window
    comm_duration = 5, Any  #: maximum diration of communication window
    trigger_thresholds = 6, Any  #: trigger thresholds converted to voltage
    l1_supression_value = 7, Any  #: This provieds the L1 supression value for given event
    internal_clock_time = 8, Any  #: time since last trigger with ms precision


class showerParameters(TypedEnum):
    zenith = 1, float  #: zenith angle of the shower axis pointing towards xmax
    azimuth = 2, float  #: azimuth angle of the shower axis pointing towards xmax
    core = 3, np.ndarray  #: position of the intersection between shower axis and an observer plane
    energy = 4, float  #: total energy of the primary particle, or shower energy for in-ice particle showers
    electromagnetic_energy = 5, float  #: energy of the electromagnetic shower component
    radiation_energy = 6, float  #: totally emitted radiation energy
    electromagnetic_radiation_energy = 7, float  #: radiation energy originated from the electromagnetic emission
    primary_particle = 8, int  #: particle id of the primary particle
    shower_maximum = 9, float  #: position of shower maximum in slant depth, e.g., Xmax
    distance_shower_maximum_geometric = 10, float  #: distance to xmax in meter
    distance_shower_maximum_grammage = 11, float  #: distance to xmax in g / cm^2
    parent_id = 12, int  #: id of parent in sim particles

    #: dedicated parameter for sim showers
    refractive_index_at_ground = 100, float  #: refractivity at sea level
    atmospheric_model = 101, int  #: atmospheric model used in simulation (CORSIKA ATMOD number)
    #: offset between magnetic field and north in reconstruction corrdinatesystem
    magnetic_field_rotation = 102, float
    magnetic_field_vector = 103, np.ndarray  #: magnetic field used in simulation in local coordinate system
    observation_level = 104, float  #: altitude a.s.l where the particles are stored

    charge_excess_profile_id = 105, int  #: the id of the charge-excess profile used in the ARZ Askaryan calculation
    type = 106, str  #: for neutrino induces showers in ice: can be "HAD" or "EM"
    vertex = 107, np.ndarray  #: the interaction vertex (for air showers this corresponds to the point of X0)
    vertex_time = 108, float  #: the propagation time relative to the first interactions
    interaction_type = 109, str  #: the interaction type, e.g. cc or nc
    k_L = (
        110,
        float,
    )  #: the k_L parameter of the Alvarez2009 parameter that controls the longitudional width of the charge excess profile
    flavor = 111, int  #: the flavor of the particle initiating the shower
    n_interaction = (
        112,
        int,
    )  #: Hierarchical counter for the number of showers per event group (also accounts for showers which did not trigger and might not be saved)

    interferometric_shower_maximum = 120, float  #: depth of the maximum of the longitudinal profile of the beam-formed signal
    interferometric_shower_axis = 121, np.ndarray  #: shower axis (direction) derived from beam-formed signal
    interferometric_core = (
        122,
        np.ndarray,
    )  #: core (intersection of shower axis with obs plane) derived from beam-formed signal


class emitterParameters(TypedEnum):
    position = 1, np.ndarray  #: the interaction vertex (for air showers this corresponds to the point of X0)
    model = 2, str  #: the emitter model used to simulate the emission (as defined in NuRadioMC/SignalGen/emitter.py)
    amplitude = 3, float  #: the amplitude of the signal
    polarization = 4, np.ndarray  #: the polarization of the signal
    half_width = 5, float  #: the width of square and tone_burst signal
    frequency = 6, float  #: the frequency of a signal (for cw and tone_burst model)
    orientation_phi = (
        7,
        float,
    )  #: the orientation of the emiting antenna, defined via two vectors that are defined with two angles each
    orientation_theta = (
        8,
        float,
    )  #: the orientation of the emiting antenna, defined via two vectors that are defined with two angles each
    rotation_phi = (
        9,
        float,
    )  #: the orientation of the emiting antenna, defined via two vectors that are defined with two angles each
    rotation_theta = (
        10,
        float,
    )  #: the orientation of the emiting antenna, defined via two vectors that are defined with two angles each
    realization_id = 11, int  #: the id of the measurement of the emitted electric field
    antenna_type = 12, str  #: the type of the antenna used to simulate the emission
    time = 13, float  #: the time when the signal was emitted


class particleParameters(TypedEnum):
    parent_id = 1, int  #: the entry number of the parent particle, None if primary.
    zenith = 2, float  #: the zenith angle of the incoming neutrino direction
    azimuth = 3, float  #: the azimuth angle of the incoming neutrino direction
    energy = 4, float  #: the energy of the neutrino
    flavor = 5, int  #: the flavor of the neutrino, more generally the PDG code
    vertex = 6, np.ndarray  #: the neutrino vertex position (x,y,z)
    vertex_time = 9, float
    weight = 10, float
    inelasticity = 11, float  #: inelasticity ot neutrino interaction
    interaction_type = 12, str  #: interaction type, e.g., cc, nc
    n_interaction = 13, int  #: number of interaction
    shower_id = (
        14,
        int,
    )  #: the shower id associated with this particle. This is needed to generate HDF5 files that contain the primary particle

    cr_energy = 101, float  #: the cosmic-ray energy
    cr_zenith = 102, float  #: zenith angle of the cosmic-ray incoming direction
    cr_azimuth = 103, float  #: azimuth angle of the cosmic-ray incoming direction
    cr_energy_em = (
        104,
        float,
    )  #: the electromagnetic shower energy (the cosmic ray energy that ends up in electrons, positrons and gammas)


class generatorAttributes(TypedEnum):
    Emax = 1, float  #: maximum simulated energy
    Emin = 2, float  #: minimum simulated energy

    deposited = 3, bool  #: deposited energies or neutrino energies?

    fiducial_rmin = 4, float  #: fiducial volume parameter (if cylindrical footprint used)
    fiducial_rmax = 5, float  #: fiducial volume parameter (if cylindrical footprint used)

    fiducial_xmin = 6, float  #: fiducial volume parameter (if rectangular footprint used)
    fiducial_xmax = 7, float  #: fiducial volume parameter (if rectangular footprint used)
    fiducial_ymin = 8, float  #: fiducial volume parameter (if rectangular footprint used)
    fiducial_ymax = 9, float  #: fiducial volume parameter (if rectangular footprint used)

    fiducial_zmin = 10, float
    fiducial_zmax = 11, float

    rmin = 12, float  #: volume parameter (if cylindrical)
    rmax = 13, float  #: volume parameter (if cylindrical)

    xmin = 14, float  #: volume parameter (if rectangular)
    xmax = 15, float  #: volume parameter (if rectangular)
    ymin = 16, float  #: volume parameter (if rectangular)
    ymax = 17, float  #: volume parameter (if rectangular)

    zmin = 18, float
    zmax = 19, float

    # volume calculated from the (z r) min max or (x y z) min max parameters
    volume = 20, float
    area = 21, float

    phimax = 22, float  #: simulated space angle range
    phimin = 23, float  #: simulated space angle range
    thetamax = 24, float  #: simulated space angle range
    thetamin = 25, float  #: simulated space angle range

    flavors = 26, List[int]  #: list of simulated event flavours (PDG codes)
    dt = 27, float  #: inverse of sampling rate used in the simulation
    Tnoise = 28, float  #: noise temperature used in the simulation (None if taken from the detector description)
    Vrms = 29, float  #: noise rms used in the simulation (of the first channel)
    bandwidth = 30, float  #: integrated channel response used in the simulation (of the first channel)

    # simulated statistics
    n_events = 100, int
    n_samples = 101, int
    start_event_id = 102, int
    total_number_of_events = 103, int

    # version numbers
    NuRadioMC_EvtGen_version = 200, str
    NuRadioMC_EvtGen_version_hash = 201, str
    NuRadioMC_version = 202, str
    NuRadioMC_version_hash = 203, str


class eventParameters(TypedEnum):
    sim_config = 1, dict  #: contents of the config file that the NuRadioMC simulation was run with
    hash_NuRadioReco = 2, str  #: deprecated, since NuRadioReco is no longer its own repository
    hash_NuRadioMC = 3, str  #: git hash of the NuRadioMC commit that the file was created with


# Values that are either one value for all channels (beams) or a dict per channel (beam) id
_PER_CHANNEL_FLOAT = OneOf(value=float, per_channel=Dict[int, float])
_DELAYS = OneOf(per_beam=Dict[int, Dict[int, float]], per_channel=Dict[int, float])


class triggerParameters(TypedEnum):
    """
    Parameters of a `NuRadioReco.framework.trigger.Trigger` (all except its name and type).

    Thresholds are given in the observable of the trigger type (e.g. V, V^2 ns, ADC counts). Members are
    grouped by the trigger types that use them; each group leaves room for new members.
    """

    # common to all triggers
    triggered = 1, bool  #: flag if the trigger condition was fulfilled
    trigger_time = 2, float  #: first time the trigger condition was fulfilled, relative to the station time (or None)
    #: all times the trigger condition was fulfilled (phased triggers: dict beam -> array of times)
    trigger_times = 3, OneOf(value=np.ndarray, per_beam=Dict[int, np.ndarray])
    triggered_channels = 4, List[int]  #: ids of the channels that fulfilled the trigger condition
    channels = 5, List[int]  #: ids of the channels involved in the trigger (None: all channels)
    primary = 6, bool  #: flag if this is the primary trigger, which defines the readout window
    #: time between the start of the readout window and the trigger time (float: same for all channels, or dict
    #: channel id -> time). Used in simulations to cut out the readout window around the trigger time;
    #: in data (set by the data reader) to set the `trace_start_time` for a given trigger time.
    pre_trigger_times = 7, _PER_CHANNEL_FLOAT

    # readout-window jitter (currently only set in simulations)
    sample_block_size = 100, int  #: range of the uniform readout-window jitter (in samples)
    gaussian_jitter = 101, float  #: standard deviation of the Gaussian readout-window jitter (in samples)

    # threshold triggers: simple threshold, power integration, envelope, RNO-G surface (threshold also: phased triggers)
    threshold = 200, _PER_CHANNEL_FLOAT  #: threshold (float: same for all channels, or dict channel id -> threshold)
    number_of_coincidences = 201, int  #: number of channels that have to fulfill the trigger condition (also: high/low)
    coinc_window = 202, float  #: coincidence window between the channels (also: high/low)

    # high/low trigger
    threshold_high = 300, _PER_CHANNEL_FLOAT  #: high threshold (float: all channels, or dict channel id -> threshold)
    threshold_low = 301, _PER_CHANNEL_FLOAT  #: low threshold (float: all channels, or dict channel id -> threshold)
    high_low_window = 302, float  #: coincidence window between a high and a low threshold crossing in one channel

    # power integration trigger (power_mean, power_std also: analog envelope phased trigger)
    integration_window = 400, float  #: integration window
    power_mean = 401, float  #: mean of the noise power (after the diode)
    power_std = 402, float  #: standard deviation of the noise power (after the diode)

    # envelope trigger
    passband = 500, List[float]  #: passband (low, high) of the filter
    filter_order = 501, int  #: order of the filter

    # phased triggers (simple phased, digital envelope phased; phasing_angles, trigger_delays also: analog envelope phased)
    phasing_angles = 600, np.ndarray  #: phasing angle of each beam
    #: delay per beam and channel id: dict beam -> dict channel id -> delay (beams that triggered);
    #: analog envelope phased trigger (one beam): dict channel id -> delay
    trigger_delays = 601, _DELAYS
    maximum_amps = 602, np.ndarray  #: maximum of the beamformed signal per beam

    # simple phased trigger
    window_size = 700, int  #: integration window (in ADC samples)
    step_size = 701, int  #: stride between integration windows (in ADC samples)
    secondary_channels = 702, List[int]  #: channels of the secondary phased beam (not set by any module)
    secondary_angles = 703, np.ndarray  #: phasing angles of the secondary beam (not set by any module)
    sec_trigger_delays = 704, _DELAYS  #: as `trigger_delays` for the secondary beam (not set by any module)

    # analog envelope phased trigger
    threshold_factor = 800, float  #: threshold in units of the noise power std
    output_passband = 801, List[float]  #: passband (low, high) applied after the diode; entries can be None

    # experiment-specific triggers
    # RNO-G surface trigger
    temperature = 2000, float  #: temperature of the trigger board
    bias_voltage = 2001, float  #: bias voltage of the trigger board
