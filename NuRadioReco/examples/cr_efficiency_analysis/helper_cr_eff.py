import scipy.constants
import numpy as np
import json
import scipy.signal
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.channel import Channel
from NuRadioReco.utilities import units
import astropy


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def calculate_thermal_noise_Vrms(T_noise, T_noise_max_freq, T_noise_min_freq):
    Vrms_thermal_noise = (((scipy.constants.Boltzmann * units.joule / units.kelvin) * T_noise *
                           (T_noise_max_freq - T_noise_min_freq) * 50 * units.ohm) ** 0.5)

    return Vrms_thermal_noise


def set_random_station_time(station, station_time):
    '''this function gives a random hour for the choosen date'''
    import datetime
    date = datetime.datetime.date(datetime.datetime.strptime(station_time, "%Y-%m-%dT%H:%M:%S"))
    random_generator_hour = np.random.RandomState()
    hour = random_generator_hour.randint(0, 24)
    if hour < 10:
        station.set_station_time(astropy.time.Time('{}T0{}:00:00'.format(date, hour)))
    elif hour >= 10:
        station.set_station_time(astropy.time.Time('{}T{}:00:00'.format(date, hour)))

    return station


def create_empty_event(det, trace_samples, station_time='2019-01-01T00:00:00', station_time_random=True,
                       sampling_rate=1 * units.GHz):
    import datetime
    station_ids = det.get_station_ids()
    station_id = station_ids[0]
    channel_ids = det.get_channel_ids(station_id)
    event = Event(0, 0)
    station = Station(station_id)
    event.set_station(station)

    station.set_station_time(astropy.time.Time(station_time))

    if station_time_random:
        set_random_station_time(station, station_time)

    for channel_id in channel_ids:  # take some channel id that match your detector
        channel = Channel(channel_id)
        default_trace = np.zeros(trace_samples)
        channel.set_trace(trace=default_trace, sampling_rate=sampling_rate)
        station.add_channel(channel)

    return event, station, channel


def create_empty_channel_trace(station, trace_samples, sampling_rate=1 * units.GHz):
    for channel in station.iter_channels():
        default_trace = np.zeros(trace_samples)
        channel.set_trace(trace=default_trace, sampling_rate=sampling_rate)

    return channel


def add_random_phase(station, sampling_rate=1 * units.GHz):
    for channel in station.iter_channels():
        freq_specs = channel.get_frequency_spectrum()
        rand_phase = np.random.uniform(low=0, high=2 * np.pi, size=len(freq_specs))
        freq_specs = np.abs(freq_specs) * np.exp(1j * rand_phase)
        channel.set_frequency_spectrum(frequency_spectrum=freq_specs, sampling_rate=sampling_rate)

    return channel


def get_auger_cr_flux(energy):
    import scipy.interpolate as interpolate
    data = np.loadtxt('Auger_combined spectrum_ICRC_2019.txt', skiprows=3)
    E = 10 ** (data[:, 0]) * units.eV
    E_J = data[:, 1] * units.m ** -2 * units.second ** -1 * units.steradian ** -1
    J = E_J / E
    Err_up = data[:, 2] * units.m ** -2 * units.second ** -1 * units.steradian ** -1 / E
    Err_low = data[:, 3] * units.m ** -2 * units.second ** -1 * units.steradian ** -1 / E

    print(J)
    get_flux = interpolate.interp1d(E, J, fill_value=0, bounds_error=False)

    return get_flux(energy)


def get_auger_flux_per_energy_bin(bin_edge_low, bin_edge_high):
    import scipy.interpolate as interpolate
    from scipy.integrate import quad

    data = np.loadtxt('Auger_combined spectrum_ICRC_2019.txt', skiprows=3)
    E = 10 ** (data[:, 0]) * units.eV
    E_J = data[:, 1] * units.m ** -2 * units.second ** -1 * units.steradian ** -1
    J = E_J / E

    flux = interpolate.interp1d(E, J, fill_value=0, bounds_error=False)
    int_flux = quad(flux, bin_edge_low, bin_edge_high, limit=2 * E.shape[0], points=E)

    return int_flux[0]


def get_global_trigger_rate(single_rate, n_channels, number_coincidences, coinc_window):
    return number_coincidences * scipy.special.comb(n_channels, number_coincidences) \
           * single_rate ** number_coincidences * coinc_window ** (number_coincidences - 1)


def get_single_channel_trigger_rate(global_rate, n_channels, number_coincidences, coinc_window):
    r_single_pow_n_coincidences = global_rate / (
            number_coincidences * scipy.special.comb(n_channels, number_coincidences) *
            coinc_window ** (number_coincidences - 1))
    r_single = r_single_pow_n_coincidences ** (1 / float(number_coincidences))
    return r_single
