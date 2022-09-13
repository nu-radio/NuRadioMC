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

def calculate_thermal_noise_Vrms(T_noise, T_noise_max_freq, T_noise_min_freq):
    """ calculates thermal noise amplitude for a given temperature and frequency range.
    Parameters
    ----------
    T_noise: float
        temperature in Kelvin
    T_noise_max_freq: float
        max frequency in GHz
    T_noise_min_freq: float
        min frequency in GHz

    Returns
    -------
    amplitude of thermal noise
    """

    Vrms_thermal_noise = (((scipy.constants.Boltzmann * units.joule / units.kelvin) * T_noise *
                           (T_noise_max_freq - T_noise_min_freq) * 50 * units.ohm) ** 0.5)

    return Vrms_thermal_noise


def set_random_station_time(station, station_time='2019-01-01T00:00:00'):
    '''this function sets a random hour for the set date
     Parameters
    ----------
    station: generator_object
        station object
    station_time: iso
        station time and date

    Returns
    -------
    generator_object
    '''
    import datetime
    date = datetime.datetime.date(datetime.datetime.strptime(station_time, "%Y-%m-%dT%H:%M:%S"))
    random_generator_hour = np.random.RandomState()
    hour = random_generator_hour.randint(0, 24)
    if hour < 10:
        station.set_station_time(astropy.time.Time('{}T0{}:00:00'.format(date, hour)))
    elif hour >= 10:
        station.set_station_time(astropy.time.Time('{}T{}:00:00'.format(date, hour)))

    return station


def create_empty_event(det, trace_samples=1024, station_time='2019-01-01T00:00:00', station_time_random=True,
                       sampling_rate=1 * units.GHz):
    '''create an empty event for a given detector
    Parameters
    ----------
    det: generator_object
        detector object
    trace_samples: int
        trace length
    station_time: iso
        time and date in the station
    station_time_random: bool
        set time random: True or False
    sampling_rate: float

    Returns
    -------
    event, station, channel
    '''

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
    '''creates an trace of zeros for a certain channel
    Parameters
    ----------
    station: generator_object
        station object
    trace_samples: int
        trace length

    sampling_rate: float

    Returns
    -------
    channel
    '''
    for channel in station.iter_channels():
        default_trace = np.zeros(trace_samples)
        channel.set_trace(trace=default_trace, sampling_rate=sampling_rate)

    return channel


def add_random_phase(station, sampling_rate=1 * units.GHz):
    '''changes the signal phase in a random uniform distribution in the range 0 to 2pi Parameters
    ----------
    station: generator_object
        station object

    sampling_rate: float

    Returns
    -------
    channel
    '''
    for channel in station.iter_channels():
        freq_specs = channel.get_frequency_spectrum()
        rand_phase = np.random.uniform(low=0, high=2 * np.pi, size=len(freq_specs))
        freq_specs = np.abs(freq_specs) * np.exp(1j * rand_phase)
        channel.set_frequency_spectrum(frequency_spectrum=freq_specs, sampling_rate=sampling_rate)

    return channel


def get_global_trigger_rate(single_rate, n_channels, number_coincidences, coinc_window):
    '''calculates the global trigger rate with a given trigger rate from one channel including
     number of channels with trigger (n_channels), number_coincidences and the time coinc_window
     of the coincidence (coinc_window)

     Parameters
    ----------
    single_rate: float or array
        trigger rate of a single antenna
    n_channels: int
        number of channels with trigger
    number_coincidences: int
        number of coincidences with antennas
    coinc_window: float
        time window of coincidence in ns


    Returns
    -------
    global trigger rate of all trigger channels combined
    '''

    return number_coincidences * scipy.special.comb(n_channels, number_coincidences) \
           * single_rate ** number_coincidences * coinc_window ** (number_coincidences - 1)


def get_single_channel_trigger_rate(global_rate, n_channels, number_coincidences, coinc_window):
    '''calculates the trigger rate of a singel channel for a global trigger rate including
     number of channels with trigger (n_channels), number_coincidences and the time coinc_window
     of the coincidence (coinc_window)

     Parameters
        ----------
    global_rate: float or array
        trigger rate of a all antenna together
    n_channels: int
        number of channels with trigger
    number_coincidences: int
        number of coincidences with antennas
    coinc_window: float
        time window of coincidence in ns

    Returns
    -------
    single trigger rate of one channel
    '''
    r_single_pow_n_coincidences = global_rate / (
            number_coincidences * scipy.special.comb(n_channels, number_coincidences) *
            coinc_window ** (number_coincidences - 1))
    r_single = r_single_pow_n_coincidences ** (1 / float(number_coincidences))
    return r_single
