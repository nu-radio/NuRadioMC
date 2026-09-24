"""Synthetic events with known source positions for the reconstruction tests.

Every trace is an impulse placed at the travel time the reconstruction's own tables
predict for the chosen source, plus white noise, both passed through the
preprocessing bandpass of the reference configuration (100-700 MHz, 10th-order Butterworth, the same transfer function
`channelBandPassFilter` applies to data) at the native 3.2 GHz over 2048 samples, then
resampled to 10 GHz exactly as the reconstruction driver does. The dispersion of that
filter gives the pulses a realistic width, so the correlation landscape has the same
scale as for real events. The reconstruction is therefore tested against the exact
geometry it searches, with no simulation in the loop: travel times come from the same
loader (`_load_rz_interpolator`) and the same per-channel geometry (`_get_ant_locs`,
horizontal distance clipped at 1 m) that the reconstruction uses at run time.
"""

import os

import numpy as np

from NuRadioReco.framework.channel import Channel
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D
from NuRadioReco.utilities import signal_processing, units

VPOL_CHANNELS = [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]
HPOL_CHANNELS = [4, 8, 11, 21]
NATIVE_RATE = 3.2 * units.GHz
N_NATIVE = 2048
SAMPLING_RATE = 10 * units.GHz
BANDPASS = [0.1 * units.GHz, 0.7 * units.GHz]
BANDPASS_ORDER = 10


def antenna_locations(det, station_id):
    """Return channel -> [x_rel, y_rel, z_abs] using the reconstruction's own helper."""
    return InterferometricReco3D._get_ant_locs(station_id, det)


def pa_center(ant_locs):
    """Return the phased-array reference point the reconstruction centres its grid on."""
    return (ant_locs[1] + ant_locs[2]) / 2.0


def cylindrical_to_enu(rho, phi_deg, z_abs, pa):
    """Convert reconstruction-frame (rho, phi, z) to the source ENU vector.

    Args:
        rho: Horizontal distance from the PA reference point in metres.
        phi_deg: Azimuth in degrees, counter-clockwise from east.
        z_abs: Absolute z in metres (ice surface at 0).
        pa: PA reference point from `pa_center`.

    Returns:
        np.ndarray of shape (3,) in the station-relative horizontal frame with absolute z.
    """
    phi = np.radians(phi_deg)
    return np.array([rho * np.cos(phi) + pa[0], rho * np.sin(phi) + pa[1], z_abs])


def enu_to_cylindrical(src, pa):
    """Inverse of `cylindrical_to_enu`; returns (rho, phi_deg in [0, 360), z_abs)."""
    dx, dy = src[0] - pa[0], src[1] - pa[1]
    return float(np.hypot(dx, dy)), float(np.degrees(np.arctan2(dy, dx)) % 360.0), float(src[2])


def angular_separation(reco, truth, pa):
    """Great-circle angle in degrees between two (rho, phi_deg, z_abs) directions seen from the PA."""
    vecs = []
    for rho, phi, z in (reco, truth):
        p = np.radians(phi)
        vecs.append(np.array([rho * np.cos(p), rho * np.sin(p), z - pa[2]]))
    a, b = vecs
    cos_ang = np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


class TravelTimeTables:
    """Per-channel travel-time lookup using the reconstruction's loader and NaN filling."""

    def __init__(self, table_dir, station_id, channels,
                 pattern='st{station_id}_ch{ch}_rz_table.npz'):
        """Load one table per channel.

        Args:
            table_dir: Root directory holding `station{N}/` subdirectories.
            station_id: Station number.
            channels: Channels to load.
            pattern: File name pattern; the default is the combined (min over ray types) table.
        """
        self.tables = {}
        for ch in channels:
            path = os.path.join(table_dir, f'station{station_id}',
                                pattern.format(station_id=station_id, ch=ch))
            self.tables[ch] = InterferometricReco3D._load_rz_interpolator(path, 'linear')

    def travel_time(self, ch, src_enu, ant_loc):
        """Travel time in ns from a source ENU point to one antenna."""
        r = max(np.hypot(src_enu[0] - ant_loc[0], src_enu[1] - ant_loc[1]), 1.0)
        return float(self.tables[ch].interp((r, src_enu[2])))


def _bandpass_response(n_samples, sampling_rate):
    """Complex transfer function of the reference bandpass on the rfft grid of a trace."""
    freqs = np.fft.rfftfreq(n_samples, 1.0 / sampling_rate)
    return freqs, signal_processing.get_filter_response(freqs, BANDPASS, 'butter', BANDPASS_ORDER)


def filtered_trace(t_pulse, amplitude, noise_sigma, rng, n_samples=N_NATIVE,
                   sampling_rate=NATIVE_RATE):
    """Impulse at `t_pulse` plus white noise, both through the reference bandpass.

    Args:
        t_pulse: Arrival time of the impulse in ns (sub-sample precision).
        amplitude: Peak amplitude of the filtered pulse (after the bandpass).
        noise_sigma: Standard deviation of the filtered noise (after the bandpass).
        rng: numpy random generator.

    Returns:
        Trace at the native sampling rate.
    """
    freqs, response = _bandpass_response(n_samples, sampling_rate)
    impulse_spec = np.exp(-2j * np.pi * freqs * t_pulse) * response
    pulse = np.fft.irfft(impulse_spec, n_samples)
    pulse *= amplitude / np.max(np.abs(pulse))
    white = rng.normal(0.0, 1.0, n_samples)
    noise = np.fft.irfft(np.fft.rfft(white) * response, n_samples)
    if noise_sigma > 0:
        noise *= noise_sigma / noise.std()
    else:
        noise[:] = 0.0
    return pulse + noise


def make_event(det, station_id, src_enu, channels, tables, snr=10.0, seed=0,
               t0=150.0 * units.ns, amplitude=1.0, trace_start_time=0.0,
               channel_order=None):
    """Build a NuRadioReco event whose channel traces carry a pulse at the table travel times.

    Args:
        det: Detector description (only positions are used).
        station_id: Station number.
        src_enu: Source position from `cylindrical_to_enu`.
        channels: Channels to fill.
        tables: `TravelTimeTables` for the same station and channels.
        snr: Filtered pulse peak amplitude over filtered noise standard deviation.
        seed: Seed for the noise generator; also used as the event id.
        t0: Time of the earliest pulse in the trace.
        amplitude: Pulse amplitude in arbitrary units.
        trace_start_time: Common start time of every trace (a global time shift).
        channel_order: Optional order in which channels are added to the station.

    Returns:
        (event, station, travel_times) where travel_times maps channel -> ns.

    Raises:
        ValueError: if any channel has no table solution at the source.
    """
    rng = np.random.default_rng(seed)
    ant_locs = antenna_locations(det, station_id)
    tts = {ch: tables.travel_time(ch, src_enu, ant_locs[ch]) for ch in channels}
    if not all(np.isfinite(t) for t in tts.values()):
        missing = [ch for ch, t in tts.items() if not np.isfinite(t)]
        raise ValueError(f"no table solution for channels {missing} at {src_enu}")
    t_ref = min(tts.values())
    evt = Event(0, seed)
    stn = Station(station_id)
    for ch in (channel_order or channels):
        trace = filtered_trace(t0 + tts[ch] - t_ref, amplitude, amplitude / snr, rng)
        c = Channel(ch)
        c.set_trace(trace, NATIVE_RATE, trace_start_time=trace_start_time)
        c.resample(SAMPLING_RATE)
        stn.add_channel(c)
    evt.set_station(stn)
    return evt, stn, tts


def make_noise_event(station_id, channels, seed=0):
    """Build an event of pure Gaussian noise (no pulse) for false-positive checks."""
    rng = np.random.default_rng(seed)
    evt = Event(0, seed)
    stn = Station(station_id)
    for ch in channels:
        c = Channel(ch)
        c.set_trace(filtered_trace(0.0, 0.0, 1.0, rng), NATIVE_RATE)
        c.resample(SAMPLING_RATE)
        stn.add_channel(c)
    evt.set_station(stn)
    return evt, stn
