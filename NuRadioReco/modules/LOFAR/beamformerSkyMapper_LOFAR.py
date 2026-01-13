"""
Full-Sky Beamformer Sky Mapper for LOFAR CR Direction Finding
==============================================================

This module performs full-sky interferometric imaging via frequency-domain 
beamforming to find cosmic ray arrival directions. It produces sky maps,
identifies peak signal directions, and optionally generates time-windowed
animations to distinguish transient CR signals from continuous sources.

Works per-station (assumes plane wave approximation valid within single station).

.. moduleauthor:: Karen Terveer
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

import radiotools.helper as hp
from NuRadioReco.utilities import units, fft
from NuRadioReco.framework.parameters import stationParameters, channelParameters, showerParameters
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.LOFAR.beamforming_utilities import geometric_delay_far_field, beamformer, lightspeed


class beamformerSkyMapper:
    """
    Full-sky beamforming for cosmic ray direction determination.
    
    Computes sky power maps via frequency-domain beamforming and identifies
    the strongest signal direction as the CR arrival direction. Supports
    time-windowed analysis to detect transient signals.
    
    This module is designed to be called per-station, as the plane wave 
    approximation is only valid within a single LOFAR station.
    
    Example
    -------
    >>> mapper = beamformerSkyMapper()
    >>> mapper.begin(n_zenith=45, n_azimuth=180, debug=True)
    >>> for station in event.get_stations():
    ...     mapper.run(event, station, detector)
    >>> mapper.end()
    """
    
    def __init__(self):
        self.logger = logging.getLogger("NuRadioReco.beamformerSkyMapper")
        
        # Configuration parameters
        self.__n_zenith = None
        self.__n_azimuth = None
        self.__n_time_windows = None
        self.__window_overlap = None
        self.__cr_snr = None
        self.__debug = None
        self.__output_dir = None
        self.__freq_range = None
        self.__use_efields = None
        self.__mark_sun = None
    
    def begin(self, n_zenith=45, n_azimuth=180, 
              n_time_windows=10, window_overlap=0.5,
              cr_snr=3.0, freq_range_mhz=(30, 80),
              use_efields=False, mark_sun=True,
              debug=False, output_dir=None,
              logger_level=logging.WARNING):
        """
        Configure the beamformer sky mapper.
        
        Parameters
        ----------
        n_zenith : int, default=45
            Number of zenith angle bins (0 to 90 deg)
        n_azimuth : int, default=180
            Number of azimuth bins (0 to 360 deg)
        n_time_windows : int, default=10
            Number of time windows for transient analysis
        window_overlap : float, default=0.5
            Overlap fraction between time windows
        cr_snr : float, default=3.0
            Minimum SNR for a channel to be included
        freq_range_mhz : tuple, default=(30, 80)
            Frequency range in MHz to use for beamforming
        use_efields : bool, default=False
            If True, use E-field traces. If False, use raw voltages.
        mark_sun : bool, default=True
            Mark Sun position on sky maps (requires astropy)
        debug : bool, default=False
            Generate debug sky map plots and GIF animations
        output_dir : str, optional
            Directory for debug output (default: current directory)
        logger_level : int, default=logging.WARNING
            Logging level
        """
        self.__n_zenith = n_zenith
        self.__n_azimuth = n_azimuth
        self.__n_time_windows = n_time_windows
        self.__window_overlap = window_overlap
        self.__cr_snr = cr_snr
        self.__freq_range = (freq_range_mhz[0] * units.MHz, freq_range_mhz[1] * units.MHz)
        self.__use_efields = use_efields
        self.__mark_sun = mark_sun
        self.__debug = debug
        self.__output_dir = output_dir or "."
        
        self.logger.setLevel(logger_level)
    
    def _collect_traces(self, station, detector):
        """
        Collect traces and positions from station.
        
        Returns
        -------
        fft_traces : np.ndarray
            FFT of traces, shape (n_antennas, n_freq)
        frequencies : np.ndarray
            Frequency values in internal units
        positions : np.ndarray
            Antenna positions, shape (n_antennas, 3)
        sampling_rate : float
            Sampling rate
        n_samples : int
            Number of time samples
        """
        if self.__use_efields:
            # Use E-field traces
            electric_fields = station.get_electric_fields()
            if len(electric_fields) == 0:
                return None, None, None, None, None
            
            # Determine dominant polarisation
            random_traces = np.random.choice(electric_fields, 
                                              size=min(5, len(electric_fields)), 
                                              replace=False)
            dominant_pol_traces = []
            for field in random_traces:
                trace_envelope = np.abs(hilbert(field.get_trace(), axis=0))
                dominant_pol_traces.append(np.argmax(np.max(trace_envelope, axis=1)))
            dominant_pol = np.argmax(np.bincount(dominant_pol_traces))
            
            traces = []
            positions = []
            for field in electric_fields:
                trace = field.get_trace()[dominant_pol].astype(np.float64)
                trace -= np.mean(trace)
                traces.append(trace)
                positions.append(field.get_position())
            
            traces = np.array(traces)
            positions = np.array(positions)
            
            # Get sampling rate from first field
            times = electric_fields[0].get_times()
            sampling_rate = 1.0 / ((times[1] - times[0]) * units.ns)
            
        else:
            # Use raw voltage traces
            channel_ids = station.get_channel_ids()
            if len(channel_ids) == 0:
                return None, None, None, None, None
            
            # Group by group_id
            groups = {}
            for ch_id in channel_ids:
                channel = station.get_channel(ch_id)
                group_id = channel.get_group_id()
                if group_id not in groups:
                    groups[group_id] = []
                groups[group_id].append(ch_id)
            
            traces = []
            positions = []
            sampling_rate = None
            
            for group_id, ch_ids in groups.items():
                trace_sum = None
                pos = None
                
                for ch_id in ch_ids:
                    channel = station.get_channel(ch_id)
                    trace = channel.get_trace().astype(np.float64)
                    trace -= np.mean(trace)
                    
                    if sampling_rate is None:
                        sampling_rate = channel.get_sampling_rate()
                    
                    if pos is None:
                        pos = detector.get_absolute_position(station.get_id()) + \
                              detector.get_relative_position(station.get_id(), ch_id)
                    
                    if trace_sum is None:
                        trace_sum = trace
                    else:
                        trace_sum = trace_sum + trace
                
                traces.append(trace_sum)
                positions.append(pos)
            
            traces = np.array(traces)
            positions = np.array(positions)
        
        n_antennas, n_samples = traces.shape
        
        # Compute FFT
        fft_traces = np.fft.rfft(traces, axis=1)
        frequencies = np.fft.rfftfreq(n_samples, 1.0 / sampling_rate)
        
        return fft_traces, frequencies, positions, sampling_rate, n_samples
    
    def _beamform_direction(self, fft_data, frequencies, positions, zenith, azimuth):
        """
        Beamform towards a specific direction.
        
        Returns
        -------
        power : float
            Total power in the beamformed signal
        """
        direction = hp.spherical_to_cartesian(zenith, azimuth)
        delays = geometric_delay_far_field(positions, direction)
        beamformed_spectrum = beamformer(fft_data, frequencies, delays)
        
        # Apply frequency mask
        freq_mask = (frequencies >= self.__freq_range[0]) & (frequencies <= self.__freq_range[1])
        power = np.sum(np.abs(beamformed_spectrum[freq_mask])**2)
        
        return power
    
    def _compute_sky_image(self, fft_data, frequencies, positions):
        """
        Compute full-sky beamformed power map.
        
        Returns
        -------
        zenith_grid : np.ndarray
            Zenith angles in radians
        azimuth_grid : np.ndarray
            Azimuth angles in radians
        power_map : np.ndarray
            Power at each (zenith, azimuth)
        """
        zenith_grid = np.linspace(0, np.pi/2, self.__n_zenith)
        azimuth_grid = np.linspace(0, 2*np.pi, self.__n_azimuth, endpoint=False)
        
        power_map = np.zeros((self.__n_zenith, self.__n_azimuth))
        
        for i, zen in enumerate(zenith_grid):
            for j, az in enumerate(azimuth_grid):
                power_map[i, j] = self._beamform_direction(
                    fft_data, frequencies, positions, zen, az
                )
        
        return zenith_grid, azimuth_grid, power_map
    
    def _compute_time_windowed_images(self, traces, frequencies, positions, sampling_rate):
        """
        Compute sky maps for multiple time windows.
        
        Returns
        -------
        window_maps : list of (zenith_grid, azimuth_grid, power_map)
        window_times : list of (start_time, end_time) in seconds
        """
        n_samples = traces.shape[1]
        window_size = n_samples // (1 + (self.__n_time_windows - 1) * (1 - self.__window_overlap))
        window_size = max(64, int(window_size))
        hop = int(window_size * (1 - self.__window_overlap))
        
        window_maps = []
        window_times = []
        
        for w in range(self.__n_time_windows):
            start = w * hop
            end = start + window_size
            
            if end > n_samples:
                break
            
            # FFT of windowed trace
            windowed_traces = traces[:, start:end]
            fft_windowed = np.fft.rfft(windowed_traces, axis=1)
            freq_windowed = np.fft.rfftfreq(window_size, 1.0 / sampling_rate)
            
            zen_grid, az_grid, power_map = self._compute_sky_image(
                fft_windowed, freq_windowed, positions
            )
            
            window_maps.append((zen_grid, az_grid, power_map))
            window_times.append((start / sampling_rate, end / sampling_rate))
        
        return window_maps, window_times
    
    def _find_peak_direction(self, zenith_grid, azimuth_grid, power_map):
        """
        Find the direction of maximum power.
        
        Returns
        -------
        peak_zenith : float
            Zenith angle of peak in radians
        peak_azimuth : float
            Azimuth of peak in radians
        peak_power : float
            Power at peak
        """
        max_idx = np.unravel_index(np.argmax(power_map), power_map.shape)
        peak_zenith = zenith_grid[max_idx[0]]
        peak_azimuth = azimuth_grid[max_idx[1]]
        peak_power = power_map[max_idx]
        
        return peak_zenith, peak_azimuth, peak_power
    
    def _get_sun_position(self, event):
        """
        Get Sun position in NuRadioReco coordinates if available.
        
        Returns
        -------
        sun_zenith, sun_azimuth : float or None
        """
        try:
            from astropy.coordinates import get_sun, AltAz, EarthLocation
            from astropy.time import Time
            import astropy.units as u
            
            # Get event time
            event_id = event.get_id()
            from NuRadioReco.modules.io.LOFAR.readLOFARData import LOFAR_event_id_to_unix
            event_time_unix = LOFAR_event_id_to_unix(int(event_id))
            obs_time = Time(event_time_unix, format='unix')
            
            # LOFAR core location
            lofar_core = EarthLocation(lat=52.9053*u.deg, lon=6.8680*u.deg, height=0*u.m)
            
            sun_altaz = get_sun(obs_time).transform_to(
                AltAz(obstime=obs_time, location=lofar_core)
            )
            
            # Convert to NuRadioReco convention
            sun_zenith = np.pi/2 - sun_altaz.alt.rad
            sun_azimuth = np.pi/2 - sun_altaz.az.rad
            
            return sun_zenith, sun_azimuth
            
        except Exception as e:
            self.logger.debug(f"Could not get Sun position: {e}")
            return None, None
    

    def _check_sun_coincidence(self, peak_zenith, peak_azimuth, sun_zenith, sun_azimuth, 
                                threshold_deg=10.0):
        """
        Check if peak direction coincides with Sun position.
        
        Returns
        -------
        coincides : bool
            True if angular separation < threshold
        separation_deg : float
            Angular separation in degrees
        """
        if sun_zenith is None or sun_azimuth is None:
            return False, None
        
        # Calculate angular separation (great circle distance)
        dz = peak_zenith - sun_zenith
        da = peak_azimuth - sun_azimuth
        separation = np.sqrt(dz**2 + (np.sin(peak_zenith) * da)**2)
        separation_deg = np.rad2deg(separation)
        
        coincides = separation_deg < threshold_deg
        return coincides, separation_deg

    def _plot_sky_map(self, zenith_grid, azimuth_grid, power_map, 
                      peak_zenith, peak_azimuth,
                      sun_zenith=None, sun_azimuth=None,
                      title='Sky Map', save_path=None):
        """
        Create polar sky map plot.
        """
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, 
                               figsize=(10, 10), dpi=150)
        
        Az, Zen = np.meshgrid(azimuth_grid, np.rad2deg(zenith_grid))
        power_norm = power_map / np.max(power_map) if np.max(power_map) > 0 else power_map
        
        pcm = ax.pcolormesh(Az, Zen, power_norm, cmap='bone', shading='auto')
        
        # Mark peak (CR direction)
        ax.plot(peak_azimuth, np.rad2deg(peak_zenith), 'r*', 
                markersize=20, label=f'Peak (zen={np.rad2deg(peak_zenith):.1f}°)')
        
        # Mark Sun if available
        if sun_zenith is not None and sun_azimuth is not None:
            ax.plot(sun_azimuth, np.rad2deg(sun_zenith), 'yo', 
                    markersize=15, markerfacecolor='yellow', markeredgecolor='orange',
                    markeredgewidth=2, label='Sun')
        
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 90)
        ax.set_yticks([15, 30, 45, 60, 75, 90])
        ax.set_yticklabels(['15°', '30°', '45°', '60°', '75°', '90°'], color='#FF8C00')  # Dark orange
        ax.tick_params(axis='y', colors='#FF8C00')  # Tick marks also orange
        
        ax.legend(loc='lower left', bbox_to_anchor=(0.0, -0.15))
        cbar = plt.colorbar(pcm, ax=ax, pad=0.12, shrink=0.8)
        cbar.set_label('Normalized Power', fontsize=11)
        
        ax.set_title(title, pad=20, fontsize=14)
        # Layout handled by gridspec
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved: {save_path}")
        
        plt.close(fig)
        return fig, ax
    

    def _plot_beamformed_trace(self, fft_traces, frequencies, positions, 
                                peak_zenith, peak_azimuth, sampling_rate,
                                title='Beamformed Time Trace', save_path=None):
        """
        Plot beamformed time trace towards peak direction with zoom-ins.
        Times are in ns (NuRadioReco standard unit).
        """
        import radiotools.helper as hp
        from NuRadioReco.modules.LOFAR.beamforming_utilities import geometric_delay_far_field, beamformer
        
        # Beamform towards peak direction
        direction = hp.spherical_to_cartesian(peak_zenith, peak_azimuth)
        delays = geometric_delay_far_field(positions, direction)
        beamformed_spectrum = beamformer(fft_traces, frequencies, delays)
        
        # Convert to time domain - times in ns (NRR standard)
        beamformed_trace = np.fft.irfft(beamformed_spectrum)
        n_samples = len(beamformed_trace)
        # sampling_rate is in GHz in NRR units, so 1/sampling_rate gives ns
        times_ns = np.arange(n_samples) / sampling_rate  # ns
        
        # Find peak in full trace
        total_time = times_ns[-1]
        peak_idx = np.argmax(np.abs(beamformed_trace))
        peak_time = times_ns[peak_idx]
        
        # Define zoom-safe range (avoid first/last 15% for zooms only)
        zoom_margin = 0.15 * total_time
        zoom_start = zoom_margin
        zoom_end = total_time - zoom_margin
        
        # Create figure: full trace on top (2 cols), 6 zoom panels below (3x2)
        fig = plt.figure(figsize=(16, 14), dpi=150)
        gs = fig.add_gridspec(4, 2, height_ratios=[1.5, 1, 1, 1], hspace=0.3, wspace=0.25)
        
        # Top panel: Full trace spanning 2 columns
        ax_full = fig.add_subplot(gs[0, :])
        ax_full.plot(times_ns, beamformed_trace, 'k-', lw=0.3, alpha=0.8)
        ax_full.axvline(peak_time, color='r', linestyle='--', alpha=0.7, lw=1, label=f'Peak @ {peak_time:.0f} ns')
        ax_full.set_xlabel('Time [ns]', fontsize=11)
        ax_full.set_ylabel('Amplitude', fontsize=11)
        ax_full.set_title('Full Trace', fontsize=12)
        ax_full.legend(loc='upper right', fontsize=9)
        
        # Define 6 zoom regions (~2000ns each, avoiding first/last 15%)
        zoom_duration = zoom_end - zoom_start
        zoom_width = 4000.0  # 2000 ns per zoom
        n_zooms = 5  # 5 evenly distributed + 1 around peak
        
        # Distribute 5 zoom windows evenly across valid range
        zoom_centers = np.linspace(zoom_start + zoom_width/2, zoom_end - zoom_width/2, n_zooms)
        
        zoom_regions = []
        for i, center in enumerate(zoom_centers):
            t_start = center - zoom_width/2
            t_end = center + zoom_width/2
            zoom_regions.append((f'Zoom {i+1}: {t_start:.0f}-{t_end:.0f} ns', t_start, t_end))
        
        # Add zoom around peak (last panel)
        peak_zoom_start = max(0, peak_time - zoom_width/2)
        peak_zoom_end = peak_time + zoom_width/2
        zoom_regions.append((f'Around Peak: {peak_zoom_start:.0f}-{peak_zoom_end:.0f} ns', peak_zoom_start, peak_zoom_end))
        
        # 6 zoom panels in 3 rows x 2 columns
        for i, (label, t_start, t_end) in enumerate(zoom_regions):
            row = 1 + i // 2
            col = i % 2
            ax = fig.add_subplot(gs[row, col])
            
            # Find indices for this time window
            mask = (times_ns >= t_start) & (times_ns <= t_end)
            if not np.any(mask):
                continue
            
            ax.plot(times_ns[mask], beamformed_trace[mask], 'k-', lw=0.5, alpha=0.8)
            
            # Mark peak if in this region
            if t_start <= peak_time <= t_end:
                ax.axvline(peak_time, color='r', linestyle='--', alpha=0.7, 
                          label=f'Peak @ {peak_time:.0f} ns')
                ax.legend(loc='upper right', fontsize=9)
            
            ax.set_xlabel('Time [ns]', fontsize=10)
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.set_title(label, fontsize=11)
        
        # Placeholder for axes return
        axes = None
            
        
        fig.suptitle(title, fontsize=14, y=1.02)
        # Layout handled by gridspec
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved: {save_path}")
        
        plt.close(fig)
        return fig, axes

    def _create_gif(self, window_maps, window_times, 
                    sun_zenith=None, sun_azimuth=None,
                    save_path=None, station_id=None, event_id=None):
        """
        Create animated GIF of time-windowed sky maps.
        """
        try:
            import imageio
        except ImportError:
            self.logger.warning("imageio not installed, cannot create GIF")
            return
        
        frames = []
        
        for i, (maps, times) in enumerate(zip(window_maps, window_times)):
            zen_grid, az_grid, power_map = maps
            start_us, end_us = times[0] * 1e6, times[1] * 1e6
            
            peak_zen, peak_az, _ = self._find_peak_direction(zen_grid, az_grid, power_map)
            
            # Create frame
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, 
                                   figsize=(8, 8), dpi=100)
            
            Az, Zen = np.meshgrid(az_grid, np.rad2deg(zen_grid))
            power_norm = power_map / np.max(power_map) if np.max(power_map) > 0 else power_map
            
            ax.pcolormesh(Az, Zen, power_norm, cmap='bone', shading='auto')
            ax.plot(peak_az, np.rad2deg(peak_zen), 'r*', markersize=15)
            
            if sun_zenith is not None:
                ax.plot(sun_azimuth, np.rad2deg(sun_zenith), 'yo', 
                        markersize=12, markerfacecolor='yellow', markeredgecolor='orange')
            
            ax.set_theta_zero_location('E')
            ax.set_theta_direction(-1)
            ax.set_ylim(0, 90)
            ax.set_title(f'Window {i+1}/{len(window_maps)}: {start_us:.1f}-{end_us:.1f} µs')
            
            # Layout handled by gridspec
            
            # Save to buffer
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            plt.close(fig)
        
        if save_path and frames:
            imageio.mimsave(save_path, frames, fps=2, loop=0)
            self.logger.info(f"Saved: {save_path}")
    
    @register_run()
    def run(self, event, station, detector):
        """
        Find CR direction via full-sky beamforming for a single station.
        
        Call this within a station loop, e.g.:
            for station in event.get_stations():
                mapper.run(event, station, detector)
        
        Parameters
        ----------
        event : Event
            The event being processed
        station : Station
            The station to process
        detector : Detector
            The detector description
        
        Notes
        -----
        Sets the following station parameters:
        
        - stationParameters.cr_zenith : CR arrival zenith angle
        - stationParameters.cr_azimuth : CR arrival azimuth angle
        - stationParameters.zenith : same, for voltageToEfieldConverter
        - stationParameters.azimuth : same, for voltageToEfieldConverter
        
        Also sets channelParameters.signal_regions if a signal window is detected.
        """
        # Check if station has data
        has_data = len(station.get_channel_ids()) > 0
        if not has_data:
            self.logger.debug(f"Station {station.get_id()}: no data, skipping")
            return
        
        station_id = station.get_id()
        event_id = event.get_id()
        
        self.logger.debug(f"Processing station CS{station_id:03d}")
        
        # Collect traces
        fft_traces, frequencies, positions, sampling_rate, n_samples = \
            self._collect_traces(station, detector)
        
        if fft_traces is None or len(fft_traces) < 2:
            self.logger.warning(f"Station {station_id}: insufficient data")
            return
        
        self.logger.debug(f"Station {station_id}: {len(fft_traces)} antennas, {n_samples} samples")
        
        # Get raw traces for time windowing
        if self.__use_efields:
            electric_fields = station.get_electric_fields()
            dominant_pol = 0  # Simplified
            raw_traces = np.array([f.get_trace()[dominant_pol] for f in electric_fields])
        else:
            channel_ids = station.get_channel_ids()
            groups = {}
            for ch_id in channel_ids:
                channel = station.get_channel(ch_id)
                gid = channel.get_group_id()
                if gid not in groups:
                    groups[gid] = []
                groups[gid].append(ch_id)
            
            raw_traces = []
            for gid, ch_ids in groups.items():
                trace_sum = None
                for ch_id in ch_ids:
                    trace = station.get_channel(ch_id).get_trace().astype(np.float64)
                    trace -= np.mean(trace)
                    trace_sum = trace if trace_sum is None else trace_sum + trace
                raw_traces.append(trace_sum)
            raw_traces = np.array(raw_traces)
        
        # Compute full-trace sky map
        self.logger.debug("Computing full-trace sky map...")
        zen_grid, az_grid, power_map = self._compute_sky_image(
            fft_traces, frequencies, positions
        )
        
        # Find peak in full-trace map
        peak_zen_full, peak_az_full, peak_power_full = self._find_peak_direction(
            zen_grid, az_grid, power_map
        )
        
        # Compute time-windowed maps
        self.logger.debug("Computing time-windowed sky maps...")
        window_maps, window_times = self._compute_time_windowed_images(
            raw_traces, frequencies, positions, sampling_rate
        )
        
        # Find strongest peak across all windows
        best_window_peak = (None, None, 0)
        best_window_idx = -1
        
        for w_idx, (maps, _) in enumerate(zip(window_maps, window_times)):
            w_zen_grid, w_az_grid, w_power_map = maps
            w_peak_zen, w_peak_az, w_peak_power = self._find_peak_direction(
                w_zen_grid, w_az_grid, w_power_map
            )
            if w_peak_power > best_window_peak[2]:
                best_window_peak = (w_peak_zen, w_peak_az, w_peak_power)
                best_window_idx = w_idx
        
        # Decide: use window peak if it's significantly stronger (transient)
        # This helps detect CR which are brief vs. continuous sources
        window_boost = best_window_peak[2] / (peak_power_full + 1e-30)
        
        if best_window_idx >= 0 and window_boost > 1.5:
            # Transient signal detected - use window peak
            cr_zenith = best_window_peak[0]
            cr_azimuth = best_window_peak[1]
            self.logger.info(f"Station {station_id}: Transient detected in window {best_window_idx+1}")
        else:
            # Use full-trace peak
            cr_zenith = peak_zen_full
            cr_azimuth = peak_az_full
        
        # Set station parameters (matching planeWaveDirectionFitter)
        station.set_parameter(stationParameters.cr_zenith, cr_zenith)
        station.set_parameter(stationParameters.cr_azimuth, cr_azimuth)
        
        # Also set zenith/azimuth for voltageToEfieldConverter
        station.set_parameter(stationParameters.zenith, cr_zenith)
        station.set_parameter(stationParameters.azimuth, cr_azimuth)
        
        # Determine CR signal window (time of strongest window)
        cr_signal_window = None
        if best_window_idx >= 0:
            cr_signal_window = window_times[best_window_idx]
            # Store as station parameter (start_time, end_time in seconds)
            # Using signal_regions format: (start_sample, end_sample)
            start_sample = int(cr_signal_window[0] * sampling_rate)
            end_sample = int(cr_signal_window[1] * sampling_rate)
            # Store in all channels of this station
            for channel in station.iter_channels():
                try:
                    channel.set_parameter(channelParameters.signal_regions, (start_sample, end_sample))
                except:
                    pass
        
        self.logger.info(
            f"Station CS{station_id:03d}: CR direction = "
            f"zen={np.rad2deg(cr_zenith):.1f}°, az={np.rad2deg(cr_azimuth):.1f}°"
        )
        
        if cr_signal_window is not None:
            self.logger.info(
                f"Station CS{station_id:03d}: CR signal window = "
                f"{cr_signal_window[0]*1e6:.1f} - {cr_signal_window[1]*1e6:.1f} µs"
            )
        
        # Check for Sun coincidence (warning)
        sun_zen, sun_az = None, None
        if self.__mark_sun:
            sun_zen, sun_az = self._get_sun_position(event)
            
        if sun_zen is not None:
            coincides, sep_deg = self._check_sun_coincidence(
                cr_zenith, cr_azimuth, sun_zen, sun_az, threshold_deg=10.0
            )
            if coincides:
                self.logger.warning(
                    f"Station CS{station_id:03d}: Peak direction is within {sep_deg:.1f}° of the Sun! "
                    f"This may be solar emission, not a cosmic ray."
                )
        
        # Debug output
        if self.__debug:
            # sun_zen, sun_az already computed above
            
            # Full-trace sky map
            self._plot_sky_map(
                zen_grid, az_grid, power_map,
                cr_zenith, cr_azimuth,
                sun_zenith=sun_zen, sun_azimuth=sun_az,
                title=f'Station CS{station_id:03d} - Event {event_id}\n'
                      f'CR: zen={np.rad2deg(cr_zenith):.1f}°, az={np.rad2deg(cr_azimuth):.1f}°',
                save_path=f"{self.__output_dir}/skymap_CS{station_id:03d}_{event_id}.png"
            )
            
            # Beamformed time trace towards peak
            self._plot_beamformed_trace(
                fft_traces, frequencies, positions,
                cr_zenith, cr_azimuth, sampling_rate,
                title=f'Station CS{station_id:03d} - Beamformed Trace (zen={np.rad2deg(cr_zenith):.1f}°, az={np.rad2deg(cr_azimuth):.1f}°)',
                save_path=f"{self.__output_dir}/beamtrace_CS{station_id:03d}_{event_id}.png"
            )
            
            # Animated GIF of time windows
            if len(window_maps) > 1:
                self._create_gif(
                    window_maps, window_times,
                    sun_zenith=sun_zen, sun_azimuth=sun_az,
                    save_path=f"{self.__output_dir}/skymap_CS{station_id:03d}_{event_id}.gif",
                    station_id=station_id, event_id=event_id
                )
        
        # Parameters have been set on station
    
    def end(self):
        pass

