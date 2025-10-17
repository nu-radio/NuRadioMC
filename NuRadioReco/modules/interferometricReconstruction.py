import numpy as np 
import itertools 
import scipy.signal as signal 
from scipy.signal import windows
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import scipy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.colors import TwoSlopeNorm
from NuRadioReco.utilities.signal_processing import resample
from NuRadioReco.utilities import units
import os 

class InterferometricReco:

    """
    Module that does interferometric reconstruction for the diffuse neutrino search for RNO-G

    """

    def __init__(self, detector, station_id, tt_paths): 
        """
        Initializes reconstruction 

        Parameters 
        
        ----------
        detector 

        station_id: int

        tt_paths: dictionary of npz file paths
            contains an array of travel times for each channel in station alongside metadata
            metadata: r_range, z_range, antenna_z and ice_model
        """
        if (isinstance(tt_paths, dict) == False):
            raise TypeError("tt_paths must be a dictionary")

        #origin
        self.origin = detector.get_relative_position(station_id, 0)
        
        #load travel time maps 
        
        self.__tt_map = self.load_tt_maps(tt_paths) 

        #detector 
        self.detector = detector

        #taking the hilbert envelope of traces for reconstruction, default = True
        self.do_envelope = True
    
    def load_tt_maps(self, tt_paths):
        """
        Loads the travel time maps 

        Parameters

        ----------
        
        tt_paths: dictionary of npz file paths
            contains an array of travel times for each channel in station alongside metadata
            metadata: r_range, z_range, antenna_z and ice_model

        """
        tt_map = {}
        for ch in tt_paths:
            path = tt_paths[ch]
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path does not exist: {path}")
            else:
                try:
                    with np.load(path) as npz:
                        tt_map[ch] = {
                                "r_range": npz["r_range"].copy(),
                                "z_range": npz["z_range"].copy(),
                                "data": npz["data"].copy()}
                        #tt_map[ch] = np.load(path)
                except (OSError, ValueError, EOFError) as e:
                    raise RuntimeError(f"Failed to load file at {path}: {e}")

        return tt_map

    #Basic helper functions for unit conversions and queuing channel positions

    def get_channel_positions(self, det, station_id, channels):
        """
        Outputs a dictionary of channel positions
        
        Parameters

        ----------

        det: detector object 

        station_id: int 
        
        channels: list or numpy array 
            channels_to_include 

        """
        channel_positions = {}
        for channel in channels:
            channel_positions[channel] = (det.get_relative_position(station_id, channel))
        return channel_positions
    
    def ang_to_cart(self, elevation, azimuth, radius, origin_xyz):
        """
        Conversion from spherical to cartesian coordinates accounting for origin offset
        origin_xyz must be a numpy array
        
        Parameters

        ----------
        elevation: numpy array 
            elevation values for conversion in radians 
            elevation = arctan(z/r) where r = x^2 + y^2
            valid range: -pi/2 to pi/2

        azimuth: numpy array 
            azimuth value for conversion in radians 
            azimuth = arctan(y/x)
            valid range: -pi to pi

        radius: float 
            radius for conversion in meters 

        origin_xyz: numpy array 
            origin for reconstruction in x,y,z coordinates, default = position of channel 0

        """
        xx = radius * np.cos(elevation) * np.cos(azimuth)
        yy = radius * np.cos(elevation) * np.sin(azimuth)
        zz = radius * np.sin(elevation)


        xyz = np.stack([xx, yy, zz], axis = -1) + origin_xyz
        return xyz
    
    def rz_to_cart(self, z, r, azimuth, origin_xyz):
        """
        Conversion from cylindrical to cartesian coordinates accounting for origin offset
        
        Parameters

        ----------

        z: numpy array 
            z values for conversion in meters 
        
        r: numpy array 
            r values for conversion in meters, r = x^2 + y^2 

        azimuth: float 
            azimuth value for conversion in radians 
            azimuth = arctan(y/x)
            valid range: -pi to pi

        origin_xyz: numpy array 
            origin for reconstruction in x,y,z coordinates, default = position of channel 0
        
        """
        xx = r * np.cos(azimuth)
        yy = r * np.sin(azimuth)
        zz = z

        xyz = np.stack([xx, yy, zz], axis = -1) + origin_xyz
        return xyz
    
    def to_antenna_rz_coordinates(self, pos, antenna_pos):
        """
        Converting to local antenna coordinates in the cylindrical coordinate system (rz)
        
        Parameters

        ----------

        pos: numpy array 
            x,y,z coordinates to convert 

        antenna_pos: numpy array 
            position of antenna in x,y,z coordinates

        """
        local_r = np.linalg.norm(pos[:, :2] - antenna_pos[:2], axis = 1)
        local_z = pos[:, 2]

        return np.stack([local_r, local_z], axis = -1)

    #Helper function for plotting correlation maps

    def plot(self, reco, maxcorr_point, map_type, azimuth_range, elevation_range, z_range, r_range, num_pts_z, num_pts_r, output_path, event_id, run_no):
        """
        Helper function for plotting angular and RZ correlation maps
        
        Parameters

        ----------
        
        reco: dictionary
            dictionary returned from reconstruction with values used (radius, elevation, azimuth for angular reconstruction and azimuth, z and r for rz reconstruction) along with correlation map
        
        maxcorr_point: dictionary 
            coordinate of max correlation in map 

        map_type: string 
            "ang" or "rz"

        azimuth_range: tuple
            azimuth range in radians for angular reconstruction
            azimuth = arctan(y/x)
            valid range: -pi to pi

        elevation_range: tuple
            elevation range in radians for angular reconstruction
            elevation = arctan(z/r) where r = x^2 + y^2
            valid range: -pi/2 to pi/2 

        z_range: tuple
            range in z for RZ reconstruction in meters

        r_range: tuple
            range in r for RZ reconstruction in meters
            r = x^2 + y^2 
        
        num_pts_z: int
            number of points in z for RZ reconstruction

        num_pts_r: int
            number of points in r for RZ reconstruction

        output_path: string 
            path to folder where angular and RZ correlation maps are saved
            plots saved at f"{output_path}/{map_type}_{run_no}_{event_id}.png" if run_no is provided, else saved as f"{output_path}/{map_type}_{event_id}.png"
            map_type = "ang" or "rz"

        event_id: int 
            event id number for saving plots 

        run_no: int
            run number for saving plots, default = None

        """
        fs = 13
        plot_axes = []
        plot_axes_ind = []

        if (map_type == "ang"):
            aspect = "auto"
            for ind, axis in enumerate(["elevation", "azimuth", "radius"]):
                if (axis != "radius"):
                    if len(reco[axis]) > 1:
                        plot_axes.append(axis)
                        plot_axes_ind.append(ind)
                else:
                    slice_axis = axis
                    slice_val = reco[slice_axis]
        if (map_type == "rz"):
            aspect = "equal"

            for ind, axis in enumerate(["z", "r", "azimuth"]):
                if axis != "azimuth":
                    plot_axes.append(axis)
                    plot_axes_ind.append(ind)
                else:
                    slice_axis = axis
                    slice_val = reco[slice_axis]

        axis_a, axis_b = plot_axes
        intmap_to_plot = np.squeeze(reco["map"])

        figsize = (4.6, 4)

        fig = plt.figure(figsize = figsize, layout = "constrained")
        gs = GridSpec(1, 1, figure = fig)
        ax = fig.add_subplot(gs[0])

        cscale = np.max(np.abs(reco["map"]))
        vmax = cscale
        vmin = -cscale
        xlim = None
        ylim = None
        masked_values = np.ma.masked_invalid(intmap_to_plot)
        
        if (map_type == "ang"):
            im = ax.imshow(np.flip(np.transpose(masked_values), axis = 0),
                    extent = [reco[axis_b][0], reco[axis_b][-1],
                        reco[axis_a][0], reco[axis_a][-1]],
                    cmap = 'viridis', norm=TwoSlopeNorm(vmin = vmin, vcenter= (vmin + vmax)/2, vmax=vmax), aspect = aspect, interpolation = "none")
            xlim = azimuth_range
            ylim = elevation_range
            ax.scatter(maxcorr_point[plot_axes[1]], maxcorr_point[plot_axes[0]], color = "gray", marker = "*", label = "max corr")
        if (map_type == "rz"):
            z_vals = np.linspace(z_range[0], z_range[1], num_pts_z)
            r_vals = np.linspace(r_range[0], r_range[1], num_pts_r)
            im = ax.pcolormesh(r_vals, z_vals, np.transpose(masked_values), cmap = "viridis", shading = "gouraud")
            ax.scatter(maxcorr_point[plot_axes[1]], maxcorr_point[plot_axes[0]], color = "gray", marker = "*", label = "max corr")
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
        cbar.ax.tick_params(labelsize = fs)
        cbar.set_label("Correlation", fontsize = fs)
        ax.set_xlabel(f"{axis_b} [m]", fontsize = fs)
        ax.set_ylabel(f"{axis_a} [m]", fontsize = fs)

        if (xlim != None and ylim != None):
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        ax.tick_params(axis = "y", direction = "in", left = True, right = True, labelsize = fs)
        ax.tick_params(axis = "x", direction = "in", bottom = True, top = True, labelsize = fs)

        ax.text(0.05, 0.92, f"{slice_axis} = {slice_val:.1f} m", transform = ax.transAxes, fontsize = fs)
        ax.legend()
        if (run_no == None):
            fig.savefig(f"{output_path}/{map_type}_{run_no}_{event_id}.png")
        else:
            fig.savefig(f"{output_path}/{map_type}_{event_id}.png")
            
        plt.close()

    
    #Correlation Score Calculation Class

    class CorrScoreProvider:

        def process_csp(self, pair, channel_times, channel_sigvals, upsample):
            """
            Obtaining correlation between pairs of channel traces
            
            Parameters
            
            ----------
            
            pair: tuple
                pair of channels to calculate correlation between

            channel_times: dictionary
                event times for each channel (numpy arrays)

            channel_sigvals: dictionary
                event traces (numpy arrays) for each channel, if do_envelope = True, these are hilberted traces
            
            upsample: int
                upsampling factor for traces, default = 10

            """
            ch_a, ch_b = pair
            tvals_a, tvals_b = channel_times[ch_a], channel_times[ch_b]
            sig_a, sig_b = channel_sigvals[ch_a], channel_sigvals[ch_b]

            #upsampling the original waveform

            target_dt = min(tvals_a[1] - tvals_a[0], tvals_b[1] - tvals_b[0]) / upsample
            
            sig_a_tvals_rs = np.linspace(tvals_a[0], tvals_a[-1], int((tvals_a[-1] - tvals_a[0]) / target_dt))
            sig_b_tvals_rs = np.linspace(tvals_b[0], tvals_b[-1], int((tvals_b[-1] - tvals_b[0]) / target_dt))
            sig_a_rs = resample(sig_a, upsample)
            sig_b_rs = resample(sig_b, upsample)

            #normalization (mean = 0, standard_deviation = 1)
            sig_a_rs_norm = (sig_a_rs - np.mean(sig_a_rs)) / np.std(sig_a_rs)
            sig_b_rs_norm = (sig_b_rs - np.mean(sig_b_rs)) / np.std(sig_b_rs)
            
            #number of overlapping values for normalization correlation
            normfact = signal.correlate(np.ones(len(sig_a_rs_norm)), np.ones(len(sig_b_rs_norm)), mode = "full")

            #correlate signals and convert lags to units of time 
            corrs = signal.correlate(sig_a_rs_norm, sig_b_rs_norm, mode = "full") / normfact
            lags = signal.correlation_lags(len(sig_a_rs_norm), len(sig_b_rs_norm), mode = "full")
            tvals = lags * target_dt + sig_a_tvals_rs[0] - sig_b_tvals_rs[0]
                
            #apply hann window to taper ends of correlation function

            windows3 = windows.hann(len(corrs))
            corrs *= windows3

            return (pair, corrs, tvals)
        
        def __init__(self, channel_sigvals, channel_times, channel_pairs_to_include, cores, upsample = 10):
            """
            Obtaining correlation between all pairs of channel traces specified by channel_pairs_to_include
            
            Parameters

            ----------
            
            channel_sigvals: dictionary
                event traces (numpy arrays) for each channel, if do_envelope = True, these are hilberted traces

            channel_times: dictionary
                event times for each channel (numpy arrays)

            channel_pairs_to_include: list
                all channel pairs from channels_to_include

            cores: int
                number of threads to be used to run tasks concurrently, default = 40
            
            upsample: int 
                upsampling factor for traces, default = 10 

            """
            self.corrs = {}
            self.tvals = {}

            bound_process_csp = partial(
                    self.process_csp,
                    channel_times=channel_times,
                    channel_sigvals=channel_sigvals,
                    upsample=upsample
                    )
            
            with ThreadPoolExecutor(max_workers=cores) as executor:
                results = executor.map(bound_process_csp, channel_pairs_to_include)
            for pair, corrs, tvals in results:
                self.corrs[pair] = corrs
                self.tvals[pair] = tvals

        def get(self, ch_a, ch_b, t_ab):
            """
            Obtain correlation function 
            
            Parameters

            ----------

            ch_a, ch_b: ints 
                channels to calculate correlation function for

            t_ab: numpy array 
                delay time (travel_times in ch_a - travel_times in ch_b)

            """
            corrvals = self.corrs[(ch_a, ch_b)]
            tvals = self.tvals[(ch_a, ch_b)]

            return np.interp(t_ab, tvals, corrvals, left=0, right=0)
    
    #Parsing Travel Time Maps Helper Functions
    def get_travel_time(self, ttcs, ch, coord):
        """
        Obtain travel time at the nearest pixel/coordinate in the travel time map
        
        Parameters

        ----------
        
        ttcs: dict
            travel time maps (numpy arrays) for all channels

        ch: int 
            channel to obtain travel time for 

        coord: numpy array 
            array of coordinates to find travel time for 

        """
        ch_npz = ttcs[ch]
        r = ch_npz["r_range"]
        z = ch_npz["z_range"]
        times = ch_npz["data"]
        
        r_range = (np.min(r) * 0.3, np.max(r) * 0.3)
        z_range = (np.min(z) * 0.3, np.max(z) * 0.3)
        num_pts_r = len(r)
        num_pts_z = len(z)

        domain_start = np.array([r_range[0], z_range[0]])
        domain_end = np.array([r_range[1], z_range[1]])
        domain_shape = np.array([num_pts_r, num_pts_z])

        #finding nearest travel time pixel in the grid

        pixel_2d = (coord - domain_start) / (domain_end - domain_start) * domain_shape
        pixel_int = pixel_2d.astype(int)

        pixels = np.transpose(pixel_int)

        rows, cols = pixels[0], pixels[1]

        values = (times)[rows, cols]

        return values

    #Interferometry Helper Functions

    def process_pair(self, pair, travel_times, csp):
        """
        Obtaining correlation score using correlation function for a pair of channels
        
        Parameters 

        ----------

        pair: tuple 
            pair of channels to calculate correlation function between 

        travel_times: dict
            travel time maps (numpy arrays) for all channels

        csp: object of class CorrScorProvider
            calculate correlation score

        """
        ch_a, ch_b = pair
        
        #get delay times/lags 
        t_ab = travel_times[ch_a] - travel_times[ch_b]
        corrvals = csp.corrs[(ch_a, ch_b)]
        tvals = csp.tvals[(ch_a, ch_b)]
        
        #map delay times to correlation scores 
        score = np.interp(t_ab, tvals, corrvals, left=0, right=0)

        return np.nan_to_num(score, nan=0.0)

    def calc_corr_score(self,channel_signals, channel_times, pts, ttcs, origin, channel_pairs_to_include, channel_positions,csp, cores):
        """
        Obtaining correlation score using correlation function for all channel pairs as specified by channel_pairs_to_include
        
        Parameters 
        
        ----------

        channel_signals: dictionary
            event traces (numpy arrays) for each channel, if do_envelope = True, these are hilberted traces

        channel_times: dictionary
            event times for each channel (numpy arrays)
        
        pts: numpy array 
            cartesian coordinates used for reconstruction 
        
        ttcs: dictionary
            travel time maps (numpy arrays) for all channels
        
        origin: numpy array
            origin for reconstruction in x,y,z coordinates, default = position of channel 0
        
        channel_pairs_to_include: list
            all channel pairs from channels_to_include

        channel_positions: dictionary
            x,y,z positions of all channels

        csp: object of class CorrScorProvider
            calculate correlation score

        cores: int
            number of threads to be used to run tasks concurrently, default = 40

        """
        corrs = csp.corrs
        tvals = csp.tvals

        channels = list(set([channel for pair in channel_pairs_to_include for channel in pair]))

        travel_times = {ch: self.get_travel_time(ttcs, ch, (self.to_antenna_rz_coordinates(pts, channel_positions[ch]))) for ch in channels}

        bound_process_pair = partial(self.process_pair, travel_times=travel_times, csp=csp)

        with ThreadPoolExecutor(max_workers=cores) as executor:
            scores = list(executor.map(bound_process_pair, channel_pairs_to_include))


        return tvals, np.mean(scores, axis = 0), corrs

    def build_interferometric_map_ang(self, channel_signals, channel_times, channel_pairs_to_include, channel_positions,
                                  rad, origin_xyz, elevation_range, azimuth_range, num_pts_elevation, num_pts_azimuth, ttcs, csp, cores):
        """
        Obtain angular correlation map
        
        Parameters

        ----------

        *same as function interferometric_reco_ang

        """
        elevation_vals = np.linspace(*elevation_range, num_pts_elevation)
        azimuth_vals = np.linspace(*azimuth_range, num_pts_azimuth)


        ee, aa = np.meshgrid(elevation_vals, azimuth_vals)


        # convert to cartesian points
        pts = self.ang_to_cart(ee.flatten(), aa.flatten(), radius = rad, origin_xyz = origin_xyz)


        t_ab, intmap, scores_all  = self.calc_corr_score(channel_signals, channel_times, pts, ttcs, origin_xyz, channel_pairs_to_include,
                                channel_positions = channel_positions, csp = csp, cores=cores)


        intmap = np.reshape(intmap.flatten(), (num_pts_azimuth, num_pts_elevation), order = "C")
        
        return elevation_vals, azimuth_vals, intmap, scores_all, t_ab

    def build_interferometric_map_rz(self, channel_signals, channel_times, channel_pairs_to_include, channel_positions,
                                  azimuth, origin_xyz, z_range, r_range, num_pts_z, num_pts_r, ttcs, csp, cores):
        """
        Obtain RZ correlation map

        Parameters 

        ----------

        *same as function interferometric_reco_rz

        """
        z_vals = np.linspace(*z_range, num_pts_z)
        r_vals = np.linspace(*r_range, num_pts_r)
        zz, rr = np.meshgrid(z_vals, r_vals)

        # convert to cartesian points
        pts = self.rz_to_cart(zz.flatten(), rr.flatten(), azimuth = azimuth, origin_xyz = origin_xyz)

        t_ab, intmap, scores_all = self.calc_corr_score(channel_signals, channel_times, pts, ttcs, origin_xyz, channel_pairs_to_include,
                             channel_positions = channel_positions, csp = csp, cores=cores)


        intmap = np.reshape(intmap, (num_pts_z, num_pts_r), order = "C")

        return z_vals, r_vals, intmap, scores_all, t_ab

    def interferometric_reco_ang(self, ttcs, channel_signals, channel_times,
                             rad, origin_xyz, elevation_range, azimuth_range, num_pts_elevation, num_pts_azimuth,
                             channel_pairs_to_include, channel_positions, csp, cores):
        
        """
        Return reconstructed event as dictionary with angular map
        
        Parameters 
        
        ----------

        ttcs: dictionary
            travel time maps (numpy arrays) for all channels

        channel_signals: dictionary
            event traces (numpy arrays) for each channel, if do_envelope = True, these are hilberted traces

        channel_times: dictionary
            event times for each channel (numpy arrays)

        rad: float
            radius to be used for angular reconstruction in meters

        origin_xyz: numpy array
            origin for reconstruction in x,y,z coordinates, default = position of channel 0

        elevation_range: tuple
            elevation range in radians for angular reconstruction
            elevation = arctan(z/r) where r = x^2 + y^2
            valid range: -pi/2 to pi/2
        
        azimuth_range: tuple
            azimuth range in radians for angular reconstruction
            azimuth = arctan(y/x)
            valid range: -pi to pi
        
        num_pts_elevation: int
            number of points in elevation for angular reconstruction

        num_pts_azimuth: int
            number of points in azimuth for angular  reconstruction

        channel_pairs_to_include: list
            all channel pairs from channels_to_include

        channel_positions: dictionary
            x,y,z positions of all channels

        csp: object of class CorrScorProvider
            calculate correlation score

        cores: int
            number of threads to be used to run tasks concurrently, default = 40

        """
        elevation_vals, azimuth_vals, intmap, score, t_ab = self.build_interferometric_map_ang(channel_signals, channel_times, channel_pairs_to_include,
                                                                         channel_positions = channel_positions,
                                                                         rad = rad, origin_xyz = origin_xyz, elevation_range = elevation_range, azimuth_range = azimuth_range,
                                                                         num_pts_elevation = num_pts_elevation, num_pts_azimuth = num_pts_azimuth, ttcs = ttcs, csp = csp, cores=cores)

        reco_event = {
            "elevation": elevation_vals,
            "azimuth": azimuth_vals,
            "radius": rad,
            "map": intmap
        }

        return reco_event, score, t_ab

    def interferometric_reco_rz(self, ttcs, channel_signals, channel_times,
                             azimuth, origin_xyz, z_range, r_range, num_pts_z, num_pts_r,
                             channel_pairs_to_include, channel_positions, csp, cores):
        """
        Return reconstructed event as dictionary with RZ map

        Parameters

        ----------

        ttcs: dictionary 
            travel time maps (numpy arrays) for all channels 

        channel_signals: dictionary
            event traces (numpy arrays) for each channel, if do_envelope = True, these are hilberted traces 

        channel_times: dictionary
            event times for each channel (numpy arrays)

        azimuth: float 
            azimuth associated with max correlation point from angular reconstruction in radians 

        origin_xyz: numpy array 
            origin for reconstruction in x,y,z coordinates, default = position of channel 0 

        z_range: tuple 
            range in z for RZ reconstruction in meters

        r_range: tuple
            range in r for RZ reconstruction in meters
            r = x^2 + y^2

        num_pts_z: int
            number of points in z for RZ reconstruction

        num_pts_r: int
            number of points in r for RZ reconstruction 
        
        channel_pairs_to_include: list 
            all channel pairs from channels_to_include

        channel_positions: dictionary
            x,y,z positions of all channels

        csp: object of class CorrScorProvider 
            calculate correlation score 

        cores: int
            number of threads to be used to run tasks concurrently, default = 40
            
        """
        z_vals, r_vals, intmap, score, t_ab = self.build_interferometric_map_rz(channel_signals, channel_times, channel_pairs_to_include,
                                                                         channel_positions = channel_positions,
                                                                         azimuth = azimuth, origin_xyz = origin_xyz, z_range = z_range, r_range = r_range,
                                                                         num_pts_z = num_pts_z, num_pts_r = num_pts_r, ttcs = ttcs, csp = csp, cores=cores)

        reco_event = {
            "z": z_vals,
            "r": r_vals,
            "azimuth": azimuth,
            "map": intmap
        }

        return reco_event, score, t_ab

    def run(self, event, station, channels_to_include, azimuth_range, elevation_range, radius, z_range, r_range, num_pts_r, num_pts_z, num_pts_elev, num_pts_az, output_path, cores = 2, plotting = True, run_no = None, return_reco = False, return_score = False, return_delays = False, return_maps = False):
        
        """
        Runs reconstruction 

        Parameters 
        
        ----------

        event 

        station
        
        channels_to_include: list or array 
            channels to be used for reconstruction 

        azimuth_range: tuple
            azimuth range in radians for angular reconstruction
            azimuth = arctan(y/x)
            valid range: -pi to pi

        elevation_range: tuple
            elevation range in radians for angular reconstruction
            elevation = arctan(z/r) where r = x^2 + y^2
            valid range: -pi/2 to pi/2 

        radius: float 
            radius for angular reconstruction in meters
            radius = x^2 + y^2 + z^2

        z_range: tuple
            range in z for RZ reconstruction in meters

        r_range: tuple
            range in r for RZ reconstruction in meters
            r = x^2 + y^2 

        num_pts_r: int
            number of points in r for RZ reconstruction 

        num_pts_z: int
            number of points in z for RZ reconstruction

        num_pts_elev: int
            number of points in elevation for angular reconstruction

        num_pts_az: int
            number of points in azimuth for angular  reconstruction

        output_path: string 
            path to folder where angular and RZ correlation maps are saved
        
        cores: int
            number of threads to be used to run tasks concurrently, default = 40

        plotting: boolean 
            plot angular and RZ correlation maps, default = True

        output_path: string 
            path to folder where angular and RZ correlation maps are saved
            plots saved at f"{output_path}/{map_type}_{run_no}_{event_id}.png" if run_no is provided, else saved as f"{output_path}/{map_type}_{event_id}.png"
            map_type = "ang" or "rz"

        run_no: int
            run number for saving plots, default = None
        
        return_reco: boolean 
            if set to True, returns dictionary containing correlation map (numpy array), z_vals, r_vals, and azimuth 
            z_vals, r_vals: z and r values for RZ reconstruction in meters where r = x^2 + y^2
            azimuth: azimuth associated with max correlation point from angular reconstruction in radians
            *necessary for surface correlation ratio (SCR) calculation
            default = False

        return_score = boolean
            if set to True, return a numpy array of the average correlation score over all channel pairs at each delay time 
            *necessary for coherently summed waveform (CSW) calculation 
            default = False 

        return_delays = boolean 
            if set to True, returns a numpy array of the delay times corresponding to each correlation value 
            *necessary for coherently summed waveform (CSW) calculation
            default = False

        return_maps = boolean
            if set to True, returns the dictionary of travel time maps (numpy arrays) for each channel and metadata
            metadata: r_range (cylindrical) and z_range over which maps were generated in meters  
            *necessary for coherently summed waveform (CSW) calculation
            default = False
        
        Returns 

        -------

        results: dictionary
            results["maxcorr_coord"] = dictionary of the coordinate (z, r, azimuth and elevation) of max correlation 
            results["maxcorr"] = maximum correlation value from rz reconstruction 
            if return_reco is set to True
                results["reco"] = *see return_reco parameter description above 
            if return_score is set to True
                results["score"] = *see return_score parameter description above 
            if return_delays is set to True 
                results["delays"] = *see return_delays parameter description above 
            if return_maps is set to True 
                results["maps"] = *see return_maps parameter description above 

        """
        
        #check if provided arguments from user are valid 

        if (isinstance(channels_to_include, list) == False and isinstance(channels_to_include, np.ndarray) == False):
            raise TypeError("channels to include must be a list or numpy array")
        if (isinstance(azimuth_range, tuple) == False or isinstance(elevation_range, tuple) == False or isinstance(z_range, tuple) == False or isinstance(r_range, tuple) == False):
            raise TypeError("ranges of variables for reconstruction must be provided as tuples")
        if (azimuth_range[1] <= azimuth_range[0] or elevation_range[1] <= elevation_range[0] or z_range[1] <= z_range[0] or r_range[1] <= r_range[0]):
            raise ValueError("the second value in the range must be larger than the first")
        if (azimuth_range[1] > np.pi or azimuth_range[0] < -np.pi or elevation_range[1] > np.pi/2 or elevation_range[0] < -np.pi/2):
            raise ValueError("the azimuth range must be between -pi to pi and the elevation range must be between -pi/2 to pi/2")
        if (r_range[0] < 0):
            raise ValueError("r must be positive")
        if (radius < 0):
            raise ValueError("the radius for angular reconstruction must be a positive float value")
        if (isinstance(num_pts_r, int) == False or isinstance(num_pts_z, int) == False or isinstance(num_pts_elev, int) == False or isinstance(num_pts_az, int) == False):
            raise TypeError("the number of points used for reconstruction must be provided as integers")
        if (num_pts_r <= 0 or num_pts_z <= 0 or num_pts_elev <= 0 or num_pts_az <= 0):
            raise ValueError("the number of points used for reconstruction must be a positive integer")


        channel_pairs_to_include = list(itertools.combinations(channels_to_include, 2))
        
        #extract event traces and apply hilbert envelope

        channel_signals = {}
        channel_times = {}
        for ch in station.iter_channels():
            if (ch.get_id() in channels_to_include):
                trace = ch.get_trace()
                times = ch.get_times()
                if (self.do_envelope == False):
                    channel_signals[ch.get_id()] = trace
                else:
                    channel_signals[ch.get_id()] = np.abs(scipy.signal.hilbert(trace))
                channel_times[ch.get_id()] = times
            

        csp = self.CorrScoreProvider(channel_signals, channel_times, channel_pairs_to_include, cores)
        channel_positions = self.get_channel_positions(self.detector, station_id = station.get_id(), channels = channels_to_include)
        
        #angular reconstruction
        reco_ang, score_ang, t_ab_ang  = self.interferometric_reco_ang(self.__tt_map, channel_signals, channel_times,
                                               rad = radius, origin_xyz = self.origin, elevation_range = elevation_range, azimuth_range = azimuth_range,
                                               num_pts_elevation = num_pts_elev, num_pts_azimuth = num_pts_az, channel_pairs_to_include = channel_pairs_to_include,
                                               channel_positions = channel_positions, csp = csp, cores=cores)

        #maximum correlation point in angular map 

        if np.all(np.isnan(reco_ang["map"])):
            maxcorr_point_ang = {"elevation": np.nan,"azimuth": np.nan}
            maxcorr_ang = np.nan
        else:
            mapdata = reco_ang["map"]
            nanmax_val = np.nanmax(mapdata)
            maxind = np.unravel_index(np.nanargmax(mapdata), mapdata.shape)
            maxcorr_point_ang = {"elevation": reco_ang["elevation"][maxind[1]],"azimuth": reco_ang["azimuth"][maxind[0]]}
            maxcorr_ang = mapdata[maxind[0]][maxind[1]]
        
        reco_rz, score_rz, t_ab_rz = self.interferometric_reco_rz(self.__tt_map, channel_signals, channel_times, azimuth = maxcorr_point_ang["azimuth"], origin_xyz = self.origin,
                                            z_range = z_range, r_range = r_range, num_pts_z = num_pts_z, num_pts_r = num_pts_r, 
                                            channel_pairs_to_include = channel_pairs_to_include, channel_positions = channel_positions,csp = csp, cores=cores)
        
        #maximum correlation point in RZ map
        
        if np.all(np.isnan(reco_rz["map"])):
            maxcorr_point_all = {"z": np.nan,"r": np.nan, "azimuth" : maxcorr_point_ang["azimuth"], "elevation" : maxcorr_point_ang["elevation"]}
            maxcorr_rz = np.nan
        else:
            mapdata = reco_rz["map"]
            nanmax_val = np.nanmax(mapdata)
            maxind = np.unravel_index(np.nanargmax(mapdata), mapdata.shape)
            maxcorr_point_all = {"z": reco_rz["z"][maxind[1]],"r": reco_rz["r"][maxind[0]], "azimuth" : maxcorr_point_ang["azimuth"], "elevation" : maxcorr_point_ang["elevation"]}
            maxcorr_rz = mapdata[maxind[0]][maxind[1]]
        
        #plot correlation maps 
        if (plotting == True):
            if os.path.exists(output_path):
                self.plot(reco_ang, maxcorr_point_ang, "ang", azimuth_range, elevation_range, z_range, r_range, num_pts_z, num_pts_r, output_path, event.get_id(), run_no)
                self.plot(reco_rz, maxcorr_point_all, "rz", azimuth_range, elevation_range, z_range, r_range, num_pts_z, num_pts_r, output_path, event.get_id(), run_no)
            else:
                raise FileNotFoundError(f"Path does not exist: {output_path}")
        
        results = {}
        if (return_reco == True):
            results["reco"] = reco_rz 
        if (return_score == True):
            results["score"] = score_rz 
        if (return_delays == True):
            results["delays"] = t_ab_rz 
        if (return_maps == True):
            results["maps"] = self.__tt_map 
        
        results["maxcorr_coord"] = maxcorr_point_all
        results["maxcorr"] = maxcorr_rz 

        return results 


class CSW(InterferometricReco):
    def __init__(self, station_id, detector):
        """
        Initialize CSW class

        Parameters

        ----------

        station_id: int

        detector
        """

        #window around reconstructed delay times to find final delay time values
        self.__zoom_window = 40 * units.ns

        #detector
        self.det = detector

        #origin
        self.origin = detector.get_relative_position(station_id, 0)

    def get_arrival_delays_reco(self, reco_results, channels_to_include, channel_positions, reference_ch, ttcs):
        """
        Obtain difference between travel time at the maximum correlation point for each channel in channels_to_include and reference channel 

        Parameters

        ----------
        reco_results: dictionary
            maximum correlation point in r, z, azimuth and elevation as obtained from reconstruction

        channels_to_include: list or numpy array
            list of channels used to make the CSW
    
        channel_positions: dictionary
            x,y,z positions of each channel in channels_to_include 

        reference_ch: int
            channel with the maximum voltage in its trace (highest peak)

        ttcs: dictionary
            dictionary of travel time maps (numpy arrays) for each channel and metadata
            metadata: r_range (cylindrical) and z_range over which maps were generated in meters
        """

        z, r = np.meshgrid(reco_results["z"], reco_results["r"])
        src_pos = self.rz_to_cart(z.flatten(), r.flatten(), reco_results["azimuth"], self.origin)

        arrival_times = {}

        #obtain travel time for each channel at coordinate of maximum correlation 
        for ch in channels_to_include:
            arrival_times[ch] = self.get_travel_time(ttcs, ch, self.to_antenna_rz_coordinates(src_pos, channel_positions[ch]))

        reference_arrival_time = arrival_times[reference_ch]
        arrival_delays = {}

        #obtain difference with respect to reference channel arrival time
        for ch in channels_to_include:
            arrival_delays[ch] = arrival_times[ch] - reference_arrival_time


        return arrival_delays
    
    def get_arrival_delays_xcorr(
        self, channel_signals, channel_times, channels_to_include, reference_ch, reco_delays,channel_positions, score, t_ab
    ):
        """
        Obtain delay time within a window (self.window) around the reconstructed delay time from get_arrival_delays_reco
        
        Parameters

        ----------
        channel_signals: dictionary
            event traces (numpy arrays) for each channel

        channel_times: dictionary
            event times for each channel (numpy arrays) 
        
        channels_to_include: list or numpy array
            list of channels used to make the CSW

        reference_ch: int
            channel with the maximum voltage in its trace (highest peak)

        reco_delays: dictionary
            difference between travel time at the maximum correlation point for each channel in channels_to_include and reference channel

        channel_positions: dictionary
            x,y,z positions of each channel in channels_to_include
        
        score: array
            average correlation score over all channel pairs at each delay time as obtained from reconstruction

        t_ab: array
            delay times corresponding to each correlation value as obtained from reconstruction
        """
        delays = {}

        for ch_ID in channels_to_include:

            #delay of reference_ch with respect to reference_ch is 0

            if ch_ID == reference_ch:
                delay = 0
            else:
                #account for the fact that delay times (t_ab) are calculated as t_a - t_b 
                if (ch_ID > reference_ch):
                    channel_pair = (reference_ch, ch_ID)
                    xcorr_times, xcorr_volts = t_ab[channel_pair], score[channel_pair]
                    xcorr_times *= -1
                else:
                    channel_pair = (ch_ID, reference_ch)
                    xcorr_times, xcorr_volts = t_ab[channel_pair], score[channel_pair]

                #look at window around delay time

                zoomed_indices = np.where(
                    (np.abs( xcorr_times - reco_delays[ch_ID] )) < self.__zoom_window // 2
                )[0]

                #if nothing is in the window, keep delay as is 
                if len(zoomed_indices) == 0:
                    delay = xcorr_times[ np.argmax(xcorr_volts) ]

                #else find the delay associated with maximum correlation in zoomed array
                else:
                    zoomed_indices_filled = np.arange(min(zoomed_indices), max(zoomed_indices), 1)


                    delay = xcorr_times[
                        np.argmax(xcorr_volts[zoomed_indices])
                        + zoomed_indices[0]
                    ]


            delays[ch_ID] = delay

        return delays

    def run(self, event, station, channels_to_include, ttcs, reco_results, score, t_ab):
        """
        Finding the coherently summed waveform (CSW) using results of event reconstruction 

        Parameters 

        ----------

        event 

        station 

        channels_to_include: list or numpy array
            list of channels used to make the CSW

        ttcs: dictionary 
            dictionary of travel time maps (numpy arrays) for each channel and metadata
            metadata: r_range (cylindrical) and z_range over which maps were generated in meters 

        reco_results: dictionary
            maximum correlation point in r, z, azimuth and elevation as obtained from reconstruction 

        score: array
            average correlation score over all channel pairs at each delay time as obtained from reconstruction 

        t_ab: array
            delay times corresponding to each correlation value as obtained from reconstruction 

        """

        #get channel positions 

        channel_positions = self.get_channel_positions(self.det, station_id = station.get_id(), channels = channels_to_include)

        #obtain traces and times for each channel
        channel_times = {}
        channel_signals = {}

        for channel in station.iter_channels():
            if (channel.get_id() in channels_to_include):
                volts = channel.get_trace()
                times = channel.get_times()
                channel_times[channel.get_id()] = times
                channel_signals[channel.get_id()] = volts

        #set the reference channel as the channel with the maximum voltage in its trace
        reference_ch = -123456
        reference_ch_max_voltage = -1
        for ch_ID in channels_to_include:
            this_max_voltage = np.max(channel_signals[ch_ID])
            if this_max_voltage > reference_ch_max_voltage:
                reference_ch_max_voltage = this_max_voltage
                reference_ch = ch_ID
        
        #obtain arrival delays at reconstructed maximum correlation point 
        arrival_delays_reco = self.get_arrival_delays_reco(reco_results, channels_to_include, channel_positions, reference_ch, ttcs)

        #obtain delay times within a window (self.window) around arrival_delays_reco
        arrival_delays = self.get_arrival_delays_xcorr(channel_signals, channel_times, channels_to_include, reference_ch, arrival_delays_reco, channel_positions, score, t_ab)

        #time associated with maximum voltage in reference channel trace
        expected_signal_time = np.asarray(channel_times[reference_ch])[
            np.argmax( np.asarray(channel_signals[reference_ch]) )
        ]

        #find channel_id and length of shortest waveform
        shortest_wf_ch = 123456
        shortest_wf_length = np.inf
        for ch_ID in channels_to_include:
            if len(channel_signals[ch_ID]) < shortest_wf_length:
                shortest_wf_length = len(channel_signals[ch_ID])
                shortest_wf_ch = ch_ID

        #define csw time and voltage arrays with lengths equal to shortest waveform length
        csw_values = np.zeros((1, shortest_wf_length))
        csw_times = np.asarray(
            channel_times[reference_ch])[:shortest_wf_length]

        csw_dt = csw_times[1] - csw_times[0]

        for ch_ID in channels_to_include:
            values = np.asarray(channel_signals[ch_ID])

            #offset the channel time array by the delay time
            times = np.asarray(channel_times[ch_ID]) - (arrival_delays[ch_ID]//csw_dt)*csw_dt

            #initial time shift needed to align this channel's trace with the CSW
            rebinning_shift = (
                (csw_times[0] - times[0])
                % csw_dt)

            #If the channel trace is longer than the CSW, it needs to be trimmed
            if len(times) > len(csw_times):
                trim_ammount = len(times) - len(csw_times)
                if (
                    ( times[0] - csw_times[0] < 0 ) # this wf has an earlier start time than the CSW
                    and ( times[-1] - csw_times[-1] <= csw_dt/2) # this wf has a earlier or equal end time than the CSW
                ): # trim from the beginning of the waveform
                    times  = times [trim_ammount:]
                    values = values[trim_ammount:]
                elif (
                    ( times[0] - csw_times[0] > -csw_dt/2) # this wf has a later or equal start time than the CSW
                    and (times[-1] - csw_times[-1] > 0) # this wf has a later end time than the CSW
                ): # trim from the end of the waveform
                    times  = times [:-trim_ammount]
                    values = values[:-trim_ammount]
                elif (
                    ( times[0] - csw_times[0] < 0 ) # this wf starts earlier than the CSW
                    and ( times[-1] - csw_times[-1] > 0 ) # this wf ends later than the CSW
                ): # trim from both ends of the waveform
                    leading_trimmable = np.argwhere(
                        np.round(times,5) < np.round(csw_times[0], 5) )
                    trailing_trimmable = np.argwhere(np.round(times, 5) > np.round(csw_times[-1], 5) )
                    times  = times [ len(leading_trimmable) : -len(trailing_trimmable) ]
                    values = values[ len(leading_trimmable) : -len(trailing_trimmable) ]

            # Calculate the number of bins to roll the channel trace so that it aligns with the CSW
            roll_shift_bins = (csw_times[0] - times[0]) / csw_dt
            roll_shift_time = roll_shift_bins*(times[1] - times[0])
            roll_shift_bins = int(roll_shift_bins)

            # Roll trace to align with CSW and adjust the time array accordingly 
            rolled_wf = np.roll( values, -roll_shift_bins )
            rolled_times = np.linspace(
                times[0] + roll_shift_time,
                times[-1] + roll_shift_time,
                len(times)
            )

            # Add this channel's waveform to the CSW
            csw_values = np.sum( np.dstack( (csw_values[0], rolled_wf) ), axis=2)

        csw_values = np.squeeze(csw_values)

        return (csw_times, csw_values)

class SurfaceCorr:

    def __init__(self, station_id, detector):
        """
        Initializes surface correlation class 

        Parameters
        
        ----------
        detector

        station_id: int
        """
        #threshold = +/- z_thresh within which the maximum surface correlation is looked for in meters
        self.z_thresh = -10

        #detector
        self.detector = detector

        #origin 
        self.origin = detector.get_relative_position(station_id, 0)

    def run(self, intmap, maxcorr):
        """
        Calculate surface correlation ratio (SCR) i.e. ratio between maximum correlation in a +/- self.z_thresh band around the surface to maximum correlation across whole range (maxcorr)

        Parameters

        ----------
        intmap: dictionary 
            returned by rz reconstruction containing correlation map (numpy array), z_vals, r_vals, and azimuth 
            z_vals, r_vals: z and r values for RZ reconstruction in meters where r = x^2 + y^2
            azimuth: azimuth associated with max correlation point from angular reconstruction in radians

        maxcorr: float 
            maximum correlation in the correlation map obtained from RZ reconstruction

        """
        _, _, avg_z = self.origin

        #threshold around surface accounting for the origin offset
        z_thresh = (abs(avg_z) + self.z_thresh)
        z_thresh_up = (abs(avg_z) - self.z_thresh)

        #maximum correlation within surface region along with r,z coordinate 
        row, col = intmap["map"].shape
        cols = np.where(np.logical_and(intmap["z"].flatten() >= z_thresh, intmap["z"].flatten() <= z_thresh_up))
        surf_array = (intmap["map"])[:row, min(cols[0]):max(cols[0])+1]
        max_surf_corr = np.max(surf_array)

        maxind = np.unravel_index(np.argmax(surf_array), surf_array.shape)
        max_z = intmap["z"].flatten()[min(cols[0]):max(cols[0])+1][maxind[1]]
        max_r = intmap["r"].flatten()[:row][maxind[0]]

        #ratio of maximum surface correlation to maxcorr
        if (maxcorr != 0):
            surf_corr_ratio = max_surf_corr / maxcorr
        else:
            surf_corr_ratio = np.inf

        return surf_corr_ratio, max_surf_corr, max_r, max_r

    
