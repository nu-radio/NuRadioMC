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
        tt_map = {}
        for ch in tt_paths:
            path = tt_paths[ch]
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path does not exist: {path}")
            else:
                try:
                    tt_map[ch] = np.load(path)
                except (OSError, ValueError, EOFError) as e:
                    raise RuntimeError(f"Failed to load file at {path}: {e}")
    
        self.__tt_map = tt_map 

        #detector 
        self.detector = detector

        #taking the hilbert envelope of traces for reconstruction, default = True
        self.do_envelope = True



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
                number of threads to be used to run tasks concurrently, default = 2
            
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
            number of threads to be used to run tasks concurrently, default = 2

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
            number of threads to be used to run tasks concurrently, default = 2

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
            number of threads to be used to run tasks concurrently, default = 2
            
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
    
    def run(self, event, station, channels_to_include, azimuth_range, elevation_range, radius, z_range, r_range, num_pts_r, num_pts_z, num_pts_elev, num_pts_az, output_path, cores = 2, plotting = True, run_no = None):
        
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
            number of threads to be used to run tasks concurrently, default = 2

        plotting: boolean 
            plot angular and RZ correlation maps, default = True

        output_path: string 
            path to folder where angular and RZ correlation maps are saved
            plots saved at f"{output_path}/{map_type}_{run_no}_{event_id}.png" if run_no is provided, else saved as f"{output_path}/{map_type}_{event_id}.png"
            map_type = "ang" or "rz"

        run_no: int
            run number for saving plots, default = None

        Returns
        
        -------
        
        score_rz, t_ab_rz : numpy arrays
            can be used to rebuild correlation map for later usage 

        maxcorr_point_all: dictionary 
            coordinates of maximum correlation (r,z,azimuth and elevation)
        
        maxcorr_rz: float
            maximum correlation value across rz correlation map 

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


        return score_rz, t_ab_rz, maxcorr_point_all, maxcorr_rz




