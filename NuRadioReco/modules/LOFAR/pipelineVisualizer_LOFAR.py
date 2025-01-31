"""
This module contains the pipelineVisualizer class for LOFAR.

.. moduleauthor:: Karen Terveer <karen.terveer@fau.de>
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import radiotools
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import stationParameters, channelParameters, showerParameters
from NuRadioReco.modules.base.module import register_run


def check_for_good_ant(event, detector):
    """
    Create a dictionary which for every station in the event contains a list of antennas which have not
    been flagged.
    """
    good_antennas_dict = {}
    for station in event.get_stations():
        if station.get_parameter(stationParameters.triggered):
            good_antennas_dict[station.get_id()] = []
            flagged_channels = station.get_parameter(stationParameters.flagged_channels)
            # Get all group IDs which are still present in the station
            station_channel_group_ids = set([channel.get_group_id() for channel in station.iter_channels()])

            # Get the dominant polarisation orientation as calculated by stationPulseFinder
            dominant_orientation = station.get_parameter(stationParameters.cr_dominant_polarisation)

            good_channel_pair_ids = np.zeros((len(station_channel_group_ids), 2), dtype=int)
            for ind, channel_group_id in enumerate(station_channel_group_ids):
                for channel in station.iter_channel_group(channel_group_id):
                    if np.all(detector.get_antenna_orientation(station.get_id(), channel.get_id()) == dominant_orientation):
                        good_channel_pair_ids[ind, 0] = channel.get_id()
                    else:
                        good_channel_pair_ids[ind, 1] = channel.get_id()

                # Check if dominant channel has been flagged
                channel = station.get_channel(good_channel_pair_ids[ind, 0])
                if channel.get_id() not in flagged_channels:
                    good_antennas_dict[station.get_id()].append(channel.get_id())

    return good_antennas_dict


class pipelineVisualizer:
    """
    Creates debug plots from the LOFAR pipeline - 
    This is the pipelineVisualizerTM for LOFAR.

    Any significant plots resulting from the pipeline
    should be added here by creating a function for them,
    and calling all functions sequentially.
    """

    def __init__(self):
        self.plots = None
        self.logger = logging.getLogger("NuRadioReco.pipelineVisualizer")


    def begin(self, logger_level=logging.NOTSET):
        self.logger.setLevel(logger_level)


    def plot_polarization(self, event, detector):
        """
        Plot the polarization of the electric field.
        This method calculates the stokes parameters of the pulse
        using get_stokes from framework.electric_field, and
        determines the polarization angle and degree, plotting
        them as arrows in the vxB and vxvxB plane.
        It estimates uncertainties by picking a pure noise value of
        stokes parameters, propagating through the angle and degree
        formulas and plotting them as arrows with reduced opacity.
        Author: Karen Terveer

        Parameters
        ----------
        event : Event object
            The event containing the stations and electric fields.
        detector : Detector object
            The detector object containing information about the detector.

        Returns
        -------
        fig_pol : matplotlib Figure object
            The generated figure object containing the polarization plot.
        """

        from NuRadioReco.framework.electric_field import get_stokes

        fig_pol, ax = plt.subplots(figsize=(8,7))

        triggered_station_ids = [
            station.get_id() for station in event.get_stations() if station.get_parameter(stationParameters.triggered)
        ]
        num_stations = len(triggered_station_ids)

        cmap = get_cmap('jet')  
        norm = Normalize(vmin=0, vmax=num_stations-1) 

        lora_core = event.get_hybrid_information().get_hybrid_shower("LORA").get_parameter(showerParameters.core)

        try:
            core = event.get_first_shower().get_parameter(showerParameters.core)
        except KeyError:
            self.logger.warning("No radio core found, using LORA core instead")
            core = lora_core

        for i, station in enumerate(event.get_stations()):
            if station.get_parameter(stationParameters.triggered):

                zenith = station.get_parameter(stationParameters.cr_zenith) / units.rad
                azimuth = station.get_parameter(stationParameters.cr_azimuth) / units.rad

                cs = radiotools.coordinatesystems.cstrafo(
                    zenith, azimuth, magnetic_field_vector=None, site="lofar"
                )

                efields = station.get_electric_fields()

                station_pos = detector.get_absolute_position(station.get_id())
                station_pos_vB = cs.transform_to_vxB_vxvxB(station_pos, core=core)[0]
                station_pos_vvB = cs.transform_to_vxB_vxvxB(station_pos, core=core)[1]

                ax.scatter(
                    station_pos_vB, station_pos_vvB,
                    color=cmap(norm(i)), s=20, label=f'Station CS{station.get_id():03d}'
                )

                for field in efields:

                    ids = field.get_channel_ids()
                    pos = station_pos + detector.get_relative_position(station.get_id(), ids[0])

                    pos_vB = cs.transform_to_vxB_vxvxB(pos, core=core)[0]
                    pos_vvB = cs.transform_to_vxB_vxvxB(pos, core=core)[1]

                    pulse_window_start, pulse_window_end = station.get_channel(ids[0]).get_parameter(channelParameters.signal_regions)
                    pulse_window_len = pulse_window_end - pulse_window_start

                    trace = field.get_trace()[:,pulse_window_start:pulse_window_end]

                    efield_trace_vxB_vxvxB = cs.transform_to_vxB_vxvxB(
                        cs.transform_from_onsky_to_ground(trace)
                    )

                    #get stokes parameters
                    stokes = get_stokes(*efield_trace_vxB_vxvxB[:2], window_samples=64)

                    stokes_max = np.argmax(stokes[0])

                    I = stokes[0,stokes_max]
                    Q = stokes[1,stokes_max]
                    U = stokes[2,stokes_max]
                    V = stokes[3,stokes_max]

                    # get stokes uncertainties by picking a pure noise value
                    I_sigma = stokes[0, stokes_max-pulse_window_len//4]
                    Q_sigma = stokes[1, stokes_max-pulse_window_len//4]
                    U_sigma = stokes[2, stokes_max-pulse_window_len//4]
                    V_sigma = stokes[3, stokes_max-pulse_window_len//4]

                    pol_angle = 0.5 * np.arctan2(U,Q)
                    pol_angle_sigma= np.sqrt((U_sigma**2*(0.5*Q/(U**2+Q**2))**2 + Q_sigma**2*(0.5*U/(U**2+Q**2))**2))

                    # if the polarization deviates from the vxB direction by more than 80 degrees,
                    # this could indicate something wrong with the antenna. Show a warning including
                    # the channel ids
                    if np.abs(0.5 * np.arctan2(U,Q)) > 80*np.pi/180:
                        self.logger.warning("strange polarization direction in channel group %s" % ids)

                    pol_degree= np.sqrt(U**2 + Q**2 + V**2) / I
                    pol_degree *= 7 # scale for better visibility

                    dx = pol_degree * np.cos(pol_angle)
                    dy = pol_degree * np.sin(pol_angle)

                    dx_sigma_plus = pol_degree * np.cos(pol_angle + pol_angle_sigma)
                    dy_sigma_plus = pol_degree * np.sin(pol_angle + pol_angle_sigma)

                    dx_sigma_minus = pol_degree * np.cos(pol_angle - pol_angle_sigma)
                    dy_sigma_minus = pol_degree * np.sin(pol_angle - pol_angle_sigma)

                    ax.arrow(
                        pos_vB, pos_vvB, dx_sigma_plus, dy_sigma_plus,
                        head_width=2, head_length=5,
                        fc=cmap(norm(i)), ec = cmap(norm(i)), alpha=0.5
                    )
                    ax.arrow(
                        pos_vB, pos_vvB, dx_sigma_minus, dy_sigma_minus,
                        head_width=2, head_length=5,
                        fc=cmap(norm(i)), ec = cmap(norm(i)), alpha=0.5
                    )
                    ax.arrow(
                        pos_vB, pos_vvB, dx, dy,
                        head_width=2, head_length=6,
                        fc=cmap(norm(i)), ec = cmap(norm(i))
                    )

        if np.any(core != lora_core):
            lora_vB = cs.transform_to_vxB_vxvxB(lora_core, core=core)[0]
            lora_vvB = cs.transform_to_vxB_vxvxB(lora_core, core=core)[1]
            ax.scatter(lora_vB, lora_vvB, color='tab:red', s=50, label='LORA core', marker = 'x')
            label = 'radio core'
        else:
            label = 'LORA core'

        ax.scatter([0], [0], color='black', s=50, label=label, marker = 'x')
        ax.legend()
        ax.set_xlabel('Direction along $v \\times B$ [m]')
        ax.set_ylabel('Direction along $v \\times (v \\times B)$ [m]')   

        return fig_pol
    
    def show_direction_plot(self, event):
        """
        Make a comparison of the reconstructed direction per station with the LORA direction.

        Author: Philipp Laub

        Parameters
        ----------
        event : Event object
            The event for which to show the final plots.
        
        Returns
        -------
        fig_dir : matplotlib Figure object
            The generated figure object containing the direction plot.
        """
        
        # plot reconstructed directions of all stations and compare to LORA in polar plot:
        fig_dir, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)

        triggered_station_ids = [
            station.get_id() for station in event.get_stations() if station.get_parameter(stationParameters.triggered)
        ]
        num_stations = len(triggered_station_ids)

        cmap = get_cmap('jet')
        norm = Normalize(vmin=0, vmax=num_stations-1) 

        for i, station in enumerate(event.get_stations()):
            if station.get_parameter(stationParameters.triggered):
                try:
                    zenith = station.get_parameter(stationParameters.cr_zenith)
                    azimuth = station.get_parameter(stationParameters.cr_azimuth)
                except KeyError:
                    self.logger.info(
                        f"Station CS{station.get_id():03d} does not have a reconstructed direction, "
                        f"so I am not plotting this one."
                    )
                    continue
                ax.plot(
                    azimuth, zenith,
                    label=f'Station CS{station.get_id():03d}',
                    marker='P',
                    markersize=7,
                    linestyle='',
                    color=cmap(norm(i))
                )

        ax.plot(
            event.get_hybrid_information().get_hybrid_shower("LORA").get_parameter(showerParameters.azimuth),
            event.get_hybrid_information().get_hybrid_shower("LORA").get_parameter(showerParameters.zenith),
            label='LORA',
            marker="X",
            markersize=7,
            linestyle='',
            color='black'
        )

        ax.legend()
        ax.set_title("Reconstructed arrival directions")

        return fig_dir

    def show_time_fluence_plot(self, event, detector, min_number_good_antennas=4):

        """
        Plot the antenna positions, marking arrival time by color and pseudofluence by markersize.
        The reconstructed arrival direction per station is indicated with an arrow.

        Author: Philipp Laub

        Parameters
        ----------
        event : Event object
            The event for which to show the final plots.
        detector : Detector object
            The detector for which to show the final plots.
        min_number_good_antennas : int, default=4
            The minimum number of good antennas that should be 
            present in a station to consider it for the fit.

        Returns
        -------
        fig_pol : matplotlib Figure object
            The generated figure object containing the polarization plot.
        """
        time = detector.get_detector_time().utc

        if time.mjd < 56266:
            self.logger.warning("Event was before Dec 1, 2012. The non-core station clocks might be off.")

        good_antennas_dict = check_for_good_ant(event, detector)

        fig_time, ax = plt.subplots(dpi=150, figsize=(8, 5))

        triggered_station_ids = [
            station.get_id() for station in event.get_stations() if station.get_parameter(stationParameters.triggered)
        ]
        num_stations = len(triggered_station_ids)

        cmap = get_cmap('jet')
        norm = Normalize(vmin=0, vmax=num_stations-1) 
        
        fluences = []
        positions = []
        SNRs = []
        for station in event.get_stations():
            if station.get_parameter(stationParameters.triggered):
                try:
                    azimuth = station.get_parameter(stationParameters.cr_azimuth)
                except KeyError:
                    self.logger.info(
                        f"Station CS{station.get_id():03d} does not have a reconstructed direction, "
                        f"so I am not plotting this one."
                    )
                    continue
                good_antennas = good_antennas_dict[station.get_id()]
                if len(good_antennas) >= min_number_good_antennas:
                    for antenna in good_antennas:
                        positions.append(
                            detector.get_relative_position(station.get_id(), antenna) + detector.get_absolute_position(station.get_id())
                        )
                        channel = station.get_channel(antenna)
                        SNRs.append(channel.get_parameter(channelParameters.SNR))
                        fluences.append(np.sum(np.square(channel.get_trace())))
                    station_pos = detector.get_absolute_position(station.get_id())
                    ax.quiver(station_pos[0], station_pos[1], 
                            np.cos(azimuth), np.sin(azimuth), 
                            color='black', 
                            scale=0.02, 
                            scale_units='xy', 
                            angles='uv',
                            width=0.005)
        
        timelags = []
        for station in event.get_stations():
            if station.get_parameter(stationParameters.triggered):
                good_antennas = good_antennas_dict[station.get_id()]
                if len(good_antennas) >= min_number_good_antennas:
                    for channel_id in good_antennas:
                        timelags.append(station.get_channel(channel_id).get_parameter(channelParameters.signal_time))

        for i, station in enumerate(event.get_stations()):
            # plot absolute station positions
            if station.get_parameter(stationParameters.triggered):
                station_pos = detector.get_absolute_position(station.get_id())
                ax.scatter(station_pos[0], station_pos[1], color=cmap(norm(i)), s=20, label=f'Station CS{station.get_id():03d}')

        timelags = np.array(timelags)
        timelags -= timelags[0]  # get timelags wrt 1st antenna
        # plot all locations and use arrival time for color and fluence for marker size and add a colorbar
        positions = np.array(positions)
        fluences = np.array(fluences)
        SNRs = np.array(SNRs)
        fluence_norm = Normalize(vmin=np.min(fluences), vmax=np.max(fluences))
        sc = ax.scatter(
            positions[:,0], 
            positions[:,1], 
            c=timelags, 
            s=15 * fluence_norm(fluences), 
            cmap='viridis',
            zorder=-1)
        
        ax.set_aspect('equal')
        plt.colorbar(sc, label='Relative arrival time [ns]', shrink=0.7)
        ax.set_xlabel('Meters east [m]')
        ax.set_ylabel('Meters north [m]')
        plt.legend()
        plt.title("Antenna positions and arrival time")

        return fig_time


    @register_run()
    def run(self, event, detector, save_dir='.', polarization=False, direction=False):
        """
        Produce pipeline plots for the given event.

        Parameters
        ----------
        event : Event object
            The event for which to visualize the pipeline.
        detector : Detector object
            The detector for which to visualize the pipeline.
        save_dir : str, optional
            The directory to save the plots to. Default is the 
            current directory.
        """

        plots = []
        if polarization:
            pol_plot = self.plot_polarization(event, detector)
            plots.append(pol_plot)
            pol_plot.savefig(f'{save_dir}/polarization_plot_{event.get_id()}.png')

        if direction:
            dir_plot = self.show_direction_plot(event)
            plots.append(dir_plot)
            dir_plot.savefig(f'{save_dir}/direction_plot_{event.get_id()}.png')

            time_fluence_plot = self.show_time_fluence_plot(event, detector)
            plots.append(time_fluence_plot)
            time_fluence_plot.savefig(f'{save_dir}/time_fluence_plot_{event.get_id()}.png')

        self.plots = [plot for plot in plots]

    def end(self):
        pass
