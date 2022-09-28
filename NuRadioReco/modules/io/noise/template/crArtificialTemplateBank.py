import os
import sys
import timeit
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
import astropy
import numpy as np
import NuRadioReco.modules.io.coreas.readCoREASShower
from NuRadioReco.detector.generic_detector import GenericDetector
import NuRadioReco.modules.efieldToVoltageConverter
import logging
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.channelGenericNoiseAdder
import crTemplateCorrelator
import NuRadioReco.modules.io.NuRadioRecoio
import json
import scipy
from NuRadioReco.modules.base import module
import datetime
from NuRadioReco.modules.io import NuRadioRecoio
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.channel import Channel
from NuRadioReco.framework.sim_station import SimStation
from NuRadioReco.framework.sim_channel import SimChannel
from NuRadioReco.framework.parameters import stationParameters
from NuRadioReco.framework.parameters import electricFieldParameters
from NuRadioReco.framework.electric_field import ElectricField
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import h5py
from tqdm import tqdm


logger = logging.getLogger('crArtificialTemplateBank')

class crArtificialTemplateBank:

    def __init__(self):
        self.__detector_file = None

        self.__template_run_id = None
        self.__template_channel_id = None
        self.__template_station_id = None

        self.__sampling_rate = None
        self.__template_sample_number = None

        self.__antenna_rotation = None
        self.__Efield_width = None
        self.__Efield_amplitudes = None
        self.__template_event_id = None

        self.__cr_zenith = None
        self.__cr_azimuth = None

        self.__logger_level = None

        self.__efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
        self.__hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
        self.__crTemplateCorrelator = crTemplateCorrelator.crTemplateCorrelator()
        self.__channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
        self.__readCoREASShower = NuRadioReco.modules.io.coreas.readCoREASShower.readCoREASShower()
        self.__channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

        self.__path_to_template_bank = None
        self.__path_to_used_simulations = None
        self.__path_to_all_simulations = None

        self.__correlation_cut_dic = None
        self.__trigger_cut_dic = None
        self.__denoised_hand_dic = None

    def begin(self, detector_file='', template_run_id=0, template_channel_id=0, template_station_id=101,
              path_template_bank='/home/henrichs/software/template_search/artificial_template_bank/', logger_level=logging.NOTSET):
        """
                begin method

                initialize crArtificialTemplateBank module

                Parameters
                ----------
                detector_file: string
                    path to the detector file used for the template set
                template_run_id: int
                    number of the run of the artificial templates
                template_channel_id:
                    channel number of the artificial templates
                template_station_id:
                    station nuber of the artificial templates
                path_template_bank: string
                    path to a folder where the results and all information are stored
                logger_level: string or logging variable
                    Set verbosity level for logger (default: logging.NOTSET)
                """

        self.__detector_file = detector_file

        self.__logger_level = logger_level
        logger.setLevel(self.__logger_level)

        self.__template_run_id = template_run_id
        self.__template_channel_id = template_channel_id
        self.__template_station_id = template_station_id

        # set up the efield to voltage converter
        self.__efieldToVoltageConverter.begin(debug=False)

        self.__path_to_template_bank = path_template_bank
        self.__path_to_used_simulations = path_template_bank + 'used_simulations_cr_greenland.csv'
        self.__path_to_all_simulations = path_template_bank + 'all_simulations_cr_greenland.csv'

    def set_parameter_templates(self, template_event_id, Efield_width=[5,3,8], antenna_rotation=[160, 160 ,160], Efield_amplitudes=[-0.2,0.8], cr_zenith=[55, 55, 55], cr_azimuth=[0, 0, 0], sampling_rate=3.2*units.GHz, number_of_samples=2048):
        """
        set_parameter_templates method

        sets the parameter to create the template set

        Parameters
        ----------
        Efield_width: list of int
            number of data points for the width of the gaussian function used to create the Efield
        antenna_rotation: list of int
            rotation angle of the LPDA
        Efield_ratio: list
            array with the amplitudes of the Efield components [E_theta, E_phi]
        template_event_id: int
            event number of the main artificial template
        cr_zenith: list of int
            zenith angle of the cr for the template
        cr_azimuth: list of int
            azimuth angle of the cr for the template
        sampling_rate: float
            smapling rate used to build the template
        number_of_samples: int
            number of samples used for the trace
        """

        self.__Efield_width = Efield_width
        self.__Efield_amplitudes = Efield_amplitudes
        self.__antenna_rotation = antenna_rotation

        self.__template_event_id = template_event_id

        self.__sampling_rate = sampling_rate
        self.__template_sample_number = number_of_samples

        self.__cr_zenith = cr_zenith
        self.__cr_azimuth = cr_azimuth

    def get_templates(self):
        """
        get_templates

        returns the Efield trace of the artificial templates

        Returns
        ----------
        efield: list of EField objects
            efield trace of the artificial template [E_theta, E_phi]
        """

        # if no parameters are set, the standard parameters are used
        if self.__Efield_width is None:
            self.set_parameter_templates(0)

        template_events = []
        # loop over the different antenna rotation angles:
        for e_width, antrot, cr_zen, cr_az in zip(self.__Efield_width, self.__antenna_rotation, self.__cr_zenith, self.__cr_azimuth):
            # create the detector
            det_temp = GenericDetector(json_filename=self.__detector_file, antenna_by_depth=False, create_new=True)
            det_temp.update(datetime.datetime(2025, 10, 1))
            det_temp.get_channel(101, 0)['ant_rotation_phi'] = antrot

            station_time = datetime.datetime(2025, 10, 1)

            temp_evt = create_Efield(det_temp, self.__template_run_id, self.__template_event_id, self.__template_channel_id, self.__template_station_id, station_time, self.__template_sample_number, e_width, self.__Efield_amplitudes[1], self.__Efield_amplitudes[0], cr_zen, cr_az, self.__sampling_rate)

            self.__efieldToVoltageConverter.run(temp_evt, temp_evt.get_station(self.__template_station_id), det_temp)
            self.__hardwareResponseIncorporator.run(temp_evt, temp_evt.get_station(self.__template_station_id), det_temp, sim_to_data=True)

            template_events.append(temp_evt)

        return template_events

    def create_position_lookup_files(self, zenith_angles=np.arange(5,80,5), controlPlot=False):
        """
        create_position_lookup_files

        creates look up files for the station positions (depends on the zenith angle)

        Parameters
        ----------
        zenith_angles: list of int
            zenith angles for which the lookup files are created
        controlPlot: boolean
            if true, a plot is shown to control if function is working correctly

        Returns
        ----------
        """
        path = np.loadtxt(self.__path_to_all_simulations, dtype='str', usecols=(0), skiprows=1, delimiter=",", unpack=True)
        energy, azimuth, zenith, xmax = np.loadtxt(self.__path_to_all_simulations, dtype='float,float,float,float', usecols=(1, 2, 3, 4), skiprows=1,
                                                   delimiter=",", unpack=True)

        # it was checked that the energy, xmax and antenna rotation have no influence on the station positions
        for zen_ang in zenith_angles:
            logger.info(f'Creating the lookup file for zenith angle: {zen_ang}')
            used_sim_path = ''
            # energy is used to choose one of the simulation files (xmax could have also been used)
            for i, p in enumerate(path):
                if energy[i] == 18.0 and zenith[i] == zen_ang:
                    used_sim_path = p
                    break

            # create a detector
            det_look_up = GenericDetector(json_filename=self.__detector_file, antenna_by_depth=False, create_new=True)
            det_look_up.update(datetime.datetime(2025, 10, 1))

            # get the positions of the stations
            position = []
            self.__readCoREASShower.begin([used_sim_path], det_look_up)
            for evt, det1 in self.__readCoREASShower.run():
                for sta in evt.get_stations():
                    position.append(det_look_up.get_absolute_position(sta.get_id()).tolist())

            # extract the x and y position
            x_pos = np.asarray(position)[:, 0]
            y_pos = np.asarray(position)[:, 1]

            # convert the cartesian values to polar coordinates
            r_pos, phi_pos = cartesian_to_polar(x_pos, y_pos)

            # put the positions in a dictionary
            pos_dic = {}
            for i in range(len(position)):
                pos_dic[str(i)] = [r_pos[i], phi_pos[i]*(180/np.pi)]

            # save the dictionary
            json_file = json.dumps(pos_dic)
            f = open(self.__path_to_template_bank + f"station_position_lookup_zenith_angle_{zen_ang}.json", "w")
            f.write(json_file)
            f.close()

            # if true, a plot of the CoREAS start will be plotted
            if controlPlot:
                plt.polar(phi_pos, r_pos, ls="none", marker="x")
                plt.show()

    def create_correlation_cut_file(self, cut_value, correlation_method, controlPlot=False):
        """
        create_correlation_cut_file

        creates a file with all stations that survived the correlation cut

        Parameters
        ----------
        cut_value: float
            all correlation values smaller than this value will be neglected
        correlation_method: string
            determines which correlation method will be used for the calculation
            options: 'scipy', 'window'
        controlPlot: boolean
            if true, a plot is shown to control if function is working correctly

        Returns
        ----------
        """
        logger.info(f'Calculate the correlation cut with a cutting value of: chi = {cut_value}')

        # load information from all available simulations
        path = np.loadtxt(self.__path_to_used_simulations, dtype='str', usecols=(0), skiprows=1, delimiter=",", unpack=True)
        zenith, xmax_bin = np.loadtxt(self.__path_to_used_simulations, dtype='float, float', usecols=(3, 4), skiprows=1, delimiter=",", unpack=True)
        # load information from all available simulations
        path_comp = np.loadtxt(self.__path_to_all_simulations, dtype='str', usecols=(0), skiprows=1, delimiter=",", unpack=True)
        zenith_comp, xmax_comp = np.loadtxt(self.__path_to_all_simulations, dtype='float,float', usecols=(3, 5), skiprows=1, delimiter=",", unpack=True)

        save_dic = {}
        # get for each simulation a second simulation with the same parameters but slightly different xmax
        for p, zen, xmb in zip(path, zenith, xmax_bin):
            # current_zenith_ang = zen
            # current_xmax_bin = xmb

            # get all paths and xmax from the pool of all simulations which have the current zenith angle
            path_help = []
            xmax_help = []
            for j, z_comp in enumerate(zenith_comp):
                if z_comp == zen:
                    path_help.append(path_comp[j])
                    xmax_help.append(xmax_comp[j])

            # calculate the difference between the current xmax bin and the selected xmax values from above
            diff_xmax = np.asarray(xmax_help) - xmb
            # if the miniaml difference if for the current simulation we are looking at, skip this on by setting the difference to a large value (want to find a different simulation for comparison)
            if path_help[np.where(np.abs(diff_xmax) == min(np.abs(diff_xmax)))[0][0]] == p:
                diff_xmax[np.where(np.abs(diff_xmax) == min(np.abs(diff_xmax)))[0][0]] = 10000
                # get the simulation where the xmax difference is minimal
                path_sim = path_help[np.where(np.abs(diff_xmax) == min(np.abs(diff_xmax)))[0][0]]
            else:
                # due to rounding it can be that the smallest distance to the bin center is not the current simulation (should only happend if two simulations with a Xmax close to the bin center exists)
                path_sim = path_help[np.where(np.abs(diff_xmax) == min(np.abs(diff_xmax)))[0][0]]

            # create the detector. For the antenna rotation, the value from the json file is used
            det_corr = GenericDetector(json_filename=self.__detector_file, antenna_by_depth=False, create_new=True)
            det_corr.update(datetime.datetime(2025, 10, 1))
            rot = det_corr.get_channel(101, 0)['ant_rotation_phi']
            logger.info(f' For the antenna rotation the value specified in the detetcor file will be used (ant_rot = {rot})')

            # look at both siumlations
            corsika1 = h5py.File(p, 'r') # current sinmulation
            corsika2 = h5py.File(path_sim, 'r') # simulation found with very close parameter to the current one
            logger.info('### information about the two simulations ###')
            logger.info(f'Xmax: {corsika1["CoREAS"].attrs["DepthOfShowerMaximum"]} and {corsika2["CoREAS"].attrs["DepthOfShowerMaximum"]}')
            logger.info(f'zenith angle: {np.round(corsika1["inputs"].attrs["THETAP"][0], 1)} and {np.round(corsika2["inputs"].attrs["THETAP"][0], 1)}')

            # open both simulations with readCoREASShower
            self.__readCoREASShower.begin([p, path_sim], det_corr)

            # apply efield_to_voltage and hardware response to be able to get traces
            evts = []
            for evt, det1 in self.__readCoREASShower.run():
                for sta in evt.get_stations():
                    self.__efieldToVoltageConverter.run(evt, sta, det_corr)
                    self.__hardwareResponseIncorporator.run(evt, sta, det_corr, sim_to_data=True)
                evts.append(evt)

            # load the position information for the current zenith angle
            lookup_station_pos_file = open(self.__path_to_template_bank + f"station_position_lookup_zenith_angle_{int(zen)}.json")
            lookup_station_pos = json.load(lookup_station_pos_file)

            # calculate the correlation of both simulations
            corr = []
            r_pos = []
            phi_pos = []
            for num in evts[0].get_station_ids():
                if correlation_method == 'scipy':
                    corr.append(max(abs(self.__crTemplateCorrelator.correlation_scipy(evts[0], evts[1], 0, 0, num, num, showPlot=False))))
                elif correlation_method == 'window':
                    corr.append(max(abs(self.__crTemplateCorrelator.correlation_scan_single_spacing_matrix_variable_window(evts[0], evts[1], 0, 0, num, num, 200 * units.ns, showPlot=False))))
                else:
                    logger.error('The chosen correlation method is not known.')
                    sys.exit(0)
                r_pos.append(lookup_station_pos[str(num)][0])
                phi_pos.append(lookup_station_pos[str(num)][1] * (np.pi / 180))

            # if controlPlot true, plot a visualization of the correlation plot
            if controlPlot:
                fig = plt.figure(2)
                ax = fig.add_subplot(121, projection='polar')
                sc = ax.scatter(phi_pos, r_pos, c=corr)
                cbar = plt.colorbar(sc)
                cbar.set_label(r'correlation $\chi$')

                ax1 = fig.add_subplot(122)
                ax1.plot(r_pos, corr, ls="none", marker=".")
                ax1.axhline(cut_value, color="tab:red", ls="--")
                ax1.set_ylabel(r'correlation $\chi$')
                ax1.set_xlabel('radius')
                plt.show()

            # apply cut and save to dictionary
            survive_station = []
            for ii, c in enumerate(corr):
                if c > cut_value:
                    survive_station.append(ii)

            save_dic[f'zenith{int(zen)}_xmax{int(xmb)}'] = survive_station

        # save the survived stations in a json file
        json_file = json.dumps(save_dic)
        f = open(self.__path_to_template_bank + "/" + f"station_numbers_after_corr_cut_{cut_value}.json", "w")
        f.write(json_file)
        f.close()

    def set_correlation_cut(self, path_to_corr_cut_file='', cut_value=0.6):
        """
        set_correlation_cut method

        sets the load the correlation cut file

        Parameters
        ----------
        path_to_corr_cut_file: string
            path to the correlation cut file, if '', the path_to_template_bank will be used
        cut_value: float
            cut value of the correlation cut

        Returns
        ----------
        """

        if path_to_corr_cut_file == '':
            path_to_corr_cut_file = self.__path_to_template_bank

        corr_cut_file = open(path_to_corr_cut_file + f'station_numbers_after_corr_cut_{cut_value}.json')
        corr_cut_dic = json.load(corr_cut_file)

        self.__correlation_cut_dic = corr_cut_dic
        logger.info('The correlation cut is applied.')

    def __rescale_electric_fields(self, path, multiplicator, detector):
        """
        __rescale_electric_fields method

        helper function to rescale the electric field

        Parameters
        ----------
        path: string
            path to the simulation
        multiplicator: int
            factor for what the electric field is scaled
        detector: object
            NuRadioReco detector, needed for opening the simulations

        Returns
        ----------
        rescaled event
            an event including the rescaled electric field
        """
        rescaled_event = []

        self.__readCoREASShower.begin([path], detector)
        for evt, det1 in self.__readCoREASShower.run():
            for sta in tqdm(evt.get_stations()):
                sim_station = sta.get_sim_station()
                efield = sim_station.get_electric_fields()
                sim_station.set_electric_fields(efield * multiplicator)
                cr_energy = sim_station.get_parameter(stationParameters.cr_energy)
                sim_station.set_parameter(stationParameters.cr_energy, cr_energy * multiplicator)

                self.__efieldToVoltageConverter.run(evt, sta, detector)

                self.__channelResampler.run(evt, sta, detector, 3.2 * units.GHz)

                self.__hardwareResponseIncorporator.run(evt, sta, detector, sim_to_data=True)

            rescaled_event.append(evt)

        return rescaled_event

    def __apply_an_envelope_trigger(self, event, trigger_cutoff, detector):
        """
        __apply_an_envelope_trigger method

        helper function to apply an 'envelope trigger'

        Parameters
        ----------
        event: object
            event on which the trigger should be applied
        trigger_cutoff: int
            amplitude value of the trigger
        detector: object
            NuRadioReco detector, needed for opening the bandpass filter

        Returns
        ----------
        rescaled event
            an event including the rescaled electric field
        """
        survive_stations = []
        for stat in event.get_stations():
            # create a 'envelope trigger'
            self.__channelBandPassFilter.run(event, stat, detector, passband=[80 * units.MHz, 180 * units.MHz])
            envelope = stat.get_channel(0).get_hilbert_envelope() / units.mV
            amplitude = np.max(envelope)

            if amplitude > trigger_cutoff:
                survive_stations.append(stat.get_id())

        return survive_stations

    def create_trigger_cut_file(self, cut_value=30, antenna_rotations=[120, 130, 140, 150, 160, 170]):
        """
        create_trigger_cut_file method

        creates a file with all stations that survies the trigger cut

        Parameters
        ----------
        cut_value: int
            used trigger threshold (in mV)
        antenna_rotations: list of int
            used antenna rotation angles

        Return
        ----------
        """
        # get the fitting simluations
        path = np.loadtxt(self.__path_to_all_simulations, dtype='str', usecols=(0), skiprows=1, delimiter=",", unpack=True)
        energy, azimuth, zenith, xmax = np.loadtxt(self.__path_to_all_simulations, dtype='float,float,float,float', usecols=(1, 2, 3, 4), skiprows=1, delimiter=",", unpack=True)

        for ant_rot in antenna_rotations:
            # create the detector
            det_tri_cut = GenericDetector(json_filename=self.__detector_file, antenna_by_depth=False, create_new=True)
            det_tri_cut.update(datetime.datetime(2025, 10, 1))
            det_tri_cut.get_channel(101, 0)['ant_rotation_phi'] = ant_rot

            save_dic_antrot = {}
            for used_xmax in [552, 603, 654, 705, 757, 808]:
                logger.info(f'########### xmax {used_xmax} ###############')
                # get the scaling factor the electric fields for the fitting simulations
                paths = []
                zeniths = []
                multiplicator = []
                for zen in np.arange(5, 80, 5):
                    for p, e, z, x in zip(path, energy, zenith, xmax):
                        if x == used_xmax and z == zen and (e == 18.5 or e == 17.5 or e == 16.5):
                            paths.append(p)
                            zeniths.append(z)
                            if e == 16.5:
                                multiplicator.append(10 * 10)
                            elif e == 17.5:
                                multiplicator.append(10)
                            elif e == 18.5:
                                multiplicator.append(1)
                            break

                # rescale the electric fields
                events = []
                for ip, pa in enumerate(paths):
                    rescaled_event = self.__rescale_electric_fields(pa, multiplicator[ip], det_tri_cut)[0]
                    events.append(rescaled_event)

                # apply the 'trigger'
                trigger_cutoff = cut_value
                for ie, event in enumerate(events):
                    survive_stations = self.__apply_an_envelope_trigger(event, trigger_cutoff, det_tri_cut)

                    # put the survived stations in a dictionary
                    save_dic_antrot[f'zenith{int(zeniths[ie])}_xmax{int(used_xmax)}'] = survive_stations

            # extrapolating the missing simulations for the most agressive xmax value (552)
            print('extrapolating xmax')
            save_dic_extrapolated_xmax = {}
            # trigg_cut_surv_dic = save_dic

            # most extreme xmax (chosen to be most conservative)
            xmax_low = 552

            # find the simulations (zenith angles) which already in the save_dic_antrot
            zenith_trig_cut = []
            for key in save_dic_antrot.keys():
                if f'xmax{int(xmax_low)}' in key:
                    zenith_trig_cut.append(int(key[len('zenith'):key.find('_xmax')]))

            for zen in tqdm(np.arange(5, 80, 5)):
                # if this simulations is already in save_dic_antro, just copy it over (no extrapolation needed)
                if f'zenith{zen}_xmax{xmax_low}' in save_dic_antrot.keys():
                    save_dic_extrapolated_xmax[f'zenith{zen}_xmax{xmax_low}'] = save_dic_antrot[f'zenith{zen}_xmax{xmax_low}']
                else:
                    # need to extrapolate the missing simulations
                    # find next zenith angle which has a calculated trigger cut
                    zen_up = zen
                    zen_down = zen
                    while zen_up not in zenith_trig_cut:
                        #print('test')
                        zen_up = zen_up + 5
                    while zen_down not in zenith_trig_cut:
                        #print('test')
                        zen_down = zen_down - 5

                    # select the zenith angle were the fewest stations are cut -> most conservative cut
                    selected_zenith = 0
                    if len(save_dic_antrot[f'zenith{zen_up}_xmax{xmax_low}']) > len(save_dic_antrot[f'zenith{zen_down}_xmax{xmax_low}']):
                        selected_zenith = zen_up
                    else:
                        selected_zenith = zen_down

                    # count how many stations per CoREAS arm are cut
                    n_cut = get_number_of_cut_stations_per_CoREAS_arm(self.__path_to_template_bank + f'station_position_lookup_zenith_angle_{int(selected_zenith)}.json', save_dic_antrot, selected_zenith, xmax_low)
                    #print('Hello')
                    # cut the same number per arm on the missing simulation
                    # load the station positions of the actual zenith angle
                    lookup_station_pos_file_zen = open(self.__path_to_template_bank + f'/station_position_lookup_zenith_angle_{int(zen)}.json')
                    lookup_station_pos_zen = json.load(lookup_station_pos_file_zen)

                    # get the station positions and CoREAS arm
                    position_zen = []
                    for key in lookup_station_pos_zen:
                        position_zen.append(lookup_station_pos_zen[key])
                    arm_pos_zen, arm_i_zen = get_CoREAS_arms(position_zen, polar_coord=True, controll_plot=False)
                    arm_r_sort_zen, arm_i_sort_zen = sort_CoREAS_arms(arm_pos_zen, arm_i_zen)

                    # cut away the stations for the outer radii
                    surv_stations = np.array([])
                    for iarm, arm_zen in enumerate(arm_i_sort_zen):
                        surv_stations = np.append(surv_stations, np.asarray(arm_zen[:-n_cut[iarm]]))

                    save_dic_extrapolated_xmax[f'zenith{int(zen)}_xmax{xmax_low}'] = list(surv_stations)

            # copy all simulations which are not already in save_dic_extrapolated_xmax
            for key in save_dic_antrot.keys():
                if key not in save_dic_extrapolated_xmax.keys():
                    save_dic_extrapolated_xmax[key] = save_dic_antrot[key]

            # extrapolating the missing simulations for all other xmax values (meaning holding xmax and extrapolate over zenith angle)
            print('extrapolating  zenith')
            save_dic_extrapolated_zenith = {}

            # trigg_cut_surv_dic = save_dic_extrapolated_xmax

            #zenith, xmax = np.loadtxt(self.__path_to_all_simulations, dtype='float,float', usecols=(3, 4), skiprows=1, delimiter=",", unpack=True)

            for zen in tqdm(np.arange(5, 80, 5)):
                # find for each zenith angles all possible xmax which needs to be in the trigger cut file (all xmax we have a simulation for)
                poss_xmax = []
                for z, x in zip(zenith, xmax):
                    if z == zen and x not in poss_xmax and x != 0.0:
                        poss_xmax.append(x)
                poss_xmax.sort()

                # get all the xmax which are already in the trigger cut dic
                xmax_trigg_cut = []
                for key in save_dic_extrapolated_xmax.keys():
                    if f'zenith{zen}' in key:
                        xmax_trigg_cut.append(int(key[key.find('xmax') + len('xmax'):]))

                # get the next xmax for which there is a calculated trigger cut
                for ix, xm in enumerate(poss_xmax):
                    # if the simulation already exists, just copy it to the new dictionary
                    if f'zenith{int(zen)}_xmax{int(xm)}' in save_dic_extrapolated_xmax.keys():
                        save_dic_extrapolated_zenith[f'zenith{int(zen)}_xmax{int(xm)}'] = save_dic_extrapolated_xmax[f'zenith{int(zen)}_xmax{int(xm)}']
                    else:
                        # find the next xmax value which is in the trigger cut
                        x_up = xm
                        x_down = xm
                        count_up = ix
                        while x_up not in xmax_trigg_cut and x_up != np.max(poss_xmax):
                            x_up = poss_xmax[count_up]
                            count_up = count_up + 1
                        count_down = ix
                        while x_down not in xmax_trigg_cut and x_down != np.min(poss_xmax):
                            x_down = poss_xmax[count_down]
                            count_down = count_down - 1

                        # select the xmax with the fewest cut stations -> most conservative cut
                        selected_xmax = 0
                        if x_up not in xmax_trigg_cut:
                            selected_xmax = x_down
                        elif x_down not in xmax_trigg_cut:
                            selected_xmax = 552
                        else:
                            if len(save_dic_extrapolated_xmax[f'zenith{int(zen)}_xmax{int(x_up)}']) > len(save_dic_extrapolated_xmax[f'zenith{int(zen)}_xmax{int(x_down)}']):
                                selected_xmax = x_up
                            else:
                                selected_xmax = x_down

                        selected_xmax = int(selected_xmax)

                        # calculate how many of the outer stations per arm are cut
                        n_cut = get_number_of_cut_stations_per_CoREAS_arm(self.__path_to_template_bank + f'station_position_lookup_zenith_angle_{int(zen)}.json', save_dic_extrapolated_xmax, zen, selected_xmax)

                        # cut away the stations for the outer radii (same zenith angle, so I don't need to load the positions again)
                        # load the station positions
                        lookup_station_pos_file = open(self.__path_to_template_bank + f'/station_position_lookup_zenith_angle_{int(zen)}.json')
                        lookup_station_pos = json.load(lookup_station_pos_file)

                        # get the station positions and CoREAS arm
                        position = []
                        for key in lookup_station_pos:
                            position.append(lookup_station_pos[key])
                        arm_pos, arm_i = get_CoREAS_arms(position, polar_coord=True, controll_plot=False)
                        arm_r_sort, arm_i_sort = sort_CoREAS_arms(arm_pos, arm_i)

                        # cut the same number of outer stations per CoREAS arm as extrapolated
                        surv_stations = np.array([])
                        for iarm, arm_x in enumerate(arm_i_sort):
                            surv_stations = np.append(surv_stations, np.asarray(arm_x[:-n_cut[iarm]]))

                        # save the cut stations in the dic
                        save_dic_extrapolated_zenith[f'zenith{int(zen)}_xmax{int(xm)}'] = list(surv_stations)

            # save the final dictionary
            json_file = json.dumps(save_dic_extrapolated_zenith)
            f = open(self.__path_to_template_bank + f"station_numbers_after_trigger_cut_antrot{ant_rot}.json", "w")
            f.write(json_file)
            f.close()

    def set_trigger_cut(self, path_to_trigger_cut_file='', antenna_rotations=[120, 130, 140, 150, 160, 170]):
        """
        set_trigger_cut method

        sets and loads the trigger cut files

        Parameters
        ----------
        path_to_trigger_cut_file: string
            path to the trigger cut file, if '' the path_to_template_bank will be used
        antenna_rotations: list
            list of the antenna rotations used
        Returns
        ----------
        """

        if path_to_trigger_cut_file == '':
            path_to_trigger_cut_file = self.__path_to_template_bank

        trig_cut_dic = {}
        for ant_rot in antenna_rotations:
            path = path_to_trigger_cut_file + f'station_numbers_after_trigger_cut_antrot{ant_rot}.json'
            trig_cut_file = open(path)
            surv_sta_trig_cut = json.load(trig_cut_file)

            trig_cut_dic[f'ant_rot{ant_rot}'] = surv_sta_trig_cut

        # set the trigger cut file
        self.__trigger_cut_dic = trig_cut_dic
        logger.info('The trigger cut is applied.')

    def set_denoised_by_hand(self, path_to_hand_denoised_file=''):
        """
        set_denoised_by_hand method

        sets and loads the hand denoised cut dict

        Parameters
        ----------
        path_to_hand_denoised_file: string
            path to the hand denoised file, if '' the path_to_template_bank will be used
        Returns
        ----------
        """
        if path_to_hand_denoised_file == '':
            path_to_hand_denoised_file = self.__path_to_template_bank

        denoised_file = open(path_to_hand_denoised_file + f'stations_rejected_by_hand.json')
        denoised_dic = json.load(denoised_file)

        self.__denoised_hand_dic = denoised_dic
        logger.info('The cut rejecting the stations filtered out by hand is applied.')

    def __load_simulation_event(self, path, det_input, sampling_rate):
        """
        __load_simulation_event

        helper function to load simulation events

        Parameters
        ----------
        path: string
            path to the simulation
        detector: object
            NuRadioReco detector, needed for opening the simulations
        sampling_rate: float
            sampling rate used for the template and the simulation

        Returns
        ----------
        event
            an event including the simulation
        """
        self.__readCoREASShower.begin([path], det_input)
        events = []
        for evt, det1 in self.__readCoREASShower.run():
            for sta in evt.get_stations():
                self.__efieldToVoltageConverter.run(evt, sta, det_input)

                self.__channelResampler.run(evt, sta, det_input, sampling_rate)

                self.__hardwareResponseIncorporator.run(evt, sta, det_input, sim_to_data=True)

            events.append(evt)

        return events

    def perform_completeness_scan(self, templates=[5, 3, 8], method='window',correlation_cut=True, trigger_cut=True, trigger_cut_antenna_rotations=[120,130,140,150,160,170], denoised_hand=True, template_antrot=[160,160,160], template_zenith=[55,55,55], sampling_rate=3.2*units.GHz):
        """
        perform_completeness_scan method

        peform a scan to check if all simulations are found by the templates

        Parameters
        ----------
        templates: list
            list of the efield width used for the templates
        method: string
            method used to calculate the correlation
                possibilities: 'window', 'scipy'
        correlation_cut: boolean
            if true the correlation cut will be applied
        trigger_cut: boolean
            if true the trigger cut will be applied
        trigger_cut_antenna_rotations: list
            the antenna rotations for which the trigger cut will be loaded
        denoised_hand: boolean
            if true the stations rejected by hand are not considered
        template_antrot: list
            antenna rotations used for the template creation
        template_zenith: list
            cr zenith angles used to create the templates
        sampling_rate: float
            sampling rate used for the template and the simulation
        Returns
        ----------
        """

        found_dic = {}
        for ant_rot in trigger_cut_antenna_rotations:
            logger.info(f'completeness scan: ant_rot = {ant_rot}, method = {method}, correlation cut = {correlation_cut}, trigger cut = {trigger_cut}, denoised per hand = {denoised_hand}')

            # load and set the different cuts
            if correlation_cut:
                self.set_correlation_cut()
            if trigger_cut:
                self.set_trigger_cut(antenna_rotations=trigger_cut_antenna_rotations)
            if denoised_hand:
                self.set_denoised_by_hand()

            # load the detector
            det_scan = GenericDetector(json_filename=self.__detector_file, antenna_by_depth=False, create_new=True)
            det_scan.update(datetime.datetime(2025, 10, 1))
            det_scan.get_channel(101, 0)['ant_rotation_phi'] = ant_rot

            # load the path and parameters of the simulations from the parameter space
            path_to_sim = np.loadtxt(self.__path_to_used_simulations, dtype='str', usecols=(0), skiprows=1, delimiter=",", unpack=True)
            zenith, xmax = np.loadtxt(self.__path_to_used_simulations, dtype='float,float', usecols=(3, 4), skiprows=1, delimiter=",", unpack=True)

            # start the correlator object
            self.__crTemplateCorrelator.begin([], [], logger_level='ERROR')

            # load the templates
            self.set_parameter_templates(template_event_id=0, Efield_width=templates, antenna_rotation=template_antrot, sampling_rate=sampling_rate, cr_zenith=template_zenith)
            temp_events = self.get_templates()

            count_found = 0
            count_not_found = 0
            # max_corr = []
            not_found = []
            # go through each simulation in the parameter space and see if all stations are found
            for p, zen, x in tqdm(zip(path_to_sim, zenith, xmax)):
                # restric the parameter space to zenith angles smaller than 80°
                upper_zenith = 80
                if zen < upper_zenith:
                    # load the simulation
                    sim_evt = self.__load_simulation_event(p, det_scan, sampling_rate=sampling_rate)[0]

                    # get the survived (not survived) stations after the cuts:
                    if correlation_cut:
                        surv_corr_cut = self.__correlation_cut_dic[f'zenith{int(zen)}_xmax{int(x)}']
                    else:
                        surv_corr_cut = []
                    if trigger_cut:
                        surv_trig_cut = self.__trigger_cut_dic[f'ant_rot{int(ant_rot)}'][f'zenith{int(zen)}_xmax{int(x)}']
                    else:
                        surv_trig_cut = []
                    if denoised_hand:
                        if f'zenith{int(zen)}_xmax{int(x)}_antrot{int(ant_rot)}' in list(self.__denoised_hand_dic.keys()):
                            failed_denoise = self.__denoised_hand_dic[f'zenith{int(zen)}_xmax{int(x)}_antrot{int(ant_rot)}']
                        else:
                            failed_denoise = []
                    else:
                        failed_denoise = []

                    # go through each station and see if the stations is found
                    for sta in sim_evt.get_stations():
                        # only look at stations not cut
                        if sta.get_id() in surv_corr_cut and sta.get_id() in surv_trig_cut and sta.get_id() not in failed_denoise:
                            # check one template after each other and only if the template does not find the simulation, then try the next one
                            found = False
                            counter = 0
                            while not found and counter < len(templates):
                                temp_e = temp_events[counter]
                                # calulate the correlation
                                if method == 'scipy':
                                    correlation = self.__crTemplateCorrelator.correlation_scipy(sim_evt, temp_e, 0, 0, sta.get_id(), 101, showPlot=False)
                                elif method == 'window':
                                    correlation = self.__crTemplateCorrelator.correlation_scan_single_spacing_matrix_variable_window(sim_evt, temp_e, 0, 0, sta.get_id(), 101, 200 * units.ns, showPlot=False)
                                max_corr = np.max(np.abs(correlation))

                                # check if the correlation is larger than 0.8
                                if max_corr > 0.8:
                                    # if founnd: save the information into a dict
                                    count_found = count_found + 1
                                    found_dic[f'zenith{zen}_xmax{x}_antrot{ant_rot}_sta{sta.get_id()}'] = [templates[counter], max_corr]
                                    found = True
                                else:
                                    counter = counter + 1

                            # if not found save -100 for template and correlation value
                            if not found:
                                count_not_found = + count_not_found + 1
                                found_dic[f'zenith{zen}_xmax{x}_antrot{ant_rot}_sta{sta.get_id()}'] = [-100, -100]
                                logger.info(f'zenith: {zen}, xmax: {x}, station: {sta.get_id()}')

            logger.info(f'Number of simulations found: {count_found}')
            logger.info(f'Total number of simulations: {count_not_found + count_found}')

        # save the result as a json file
        json_file = json.dumps(found_dic)
        filename = 'completeness_scan_templates_'
        for temp in templates:
            filename = filename + str(temp)
        filename = filename + f"_method_{method}_triggercut_{trigger_cut}_corrcut_{correlation_cut}_handdenoised_{denoised_hand}.json"
        f = open(self.__path_to_template_bank + filename,"w")
        f.write(json_file)
        f.close()

    def perform_parameter_space_scan_single_template(self, template, method='window',correlation_cut=True, trigger_cut=True, trigger_cut_antenna_rotations=[120,130,140,150,160,170], denoised_hand=True, template_antrot=160, template_zenith=55, sampling_rate=3.2*units.GHz):
        """
        perform_parameter_space_scan_single_template method

        perform a parameter scan with just a single template

        Parameters
        ----------
        template: int
            efield width used for the template
        method: string
            method used to calculate the correlation
                possibilities: 'window', 'scipy'
        correlation_cut: boolean
            if true the correlation cut will be applied
        trigger_cut: boolean
            if true the trigger cut will be applied
        trigger_cut_antenna_rotations: list
            the antenna rotations for which the trigger cut will be loaded
        denoised_hand: boolean
            if true the stations rejected by hand are not considered
        template_antrot: int
            antenna rotation used for the template creation
        template_zenith: int
            cr zenith angle used to create the template
        sampling_rate: float
            sampling rate used for the template and the simulation
        Returns
        ----------
        """
        save_dic = {}
        for ant_rot in trigger_cut_antenna_rotations:
            logger.info(f'parameter scan: template = {template}, ant_rot = {ant_rot}, method = {method}, correlation cut = {correlation_cut}, trigger cut = {trigger_cut}, denoised per hand = {denoised_hand}')

            # load and set the different cuts
            if correlation_cut:
                self.set_correlation_cut()
            if trigger_cut:
                self.set_trigger_cut(antenna_rotations=trigger_cut_antenna_rotations)
            if denoised_hand:
                self.set_denoised_by_hand()

            # set up the detector
            det_scan = GenericDetector(json_filename=self.__detector_file, antenna_by_depth=False, create_new=True)
            det_scan.update(datetime.datetime(2025, 10, 1))
            det_scan.get_channel(101, 0)['ant_rotation_phi'] = ant_rot

            # load the path and parameters of the simulations from the parameter space
            path_to_sim = np.loadtxt(self.__path_to_used_simulations, dtype='str', usecols=(0), skiprows=1, delimiter=",", unpack=True)
            zenith, xmax = np.loadtxt(self.__path_to_used_simulations, dtype='float,float', usecols=(3, 4), skiprows=1, delimiter=",", unpack=True)

            self.__crTemplateCorrelator.begin([], [], logger_level='ERROR')

            # load the template
            self.set_parameter_templates(template_event_id=0, Efield_width=[template], antenna_rotation=[template_antrot], cr_zenith=[template_zenith], cr_azimuth=[0], sampling_rate=sampling_rate)
            temp_event = self.get_templates()[0]

            count_found = 0
            count_not_found = 0
            # go through each simulation in the parameter space and see if all stations are found
            for p, zen, x in tqdm(zip(path_to_sim, zenith, xmax)):
                # restrict the parameter space to zenith angles smaller than 80°
                upper_zenith = 80
                if zen < upper_zenith:
                    # load the simulation
                    sim_evt = self.__load_simulation_event(p, det_scan, sampling_rate=sampling_rate)[0]

                    # get the survived (not survived) stations after the cuts:
                    if correlation_cut:
                        surv_corr_cut = self.__correlation_cut_dic[f'zenith{int(zen)}_xmax{int(x)}']
                    else:
                        surv_corr_cut = []
                    if trigger_cut:
                        surv_trig_cut = self.__trigger_cut_dic[f'ant_rot{int(ant_rot)}'][f'zenith{int(zen)}_xmax{int(x)}']
                    else:
                        surv_trig_cut = []
                    if denoised_hand:
                        if f'zenith{int(zen)}_xmax{int(x)}_antrot{int(ant_rot)}' in list(self.__denoised_hand_dic.keys()):
                            failed_denoise = self.__denoised_hand_dic[f'zenith{int(zen)}_xmax{int(x)}_antrot{int(ant_rot)}']
                        else:
                            failed_denoise = []
                    else:
                        failed_denoise = []

                    # go through the differnet stations and calculate the correlation value
                    for sta in sim_evt.get_stations():
                        # only look at stations which are not cut
                        if sta.get_id() in surv_corr_cut and sta.get_id() in surv_trig_cut and sta.get_id() not in failed_denoise:
                            if method == 'scipy':
                                correlation = self.__crTemplateCorrelator.correlation_scipy(sim_evt, temp_event, 0, 0, sta.get_id(), 101, showPlot=False)
                            elif method == 'window':
                                correlation = self.__crTemplateCorrelator.correlation_scan_single_spacing_matrix_variable_window(sim_evt, temp_event, 0, 0, sta.get_id(), 101, 200 * units.ns, showPlot=False)
                            max_corr = np.max(np.abs(correlation))

                            if max_corr > 0.8:
                                count_found = count_found + 1
                            else:
                                count_not_found = count_not_found + 1

                            save_dic[f'zenith{zen}_xmax{x}_antrot{ant_rot}_sta{sta.get_id()}'] = [template, max_corr]

            print(f'Number of simulations found: {count_found}')
            print(f'Total number of simulations: {count_not_found + count_found}')

        json_file = json.dumps(save_dic)
        f = open(f"/home/henrichs/software/template_search/artificial_template_bank/scan_template_{template}_method_{method}_triggercut_{trigger_cut}_corrcut_{correlation_cut}_handdenoised_{denoised_hand}.json", "w")
        f.write(json_file)
        f.close()

    def show_templates_overlap(self, templates=[5, 3, 8], method='window',correlation_cut=True, trigger_cut=True, trigger_cut_antenna_rotations=[120,130,140,150,160,170], denoised_hand=True):
        """
        show_templates_overlap method

        test if all templates overlap in the parameter space

        Parameters
        ----------
        templates: list
            list of the efield width used for the templates
        method: string
            method used to calculate the correlation
                possibilities: 'window', 'scipy'
        correlation_cut: boolean
            if true the correlation cut will be applied
        trigger_cut: boolean
            if true the trigger cut will be applied
        trigger_cut_antenna_rotations: list
            the antenna rotations for which the trigger cut will be loaded
        denoised_hand: boolean
            if true the stations rejected by hand are not considered
        Returns
        ----------
        """

        # load the scan for completeness
        filename = 'completeness_scan_templates_'
        for temp in templates:
            filename = filename + str(temp)
        filename = filename + f"_method_{method}_triggercut_{trigger_cut}_corrcut_{correlation_cut}_handdenoised_{denoised_hand}.json"
        dic_file = open(self.__path_to_template_bank + filename)
        scan_dic = json.load(dic_file)

        # load and set the different cuts
        if correlation_cut:
            self.set_correlation_cut()
        if trigger_cut:
            self.set_trigger_cut(antenna_rotations=trigger_cut_antenna_rotations)
        if denoised_hand:
            self.set_denoised_by_hand()

        # load the parameter scans for the templates which are not the main template
        scan_temp_dics = []
        for temp in templates[1:]:
            dic_file_temp = open(self.__path_to_template_bank + f"scan_template_{temp}_method_{method}_triggercut_{trigger_cut}_corrcut_{correlation_cut}_handdenoised_{denoised_hand}.json")
            scan_temp_dics.append(json.load(dic_file_temp))

        # calculate how many of the simulations in the complete scan are found by which template
        n_sim_found_by_template(scan_dic, templates)

        # load the path and parameters from all used simulations
        path_database = np.loadtxt(self.__path_to_used_simulations, dtype='str', usecols=(0), skiprows=1, delimiter=",", unpack=True)
        zenith_database, xmax_database = np.loadtxt(self.__path_to_used_simulations, dtype='float,float', usecols=(3, 4), skiprows=1, delimiter=",", unpack=True)

        # go through all points of the complete scan, if a point is not found by the main template -> check if a the neighbouring points are found by the same template
        found_dic = {}
        for k in tqdm(scan_dic.keys()):
            for itt, temp in enumerate(templates[1:]):
                # if the simulation point is found by the current template
                if scan_dic[k][0] == temp:
                    # get from the key the parameters of the simulation point
                    zenith = int(float(k[len('zenith'):k.find('_xmax')]))
                    xmax = int(float(k[k.find('xmax') + len('xmax'):k.find('_antrot')]))
                    ant_rot = int(k[k.find('antrot') + len('antrot'):k.find('_sta')])
                    station = int(float(k[k.find('_sta') + len('_sta'):]))

                    # get for this point all neighbouring points (first: neighbouring parameters(zenith, xmax, ant_rot), second: neighbouring stations)
                    neighbouring_parameter, neighbouring_stations = get_neighbouring_points(self.__path_to_template_bank, zenith, xmax, ant_rot, station, control_plot=False)

                    # calculate all parameter combinations
                    combinations = []
                    for zen in neighbouring_parameter['zenith']:
                        for ant in neighbouring_parameter['ant_rot']:
                            for xm in neighbouring_parameter['xmax']:
                                combinations.append([zen, ant, xm])

                    # check for all combinations (all neighbouring points) if all stations are found by the template
                    not_found = 0
                    for comb in combinations:
                        not_found = check_overlap(self.__detector_file, not_found, comb[0], comb[2], comb[1], path_database, zenith_database, xmax_database, correlation_cut, self.__correlation_cut_dic,
                                                  trigger_cut, self.__trigger_cut_dic, denoised_hand, self.__denoised_hand_dic, neighbouring_stations, scan_temp_dics, itt)

                    # if not_found == 0 means that all neighbouring stations are found by the template (complete overlap)
                    if not_found == 0:
                        # all neighbouring points are found
                        found_dic[k] = [templates[1:][itt], -100]
                    else:
                        # not all neighbouring are found the number of not found points is saved
                        found_dic[k] = [templates[1:][itt], not_found]

        # check the dict if all templates are found
        test_value = 0
        for key in found_dic.keys():
            if found_dic[key][1] != -100:
                logger.info(f'{key} not found: {found_dic[key]}')
                test_value = 1
        if test_value == 0:
            logger.info('All templates are overlapping!!!')

        # save the dict which says if the templates overlap
        json_file = json.dumps(found_dic)
        f = open(self.__path_to_template_bank + f"overlap_scan_method_{method}_triggercut_{trigger_cut}_corrcut_{correlation_cut}_handdenoised_{denoised_hand}.json","w")
        f.write(json_file)
        f.close()


def get_neighbouring_stations(_position, input_station_id):
    # returns for the input station all neigboruing stations in the CoREAS star

    # get the arms of the CoREAS star
    coreas_arms, coreas_arms_i = get_CoREAS_arms(_position, controll_plot=False)

    # assign each arm the corresponding angle
    arm_dict = {}
    for i in range(len(coreas_arms)):
        if coreas_arms[i][0][1] < 0:
            arm_dict[f"coreas_arm {i}"] = int(coreas_arms[i][0][1] + 360)
        else:
            arm_dict[f"coreas_arm {i}"] = int(coreas_arms[i][0][1])

    # find in which arm the input station is located
    if _position[input_station_id][1] < 0:
        input_angle = int(_position[input_station_id][1] + 360)
    else:
        input_angle = int(_position[input_station_id][1])
    arm_angle = []
    for arm in arm_dict:
        if arm_dict[arm] == input_angle:
            input_arm_number = int(arm[arm.find("arm") + 4 :])
        arm_angle.append(arm_dict[arm])

    # get the neighbouring arm angles
    arm_angle.sort()

    i_input_arm_angle = np.where(np.asarray(arm_angle) == input_angle)[0][0]
    if i_input_arm_angle == len(arm_angle) - 1:
        neighbour_arm_angle = [arm_angle[i_input_arm_angle - 1], arm_angle[0]]
    elif i_input_arm_angle == 0:
        neighbour_arm_angle = [arm_angle[len(arm_angle) - 1], arm_angle[i_input_arm_angle + 1]]
    else:
        neighbour_arm_angle = [arm_angle[i_input_arm_angle - 1], arm_angle[i_input_arm_angle + 1]]

    # get the neighbouring arm numbers
    neighbouring_arm_number = []
    for neigh_arm_ang in neighbour_arm_angle:
        for arm in arm_dict:
            if arm_dict[arm] == neigh_arm_ang:
                neighbouring_arm_number.append(int(arm[arm.find("arm") + 4 :]))

    # find neighbouring station numbers
    neighbouring_station_num = []
    # get the information at which number the input station comes in the CoREAS arm
    input_radius = np.round(_position[input_station_id][0], 4)
    arm_numbers = [input_arm_number, neighbouring_arm_number[0], neighbouring_arm_number[1]]
    i_radius = 0
    for arm_num in arm_numbers:
        # create a dict which saves for each station (of the three intersting arms) the corresponding radius
        arm_radii = np.asarray(coreas_arms)[arm_num][:, 0]
        arm_sta_num = coreas_arms_i[arm_num]
        station_dict = {}
        for i in range(len(arm_radii)):
            arm_radii[i] = np.round(arm_radii[i], 4)
            station_dict[arm_sta_num[i]] = arm_radii[i]  # station number: radius

        arm_radii.sort()
        neighbour_arm_radius = []
        if arm_num == input_arm_number:
            # get the position in the CoREAS arm of the input station
            i_radius = np.where(np.asarray(arm_radii) == input_radius)[0][0]
        else:
            # saves the radius of the same position stations in the neighbouring arms
            neighbour_arm_radius.append(arm_radii[i_radius])

        # get the stations above and below of the input station (or their counterpart in the other arms)
        if i_radius == len(arm_radii) - 1:
            neighbour_arm_radius.append(arm_radii[i_radius - 1])
        elif i_radius == 0:
            neighbour_arm_radius.append(arm_radii[i_radius + 1])
        else:
            neighbour_arm_radius.append(arm_radii[i_radius - 1])
            neighbour_arm_radius.append(arm_radii[i_radius + 1])

        # get from the dict the station number to the corresponding radii
        for neigh_arm_r in neighbour_arm_radius:
            for sta_d in station_dict:
                if station_dict[sta_d] == neigh_arm_r:
                    neighbouring_station_num.append(int(sta_d))

    return neighbouring_station_num


def get_neighbouring_points(path_to_temp_bank, input_zenith, input_xmax, input_ant_rot, input_station, control_plot=True):
    # define parameter space
    zenith = np.arange(5,80,5)
    xmax = [552, 603, 654, 705, 757, 808]
    ant_rot = [120, 130, 140, 150 ,160, 170]

    # get the different neighbouring values for all input parameters except the station
    parameter_points = {}

    help_list = []
    help_list.append(input_zenith)
    # get the neighbouring zenith angles:
    # check where in the parameter space the zenith angle is and get the corresponding neighbouring values
    input_zenith_i = np.where(np.asarray(zenith) == input_zenith)[0][0]
    if input_zenith_i == 0:
        help_list.append(zenith[input_zenith_i+1])
    elif input_zenith_i == len(zenith)-1:
        help_list.append(zenith[input_zenith_i - 1])
    else:
        help_list.append(zenith[input_zenith_i - 1])
        help_list.append(zenith[input_zenith_i + 1])

    parameter_points['zenith'] = help_list

    help_list = []
    help_list.append(input_ant_rot)
    # get the neighbouring ant_rot angles:
    # check where in the parameter space the antenna rotation angle is and get the corresponding neighbouring values
    input_ant_rot_i = np.where(np.asarray(ant_rot) == input_ant_rot)[0][0]
    if input_ant_rot_i == 0:
        help_list.append(ant_rot[input_ant_rot_i + 1])
    elif input_ant_rot_i == len(ant_rot) - 1:
        help_list.append(ant_rot[input_ant_rot_i - 1])
    else:
        help_list.append(ant_rot[input_ant_rot_i - 1])
        help_list.append(ant_rot[input_ant_rot_i + 1])

    parameter_points['ant_rot'] = help_list

    help_list = []
    help_list.append(input_xmax)
    # get the neighbouring xmax values:
    # check where in the parameter space the xmax value is and get the corresponding neighbouring values
    input_xmax_i = np.where(np.asarray(xmax) == input_xmax)[0][0]
    if input_xmax_i == 0:
        help_list.append(xmax[input_xmax_i + 1])
    elif input_xmax_i == len(xmax) - 1:
        help_list.append(xmax[input_xmax_i - 1])
    else:
        help_list.append(xmax[input_xmax_i - 1])
        help_list.append(xmax[input_xmax_i + 1])

    parameter_points['xmax'] = help_list

    # get for the input parameter configuration all neighbouring CoREAS stations
    var_neighbour_stations = {}

    # if true, a control polt showing the neighbouring stations will be created
    if control_plot:
        fig, ax = plt.subplots(ncols=3, subplot_kw={"projection": "polar"})

    # calculate the neighbouring stations for each zenith angle differently (necessary since the station positions and station numbers vary for the different zenith angles)
    for count_z, zen in enumerate(parameter_points['zenith']):
        # load the position data for each zenith angle from the lookup tables
        lookup_station_pos_file = open(path_to_temp_bank + f"station_position_lookup_zenith_angle_{zen}.json")
        lookup_station_pos = json.load(lookup_station_pos_file)

        # get the coordinates of the input station
        if zen == parameter_points['zenith'][0]:
            input_station_coord = lookup_station_pos[str(input_station)]

        # transform the dic 'lookup_table_station_pos' into a list 'help_pos'
        help_pos = []
        for key_dic in lookup_station_pos.keys():
            help_pos.append(lookup_station_pos[key_dic])

        # define the starting station for each zenith angle and get the neighbouring stations with respect to the starting stations
        # (it is necessary, because the station (e.g.) 100 is not at the same place for zenith=50 as for zenith=55)
        # use the number of stations counted from the core instead of the radius, because I think of the parameter space in the shower plane -> same number of stations = same distance from the shower core
        if zen == parameter_points['zenith'][0]:
            # get the neighbouring stations for the origin zenith angle
            neighbour_stations = get_neighbouring_stations(help_pos, input_station)
            # add the input station to the front of the list
            neighbour_stations = np.append(input_station, np.asarray(neighbour_stations)).tolist()

            # get the position of the input station counted from the shower core (needed for the other two zenith angles)
            coreas_arms, coreas_arms_i = get_CoREAS_arms(help_pos, controll_plot=False)
            input_arm_number = np.where(np.asarray(coreas_arms_i) == input_station)[0][0]
            coreas_arms_r = np.asarray(coreas_arms)[input_arm_number, :, 0]
            coreas_arms_r = np.sort(coreas_arms_r)
            input_station_position_number = np.where(coreas_arms_r == input_station_coord[0])[0][0]
        else:
            # for the other zenith angles: get the station id of the station at the same position in the CoREAS arm as the input station
            coreas_arms, coreas_arms_i = get_CoREAS_arms(help_pos, controll_plot=False)
            # get the CoREAS arm (of a different zenith angle) closest to the arm of the input station (by looking for the minimum of the difference of the input arm angle and the arm angles of the other shower)
            arm_angles = np.asarray(coreas_arms)[:, 0, 1]
            arm_number = np.where(abs(np.asarray(arm_angles - input_station_coord[1])) == min(abs(np.asarray(arm_angles) - input_station_coord[1])))[0][0]

            # get the station number which is at the same number position, as the station of the input zenith angle
            coreas_arms_r = np.asarray(coreas_arms)[arm_number, :, 0]
            coreas_arms_r = np.sort(coreas_arms_r)
            # where in the unsorted coreas arm array ('np.asarray(coreas_arms)[arm_number, :, 0]' is the station which is at the same position as the input station ('coreas_arms_r[input_station_position_number]')
            coreas_arms_r_i = np.where(np.asarray(coreas_arms)[arm_number, :, 0] == coreas_arms_r[input_station_position_number])[0][0]
            angle_station_num = coreas_arms_i[arm_number][coreas_arms_r_i]

            # get the neighbouring stations for the station number estimated above
            neighbour_stations = get_neighbouring_stations(help_pos, angle_station_num)
            neighbour_stations = np.append(angle_station_num, np.asarray(neighbour_stations)).tolist()

        # save all neighbouring stations in a dic
        var_neighbour_stations[f"zenith_{zen}"] = neighbour_stations

        # create a plot showing the neigbouring stations for the different zenith angles
        if control_plot:
            help_r = []
            help_phi = []
            for i in range(len(neighbour_stations)):
                help_r.append(lookup_station_pos[str(neighbour_stations[i])][0])
                help_phi.append(lookup_station_pos[str(neighbour_stations[i])][1] * (np.pi / 180))

            radii = np.asarray(help_pos)[:, 0]
            angles = np.asarray(help_pos)[:, 1] * (np.pi / 180)
            ax[count_z].plot(angles, radii, ls="none", marker="x")
            ax[count_z].plot(help_phi, help_r, ls="none", marker="x")
            if zen == parameter_points['zenith'][0]:
                ax[count_z].plot(input_station_coord[1] * (np.pi / 180), input_station_coord[0], ls="none", marker="x")
            ax[count_z].set_title(f"zenith angle: {zen}")
            ax[count_z].set_ylim(0, max(help_r) + 20)
            # ax[count_z].set_ylim(0, 50)

    if control_plot:
        # plt.ylim(0,600)
        plt.show()

    return parameter_points, var_neighbour_stations


def gaussian_func(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def create_Efield(detector, run_id, event_id, channel_id, station_id, station_time, trace_samples, gaussian_width, e_phi, e_theta, cr_zenith, cr_azimuth, sampling_rate):
    event = Event(run_id, event_id)

    station = Station(station_id)
    event.set_station(station)

    station.set_station_time(astropy.time.Time(station_time))

    channel = Channel(channel_id)
    station.add_channel(channel)

    sim_station = SimStation(station_id)
    station.set_sim_station(sim_station)

    sim_channel = SimChannel(channel_id, 0, 0)
    sim_station.add_channel(sim_channel)

    detector.set_event(run_id, event_id)

    electric_field = ElectricField([channel_id])

    e_field = [np.zeros(trace_samples), np.zeros(trace_samples), np.zeros(trace_samples)]
    x_data = np.arange(0, trace_samples, 1)
    # e_phi = 0.8
    # e_theta = -0.2
    for ii, x in enumerate(x_data):
        e_field[1][ii] = gaussian_func(x, e_theta, 1000, gaussian_width)
        e_field[2][ii] = gaussian_func(x, e_phi, 1000, gaussian_width)

    e_field = np.asarray(e_field)
    electric_field.set_trace(e_field, sampling_rate=sampling_rate)
    sim_station.add_electric_field(electric_field)

    sim_station.set_is_cosmic_ray()

    val_zenith = cr_zenith * (np.pi / 180)
    val_azimuth = cr_azimuth * (np.pi / 180)

    sim_station.set_parameter(stationParameters.zenith, val_zenith)
    sim_station.set_parameter(stationParameters.azimuth, val_azimuth)
    electric_field.set_parameter(electricFieldParameters.ray_path_type, 'direct')
    electric_field.set_parameter(electricFieldParameters.zenith, val_zenith)
    electric_field.set_parameter(electricFieldParameters.azimuth, val_azimuth)

    # self.__efieldToVoltageConverter.run(event, station, detector)
    # self.__hardwareResponseIncorporator.run(event, station, detector, sim_to_data=True)

    return event


def cartesian_to_polar(x_list, y_list):
    def radius(x,y):
        return np.sqrt(x**2 + y**2)
    def phi(x,y):
        # return np.arccos(x/np.sqrt(x**2 + y**2))
        return np.arctan2(x,y)
    r_list = []
    phi_list = []
    for i in range(len(x_list)):
        r_list.append(radius(x_list[i],y_list[i]))
        phi_list.append(phi(x_list[i], y_list[i]))
    return r_list, phi_list


def get_CoREAS_arms(_position, polar_coord=True, controll_plot=True):
    if not polar_coord:
        _cosPhi =[]
        for i in range(len(_position)):
            _angle=_position[i][0]/(np.sqrt(_position[i][0]**2+_position[i][1]**2))
            _cosPhi.append(np.arccos(_angle)*(180/np.pi))
    else:
        _cosPhi = np.asarray(_position)[:, 1]

    _arms= [[],[],[],[],[],[],[],[]]
    _arms_i= [[],[],[],[],[],[],[],[]]
    for i in range(8):
        _arms[i].append(_position[i])
        _arms_i[i].append(i)
    for i in range(len(_cosPhi)-8):
        i = i+8
        for j in range(8):
            if _cosPhi[i] < _cosPhi[j]+0.05 and _cosPhi[i] > _cosPhi[j]-0.05:
                _arms[j].append(_position[i])
                _arms_i[j].append(i)

    if controll_plot:
        for j in range(8):
            help1 = []
            help2 = []
            for i in range(len(_arms[j])):
                help1.append(_arms[j][i][0])
                help2.append(_arms[j][i][1])
            plt.plot(help1, help2, ls="none", marker="x")

        plt.show()

    return _arms, _arms_i


def sort_CoREAS_arms(arm_pos, arm_i):
    arm_i_sort = []
    arm_r = np.asarray(arm_pos)[:, :, 0]
    arm_r_sort = np.asarray(arm_pos)[:, :, 0]
    arm_r_sort.sort()
    for i, arm in enumerate(arm_r_sort):
        arm_i_sort_help = []
        for r in arm:
            arm_i_sort_help.append(arm_i[i][np.where(arm_r[i] == r)[0][0]])
        arm_i_sort.append(arm_i_sort_help)

    return arm_r_sort, arm_i_sort


def get_number_of_cut_stations_per_CoREAS_arm(path_lookup, surv_dic, selected_zenith, xmax):
    # load the station positions in the shower
    lookup_table_station_pos_file = open(path_lookup)
    lookup_table_station_pos = json.load(lookup_table_station_pos_file)

    # get the CoREAS arms
    position = []
    for key in lookup_table_station_pos:
        position.append(lookup_table_station_pos[key])
    arm_pos, arm_i = get_CoREAS_arms(position, polar_coord=True, controll_plot=False)
    arm_r_sort, arm_i_sort = sort_CoREAS_arms(arm_pos, arm_i)

    # count how many of the outer stations are cut
    n_cut = []
    for arm in arm_i_sort:
        n_cut_arm = 0
        for sta_num in arm:
            if sta_num in surv_dic[f'zenith{selected_zenith}_xmax{xmax}']:
                pass
            else:
                n_cut_arm = n_cut_arm + 1
        n_cut.append(n_cut_arm)

    return n_cut


def n_sim_found_by_template(dic, templates):
    found = np.zeros(len(templates))
    not_found = 0
    for key in dic.keys():
        check_val = 0
        for it, temp in enumerate(templates):
            if dic[key][0] == temp:
                found[it] = found[it] + 1
                check_val = 1
        if check_val == 0:
            not_found = not_found + 1

    for it, temp in enumerate(templates):
        logger.info(f'sim found by template {temp}: {found[it]}')
    logger.info(f'sim not found: {not_found}')


def check_overlap(path_detetcor_file, n_not_found, in_zenith, in_xmax, in_antrot, path_database, zenith_database, xmax_database, correlation_cut, corr_cut_dic, trigger_cut, trig_cut_dic, denoised_hand, denoised_hand_dic, neighbouring_stations, scan_temp_dics, temp_i):
    # load the detector
    det_overlap = GenericDetector(json_filename=path_detetcor_file, antenna_by_depth=False, create_new=True)
    det_overlap.update(datetime.datetime(2025, 10, 1))
    det_overlap.get_channel(101, 0)['ant_rotation_phi'] = in_antrot

    # get the sim path from the csv file for the corresponding parameter configuration
    sim_path = ''
    for p, z, x in zip(path_database, zenith_database, xmax_database):
        if z == in_zenith and x == in_xmax:
            sim_path = p
            break

    if sim_path != '':
        # get the survived (not survived) stations after the cuts:
        if correlation_cut:
            surv_corr_cut = corr_cut_dic[f'zenith{int(in_zenith)}_xmax{int(in_xmax)}']
        else:
            surv_corr_cut = []
        if trigger_cut:
            surv_trig_cut = trig_cut_dic[f'ant_rot{int(in_antrot)}'][f'zenith{int(in_zenith)}_xmax{int(in_xmax)}']
        else:
            surv_trig_cut = []
        if denoised_hand:
            if f'zenith{int(in_zenith)}_xmax{int(in_xmax)}_antrot{int(in_antrot)}' in list(denoised_hand_dic.keys()):
                failed_denoise = denoised_hand_dic[f'zenith{int(in_zenith)}_xmax{int(in_xmax)}_antrot{int(in_antrot)}']
            else:
                failed_denoise = []
        else:
            failed_denoise = []

        # go through all neighbouring stations and check if they found by the template
        for sta_id in neighbouring_stations[f'zenith_{in_zenith}']:
            if sta_id in surv_corr_cut and sta_id in surv_trig_cut and sta_id not in failed_denoise:
                # load maximal correlation from the precalculated files
                max_corr = scan_temp_dics[temp_i][f'zenith{float(in_zenith)}_xmax{float(in_xmax)}_antrot{int(in_antrot)}_sta{sta_id}'][1]
                if max_corr < 0.8:
                    n_not_found = n_not_found + 1

    return n_not_found