import scipy
from radiotools import helper as hp
import radiotools.coordinatesystems as cstrans
import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.utilities import fft
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.modules.neutrinoDirectionReconstruction import analytic_pulse
from NuRadioMC.utilities import medium
from scipy import signal
from scipy import optimize as opt
from scipy.spatial.transform import Rotation
import datetime
import math
from NuRadioReco.utilities import units
import datetime
import logging
import pickle

logger = logging.getLogger("neutrinoDirectionReconstructor")
logger.setLevel(logging.DEBUG)

class neutrinoDirectionReconstructor:

    def __init__(self):
        pass

    def begin(
            self, event, station, detector, use_channels, 
            reference_Vpol, reference_Hpol, PA_cluster_channels,
            Hpol_channels, propagation_config, 
            window_Vpol = [-10, +50], window_Hpol = [10, 40],
            Vrms_Hpol = 8.2 * units.mV, Vrms_Vpol = 8.2 * units.mV,
            passband = None, full_station=True,
            icemodel=None, att_model=None,
            sim = False, template = False,
            single_pulse_fit=False, restricted_input=False,
            grid_spacing = [.5*units.deg, 5*units.deg, .2],
            brute_force=True, use_fallback_timing=False,
            debug_formats=['.pdf']):
        """
        Initialize and set parameters for the reconstruction

        Parameters
        ----------
        event: Event
            The event to reconstruct the direction
        station: Station
            The station used to reconstruct the direction
        detector: Detector
            ``Detector`` object specifying the detector layout
        use_channels: list
            list of channel ids used for the reconstruction
        PA_cluster_channels: list of ints
            list of channel ids in the 'phased array cluster', consisting of both Vpol and adjacent Hpol antennas
        Hpol_channels: list of ints
            list of 'hpol'-type channels; all other channels are assumed to be 'vpol'-type
        template: bool, default: False
            (Not currently implemented)
            If True, ARZ templates are used for the reconstruction.
            If False, a parametrization is used. 'Alvarez2009' and 'ARZ average' is available. 
        sim: bool, default: False
            If True, the simulated vertex is used. This is for debugging purposes. Default sim_vertex = False.
        reference_Vpol: int
            channel id of the Vpol used to determine reference timing, viewing angle etc. 
            Should be the top Vpol of the phased array.
        reference_Hpol: int
            channel id of Hpol nearest to the Vpol. Timing of the Hpol 
            is determined using the vertex position, because difference is only 1 m.
        full_station: bool, default: True
            If True, all the raytypes in the list use_channels are used. If False, only the triggered pulse is used.
        brute_force: bool, default: True
            If True, brute force method is used. If False, minimization is used.
        fixed_timing: Boolean
            If True, the known positions of the pulses are used calculated using the vertex position. Only allowed when sim_vertex is True. If False, an extra correlation is used to find the exact pulse position. Default fixed_timing = False.
        restricted_input: bool, default: False
            If True, a reconstruction is performed a few degrees around the MC values. This is (of course) only for simulations.
        starting_values: Boolean
            if True, first the channels of the phased array are used to get starting values for the viewing angle and the energy.
        grid_spacing: [float, float, float]
            resolution of the minimization grid in (viewing angle, polarization angle, log10(energy))

        Other Parameters
        ----------------
        single_pulse_fit: bool, default: False
            if True, the viewing angle and energy are fitted with a PA Vpol and the polarization is fitted using an Hpol.

        """
        self._sim_vertex = sim
        self._Vrms = Vrms_Vpol
        self._Vrms_Hpol = Vrms_Hpol
        self._event = event
        self._station = station
        self._detector = detector
        self._reference_Vpol = reference_Vpol
        self._reference_Hpol = reference_Hpol
        self._ice_model = icemodel
        self._att_model = att_model
        self._prop_config = propagation_config
        self._passband = passband
        self._full_station = full_station
        self.__minimization_grid_spacings = grid_spacing
        self._use_fallback_timing = use_fallback_timing

        # We sort the channels. This is used in the minimizer,
        # where if the timing for a vpol/hpol channel cannot be determined,
        # it uses the timing of the reference vpol/nearest vpol, respectively, as a fallback. 
        vpol_channels = np.array([channel_id for channel_id in use_channels if channel_id not in Hpol_channels])
        hpol_channels = np.array([channel_id for channel_id in use_channels if channel_id in Hpol_channels])
        use_channels_sorted = np.concatenate([[reference_Vpol], vpol_channels, hpol_channels])
        _, idx = np.unique(use_channels_sorted, return_index=True)
        use_channels = use_channels_sorted[np.sort(idx)] 
        self._use_channels = use_channels

        # get the nearest channel as a fallback option for each hpol channel
        fallback_channels = dict()
        station_id = station.get_id()
        for i, hpol_id in enumerate(hpol_channels): # do we need fallback channels for vpol channels too?
            pos = detector.get_relative_position(station_id, hpol_id)
            fallback_channels[hpol_id] = []
            d_pos = np.zeros_like(vpol_channels, dtype=float)
            for ii, vpol_id in enumerate(vpol_channels):
                d_pos[ii] = (np.linalg.norm(detector.get_relative_position(station_id, vpol_id) - pos))
            # d_pos = d_pos
            # idx = np.argsort(d_pos)
            fallback_channels[hpol_id] = vpol_channels[np.argmin(d_pos)]
        self._fallback_channels = fallback_channels


        for channel in station.iter_channels():
            self._sampling_rate = channel.get_sampling_rate()
            break
        # we try to get the simulated shower energy, and other parameters
        simulated_energy = 0
        shower_ids = []
        for sim_shower in event.get_sim_showers():
            simulated_energy += sim_shower[shp.energy]
            shower_ids.append(sim_shower.get_id())
        self._shower_ids = shower_ids
        if len(shower_ids):
            shower_id = shower_ids[0]
            self._simulated_azimuth = event.get_sim_shower(shower_id)[shp.azimuth]
            self._simulated_zenith = event.get_sim_shower(shower_id)[shp.zenith]

        if sim:
            vertex = event.get_sim_shower(shower_id)[shp.vertex]
        else:
            vertex = station[stnp.nu_vertex]
        simulation = analytic_pulse.simulation(template, vertex)
        if sim: 
            rt = self._station[stnp.raytype_sim]
        else: 
            rt = self._station[stnp.raytype]
        simulation.begin(
            detector, station, use_channels, raytypesolution = rt, reference_channel= reference_Vpol,
            ice_model=self._ice_model, att_model=self._att_model,
            passband=self._passband, propagation_config=self._prop_config
        )
        self._launch_vector_sim, view =  simulation.simulation(
            detector, station, vertex[0],vertex[1], vertex[2], self._simulated_zenith,
            self._simulated_azimuth, simulated_energy, use_channels,
            first_iteration = True)[2:4]
        self._simulation = simulation
        self._single_pulse_fit = single_pulse_fit
        self._PA_cluster_channels = PA_cluster_channels
        self._Hpol_channels = Hpol_channels
        self._window_Vpol = window_Vpol
        self._window_Hpol = window_Hpol
        self._template = template
        self._restricted_input = restricted_input
        self._brute_force = brute_force
        self._debug_formats = debug_formats
        logger.debug(f'self._brute_force={self._brute_force}')
        return self._launch_vector_sim, view

    def run(
            self, debug=False, systematics = None,
            debug_path='./'
        ):

        """
        Module to reconstruct the direction of the event.

        Parameters
        ----------
        debug: bool, default: False
            if True, debug plots are produced. Default debug_plots = False.
        systematics: dict | None, default: None
            if dictionary is given, sytematic uncertainties are added to the VEL.
            dict = {"antenna response": {"gain": array with len(use_channels) (factor to mulitply VEL with),
                                         "phase": array with shift in frequency in MHz, array of len(use_channels) }}
        debug_path: str
            Path to store the debug plots. Default is './'.

        """
        event = self._event
        station = self._station
        shower_ids = self._shower_ids
        template = self._template
        use_channels = self._use_channels
        if self._single_pulse_fit:
            starting_values = True
        else:
            starting_values = False


        channl = station.get_channel(use_channels[0])
        self._sampling_rate = channl.get_sampling_rate()
        sampling_rate = self._sampling_rate
        detector = self._detector

        if self._sim_vertex:
            shower_id = self._shower_ids
            reconstructed_vertex = event.get_sim_shower(shower_id)[shp.vertex]
            logger.debug(f"simulated vertex direction reco: {event.get_sim_shower(shower_id)[shp.vertex]}")
        else:
            reconstructed_vertex = station[stnp.nu_vertex]

            logger.debug(f"reconstructed vertex direction reco {reconstructed_vertex}")
        self._vertex_azimuth = np.arctan2(reconstructed_vertex[1], reconstructed_vertex[0])
        if isinstance(self._ice_model, str):
            ice = medium.get_ice_model(self._ice_model)
        else:
            ice = self._ice_model
        self._cherenkov_angle = np.arccos(1 / ice.get_index_of_refraction(reconstructed_vertex))

        if self._station.has_sim_station(): # obtain some simulated values for debug plots
            shower_id = shower_ids[0]
            has_sim_station = True
            simulated_zenith = event.get_sim_shower(shower_id)[shp.zenith]
            simulated_azimuth = event.get_sim_shower(shower_id)[shp.azimuth]
            self._simulated_azimuth = simulated_azimuth
            simulated_energy = 0
            for i, shower_id in enumerate(np.unique(shower_ids)):
                if (event.get_sim_shower(shower_id)[shp.type] != "em"):
                    simulated_energy += event.get_sim_shower(shower_id)[shp.energy]
                    logger.debug(f"simulated energy {simulated_energy/units.eV:.4g} eV")
            self.__simulated_energy = simulated_energy
            simulated_vertex = event.get_sim_shower(shower_id)[shp.vertex]
            ### values for simulated vertex and simulated direction
            simulation = analytic_pulse.simulation(template, simulated_vertex)
            rt = self._station[stnp.raytype_sim]
            simulation.begin(
                detector, station, use_channels, raytypesolution = rt,
                reference_channel = self._reference_Vpol,
                ice_model=self._ice_model, att_model=self._att_model,
                passband=self._passband, propagation_config=self._prop_config, systematics = systematics)
            tracsim, timsim, lv_sim, vw_sim, a, pol_sim = simulation.simulation(
                detector, station, *simulated_vertex,
                simulated_zenith, simulated_azimuth, simulated_energy,
                use_channels, first_iteration = True)
            if pol_sim is None: # for some reason, didn't manage to obtain simulated vw / polarization angle
                pol_sim = np.nan * np.ones(3) # we still set them, so the debug plots don't fail
                vw_sim = np.nan
                has_sim_station = False # skip anything involving the sim station to avoid errors
            self._launch_vector_sim = lv_sim # not used?
            logger.debug(
                "Simulated viewing angle: {:.1f} deg / Polarization angle: {:.1f} deg".format(
                    vw_sim / units.deg, np.arctan2(pol_sim[2], pol_sim[1]) / units.deg
                )
            )

            ## check SNR of channels #TODO - unused?
            SNR = []
            for ich, channel in enumerate(station.get_sim_station().iter_channels()):
                # logger.debug("channel {}, SNR {}".format(channel.get_id(),(abs(min(channel.get_trace())) + max(channel.get_trace())) / (2*self._Vrms) ))
                if channel.get_id() in use_channels:
                    SNR.append((abs(abs(min(channel.get_trace()))) + max(channel.get_trace())) / (2*self._Vrms))
        else:
            lv_sim = np.nan
            vw_sim = np.nan
            pol_sim = np.nan * np.ones(3)


        simulation = analytic_pulse.simulation(template, reconstructed_vertex) ### if the templates are used, than the templates for the correct distance are loaded
        if not self._sim_vertex:
            rt = self._station[stnp.raytype] ## raytype from the triggered pulse

        simulation.begin(
            detector, station, use_channels, raytypesolution = rt,
            reference_channel = self._reference_Vpol,
            ice_model=self._ice_model, att_model=self._att_model,
            passband=self._passband, propagation_config=self._prop_config,
            systematics = systematics)
        self._simulation = simulation
        self._launch_vector = simulation.simulation(
            detector, station, *reconstructed_vertex, np.pi/2, 0, 1e17,
            use_channels, first_iteration=True)[2]
        # signal_zenith, signal_azimuth = hp.cartesian_to_spherical(*self._launch_vector)
        # sig_dir = hp.spherical_to_cartesian(signal_zenith, signal_azimuth)
        
        # initialize simulated ref values to avoid UnboundLocalError if no sim_station is present
        fsim, fsimsim, all_fsim, all_fsimsim, sim_reduced_chi2_Vpol, sim_reduced_chi2_Hpol = 6*[np.nan,]
        if station.has_sim_station():

            logger.debug(f"simulated vertex    : {simulated_vertex}")
            logger.debug(f"reconstructed vertex: {reconstructed_vertex}")
            #### values for reconstructed vertex and simulated direction
            if has_sim_station:
                # traces_sim, timing_sim, self._launch_vector_sim, viewingangles_sim, rayptypes, a = simulation.simulation(
                #     det, station, event.get_sim_shower(shower_id)[shp.vertex][0],
                #     event.get_sim_shower(shower_id)[shp.vertex][1],
                #     event.get_sim_shower(shower_id)[shp.vertex][2],
                #     simulated_zenith, simulated_azimuth, simulated_energy, use_channels, first_iter = True)


                fsimsim = self.minimizer(
                    [simulated_zenith, simulated_azimuth, np.log10(simulated_energy)], 
                    *simulated_vertex, 
                    minimize =  True, first_iter = True, 
                    ch_Vpol = self._reference_Vpol, ch_Hpol = self._reference_Hpol, 
                    full_station = self._full_station, sim = True)
                sim_simvertex_output = self.minimizer(
                    [simulated_zenith, simulated_azimuth, np.log10(simulated_energy)], 
                    *simulated_vertex, 
                    minimize =  False, first_iter = True, 
                    ch_Vpol = self._reference_Vpol, ch_Hpol = self._reference_Hpol, 
                    full_station = self._full_station, sim = True)
                
                tracsim = sim_simvertex_output[0]
                all_fsimsim = sim_simvertex_output[3]

                fsim = self.minimizer(
                    [simulated_zenith, simulated_azimuth, np.log10(simulated_energy)], 
                    *reconstructed_vertex,
                    minimize =  True, first_iter = True, 
                    ch_Vpol = self._reference_Vpol, ch_Hpol = self._reference_Hpol, 
                    full_station = self._full_station, sim = True)
                all_fsim = self.minimizer(
                    [simulated_zenith, simulated_azimuth, np.log10(simulated_energy)], 
                    *reconstructed_vertex,
                    minimize=False, first_iter=True, 
                    ch_Vpol=self._reference_Vpol, ch_Hpol=self._reference_Hpol,
                    full_station=self._full_station, sim=True)[3]
                
                logger.debug(
                    "Chi2 values for simulated direction and with/out simulated vertex are {}/{}".format(fsimsim, fsim))

                sim_reduced_chi2_Vpol = self.minimizer(
                    [simulated_zenith,simulated_azimuth, np.log10(simulated_energy)],
                    *simulated_vertex,
                    minimize =  False, 
                    ch_Vpol = self._reference_Vpol, ch_Hpol = self._reference_Hpol, 
                    full_station = self._full_station, sim = True)[4][0]

                sim_reduced_chi2_Hpol = self.minimizer(
                    [simulated_zenith,simulated_azimuth, np.log10(simulated_energy)],
                    *simulated_vertex, 
                    minimize =  False, first_iter = True, 
                    ch_Vpol = self._reference_Vpol, ch_Hpol = self._reference_Hpol, 
                    full_station = self._full_station, sim = True)[4][1]

                tracsim_recvertex = self.minimizer(
                    [simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], 
                    *reconstructed_vertex, 
                    minimize =  False, first_iter = True,
                    ch_Vpol = self._reference_Vpol, ch_Hpol = self._reference_Hpol, 
                    full_station = self._full_station)[0]                

        viewing_start = self._cherenkov_angle - np.deg2rad(10) # 15 degs
        viewing_end = self._cherenkov_angle + np.deg2rad(10)
        # d_viewing_grid = .5 * units.deg # originally .5 deg
        energy_start = 1e16 * units.eV
        energy_end = 1e19 * units.eV + 1e14 * units.eV
        # d_log_energy = .2
        theta_start = np.deg2rad(-180) #-180
        theta_end =  np.deg2rad(180) #180
        # d_theta_grid = 5 * units.deg # originally 1 degree

        d_viewing_grid, d_theta_grid, d_log_energy = self.__minimization_grid_spacings

        cop = datetime.datetime.now()
        logger.info("Starting direction reconstruction...")
        if self._brute_force and not self._restricted_input: # restricted_input:
            logger.warning("Using brute force optimization")
            if starting_values:
                results2 = opt.brute(self.minimizer, ranges=(slice(viewing_start, viewing_end, np.deg2rad(.5)), slice(theta_start, theta_end, np.deg2rad(1)), slice(np.log10(energy_start) - .15, np.log10(energy_start) + .15, .1)), full_output = True, finish = opt.fmin , args = (reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True, False, False, True, False, self._reference_Vpol, self._reference_Hpol, self._full_station))
                results1 = opt.brute(self.minimizer, ranges=(slice(viewing_start, viewing_end, np.deg2rad(.5)), slice(theta_start, theta_end, np.deg2rad(1)), slice(np.log10(energy_start) - .15, np.log10(energy_start) + .15, .1)), full_output = True, finish = opt.fmin , args = (reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True, False, False, True, False, self._reference_Vpol, self._reference_Hpol, self._full_station))
                if results2[1] < results1[1]:
                    results = results2
                else:
                    results = results1
            else:
                results = opt.brute(
                    self.minimizer,
                    ranges=(
                        slice(viewing_start, viewing_end, d_viewing_grid),
                        slice(theta_start, theta_end, d_theta_grid),
                        slice(np.log10(energy_start), np.log10(energy_end), d_log_energy)
                    ), full_output = True, finish = opt.fmin,
                    args = (
                        reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2],
                        True, False, False, True, False, self._reference_Vpol, self._reference_Hpol, self._full_station
                    )
                )

        elif self._restricted_input:
            logger.warning('Using restricted input!')
            d_angle = 1
            zenith_start =  simulated_zenith - np.deg2rad(d_angle)
            zenith_end =simulated_zenith +  np.deg2rad(d_angle)
            azimuth_start =simulated_azimuth - np.deg2rad(d_angle)
            azimuth_end = simulated_azimuth + np.deg2rad(d_angle)
            energy_start = np.log10(simulated_energy) - 1
            energy_end = np.log10(simulated_energy) + 1
            results = opt.brute(self.minimizer, ranges=(slice(zenith_start, zenith_end, np.deg2rad(.5)), slice(azimuth_start, azimuth_end, np.deg2rad(.5)), slice(energy_start, energy_end, .05)), finish = opt.fmin, full_output = True, args = (reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True, False, False, False, False, self._reference_Vpol, self._reference_Hpol, self._full_station))

        else:
            logger.warning('Using iterative fitter')
            chisq = np.nan * np.zeros(4)
            results = np.nan * np.zeros((4,3))
            is_valid = np.nan * np.zeros(4)
            
            # def constr(x):
            #     """
            #     implement a constraint
                
            #     Set some constraints on viewing angle, zenith, and energy.

            #     Returns
            #     -------
            #     bound : float
            #         constraint is satisfied if bound > 0.
                
            #     """
            #     zenith, azimuth = self._transform_angles(*x[:2])
            #     d_vw = (x[0] - self._cherenkov_angle)
            #     bound = np.max([
            #         (d_vw/(15*units.deg))**2, # viewing angle within 15 deg of cherenkov angle
            #         (1.5 * x[1] / np.pi)**2,  # polarization < 135 deg
            #         (zenith-100*units.deg)/(20*units.deg), # zenith < 120 deg
            #         (15 - x[-1]), # energy > 1e14 eV
            #         (x[-1]-20)  # energy < 1e21 eV
            #     ])

            #     return (1-bound)

            # constraint = opt.NonlinearConstraint(constr, 0, np.inf)

            # res = opt.shgo(
            #     self.minimizer, #x0=[self._cherenkov_angle, 0, 18.0], 
            #     bounds = [(self._cherenkov_angle - 15*units.deg, self._cherenkov_angle + 15*units.deg), (-3*np.pi/2, 3*np.pi/2), (15,20.5)],
            #     args = (
            #         reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], 
            #         True, False, False, True, False, self._reference_Vpol, self._reference_Hpol, 
            #         self._full_station), 
            #     constraints = [dict(type='ineq', fun=constr)]
            # )
            # print(res)
            # chisq = res.fun
            # results = res.x
            
            res = opt.minimize(
                self.minimizer, x0=[self._cherenkov_angle, 0, 18.0], 
                args = (
                    reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], 
                    True, False, False, True, False, self._reference_Vpol, self._reference_Hpol, 
                    self._full_station), 
                #method='Nelder-Mead', options=dict(xatol=1e-4,fatol=1e-2)
                # method = 'trust-constr', constraints = constraint
                # constraints = [dict(type='ineq', fun=constr)]
                tol=1e-6
            )
            viewing_sign = int(np.sign(res.x[0] - self._cherenkov_angle))
            polarization_sign = int(np.sign(res.x[1]))
            index = 1 + viewing_sign + (polarization_sign + 1) // 2 # ranges from 0-3
            chisq[index] = res.fun
            results[index] = res.x
            is_valid[index] = res.success
            logger.debug(f'First iteration: index {index}, chisq {res.fun:.2f}, result {res.x}, message: {res.message}')
            if not res.success:
                logger.warning(f'Fit {index} failed with message {res.message}')
            # to avoid local minima (wrong side of cherenkov cone, wrong polarization sign)
            # we re-run the minimizer starting at the other minima
            signs = [-1,1]
            for index in np.arange(4)[np.isnan(chisq)]:
                viewing_sign = signs[index // 2]
                polarization_sign = signs[index % 2]
                old_guess = results[np.nanargmin(chisq)] 
                # we use min/median as 'sanity checks' to avoid starting the fit at an unlikely point
                viewing_guess = self._cherenkov_angle + viewing_sign * np.min([7*units.deg, np.abs(old_guess[0] - self._cherenkov_angle)])
                polarization_guess = polarization_sign * np.min([np.pi/3,np.abs(old_guess[1])])
                energy_guess = np.median([16, old_guess[2],19])
                # logger.info(f'Iteration {index} - x0={[viewing_guess, polarization_guess, energy_guess]}, chisq={np.nanmin(chisq)}')

                res = opt.minimize(
                    self.minimizer, x0=[viewing_guess, polarization_guess, energy_guess], 
                    args = (
                        reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], 
                        True, False, False, True, False, self._reference_Vpol, self._reference_Hpol, 
                        self._full_station), 
                    #method='Nelder-Mead', options=dict(xatol=1e-4,fatol=1e-2),
                    # method = 'trust-constr', constraints = constraint
                    # constraints = [dict(type='ineq', fun=constr)]
                    tol=1e-6,
                )
                chisq[index] = res.fun
                results[index] = res.x
                is_valid[index] = res.success
                if not res.success:
                    logger.warning(f'Fit {index} failed with message: {res.message}')

            logger.debug(
                'Fitter output:\n'
                '{:8s} | {:8s} {:8s} {:8s} {:8s}\n'.format('Chisq', 'vw.ang', 'pol.ang', 'log10(E)', 'valid') +
                '\n'.join([
                    '{:8.1f} | {:8.2f} {:8.2f} {:8.2f} {:8.0f}'.format(chisq[i], *(results[i,:2]/units.deg), results[i, -1], is_valid[i])
                    for i in range(4)]
                )
            )
            results = results[np.nanargmin(chisq)]


        logger.info(f"...finished direction reconstruction in {datetime.datetime.now() - cop}")
        # print("cache statistics for analytic_pulse ray tracer")
        # print(self._simulation._raytracer.cache_info())
        vw_grid = results[-2]
        chi2_grid = results[-1]
        # np.save("{}/grid_{}".format(debug_path, self._event.get_run_number()), vw_grid)
        # np.save("{}/chi2_{}".format(debug_path, self._event.get_run_number()), chi2_grid)
        ###### GET PARAMETERS #########

        # if only_simulation:
        #     rec_zenith = simulated_zenith
        #     rec_azimuth = simulated_azimuth
        #     rec_energy = simulated_energy

        if self._brute_force and not self._restricted_input:
            # rotation_matrix = hp.get_rotation(sig_dir, np.array([0, 0,1]))
            viewing_angle = results[0][0]
            polarization_angle = results[0][1]
            rec_zenith, rec_azimuth = self._transform_angles(viewing_angle, polarization_angle)
            rec_energy = 10**results[0][2]
        elif self._restricted_input:
            rec_zenith = results[0][0]
            rec_azimuth = results[0][1]
            rec_energy = 10**results[0][2]
        else:
            viewing_angle = results[0]
            polarization_angle = results[1]
            rec_zenith, rec_azimuth = self._transform_angles(viewing_angle, polarization_angle)
            rec_energy = 10**results[2]

        ###### PRINT RESULTS ###############
        if station.has_sim_station():
            logger.info(f"simulated energy {simulated_energy/units.eV:.4g} eV")
            logger.info(f"simulated zenith {simulated_zenith/units.deg:.3f} deg")
            logger.info(f"simulated azimuth {simulated_azimuth/units.deg:.3f} deg")

        logger.info(f"reconstructed energy {rec_energy/units.eV:.4g} eV")
        logger.info(f"reconstructed zenith {rec_zenith/units.deg:.3f} deg")
        logger.info(f"reconstructed azimuth {rec_azimuth % (2*np.pi)/units.deg:.3f} deg")

        ## get the traces for the reconstructed energy and direction
        reconstruction_output = self.minimizer(
            [rec_zenith, rec_azimuth, np.log10(rec_energy)], 
            *reconstructed_vertex, minimize = False, 
            ch_Vpol = self._reference_Vpol, ch_Hpol = self._reference_Hpol,
            full_station = self._full_station)
        tracrec = reconstruction_output[0]
        fit_reduced_chi2_Vpol = reconstruction_output[4][0]
        fit_reduced_chi2_Hpol = reconstruction_output[4][1]
        channels_overreconstructed = reconstruction_output[5]
        extra_channel = reconstruction_output[6]
        chi2_dict = reconstruction_output[3]
        total_chi2 = np.sum(np.concatenate([list(i.values()) for i in chi2_dict.values()])) #np.sum([list(chi2s.values()) for chi2s in chi2_dict.values()])
        included_channels = reconstruction_output[7]

        fminfit = self.minimizer(
            [rec_zenith, rec_azimuth, np.log10(rec_energy)],
            *reconstructed_vertex, 
            minimize =  True, 
            ch_Vpol = self._reference_Vpol, ch_Hpol = self._reference_Hpol, 
            full_station = self._full_station)

        all_fminfit = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize =  False, ch_Vpol = self._reference_Vpol, ch_Hpol = self._reference_Hpol, full_station = self._full_station)[3]
        bounds = ((14, 20))
        method = 'BFGS'
        fmin_simdir_recvertex = np.nan        
        if station.has_sim_station(): 
            results = scipy.optimize.minimize(self.minimizer, [14],method = method, args=(reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True, False, False,False, [simulated_zenith, simulated_azimuth], self._reference_Vpol, self._reference_Hpol, True, False), bounds= bounds)
            fmin_simdir_recvertex = self.minimizer([simulated_zenith, simulated_azimuth, results.x[0]], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = True, ch_Vpol = self._reference_Vpol, ch_Hpol = self._reference_Hpol, full_station = self._full_station)

        ### values for reconstructed vertex and reconstructed direction
        traces_rec, timing_rec, launch_vector_rec, viewingangle_rec, a, pol_rec =  simulation.simulation( detector, station, reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], rec_zenith, rec_azimuth, rec_energy, use_channels, first_iteration = True)

        if debug:
            logger.warning("making debug plots....")
            linewidth = 2
            tracdata = reconstruction_output[1]
            timingdata = reconstruction_output[2]
            timingsim = self.minimizer(
                [simulated_zenith, simulated_azimuth, np.log10(simulated_energy)],
                *simulated_vertex,
                first_iter = True, minimize = False, 
                ch_Vpol = self._reference_Vpol, ch_Hpol = self._reference_Hpol,
                full_station = self._full_station, sim=True)[2]

            timingsim_recvertex = self.minimizer([simulated_zenith, simulated_azimuth, np.log10(simulated_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], first_iter = True, minimize = False, ch_Vpol = self._reference_Vpol, ch_Hpol = self._reference_Hpol, full_station = self._full_station)[2]
            fig, ax = plt.subplots(len(use_channels), 3, sharex=False, figsize=(16, 4*len(use_channels)))

            ich = 0
            SNRs = np.zeros((len(use_channels), 2))

            for channel in station.iter_channels():
                channel_id = channel.get_id()
                if channel_id in use_channels: # use channels needs to be sorted
                    sim_trace = None
                    for sim_channel in station.get_sim_station().get_channels_by_channel_id(channel_id):
                        if sim_trace is None:
                            sim_trace = sim_channel
                        else:
                            sim_trace += sim_channel


                    if len(tracdata[channel_id]) > 0:
                        # logger.debug("Plotting channel {}....".format(channel_id))
                        # logger.debug("Data trace: {:.0f} - {:.0f} ns".format(channel.get_times()[0], channel.get_times()[-1]))
                        # logger.debug("Sim trace: {:.0f} - {:.0f} ns".format(timingsim[channel_id][0][0], timingsim[channel_id][0][-1]))
                        ax[ich][0].grid()
                        ax[ich][2].grid()
                        ax[ich][0].set_xlabel("timing [ns]", )
                        ax[ich][0].plot(channel.get_times(), channel.get_trace(), lw = linewidth, label = 'data', color = 'black')

                        #ax[ich][0].fill_between(timingdata[channel_id][0], tracrec[channel_id][0] - tracrec[channel_id][0], tracrec[channel_id][0] +  tracrec[channel_id][0], color = 'green', alpha = 0.2)
                        ax[ich][2].plot( np.fft.rfftfreq(len(tracdata[channel_id][0]), 1/sampling_rate), abs(fft.time2freq( tracdata[channel_id][0], sampling_rate)), color = 'black', lw = linewidth)
                        ax[ich][0].plot(timingsim[channel_id][0], tracsim[channel_id][0], label = 'simulation', color = 'orange', lw = linewidth)
                        if sim_trace != None: ax[ich][0].plot(sim_trace.get_times(), sim_trace.get_trace(), label = 'sim channel', color = 'red', lw = linewidth)

                        ax[ich][0].plot(timingsim_recvertex[channel_id][0], tracsim_recvertex[channel_id][0], label = 'simulation rec vertex', color = 'lightblue' , lw = linewidth, ls = '--')

                        # show data / simulation time windows
                        window_sim = timingsim[channel_id][0][0], timingsim[channel_id][0][-1]
                        window_rec = timingdata[channel_id][0][0], timingdata[channel_id][0][-1]
                        for t in window_sim:
                            ax[ich][0].axvline(t, color='orange', ls=':')
                        for t in window_rec:
                            ax[ich][0].axvline(t, color='green', ls=':')
                        ax[ich][0].set_xlim(np.min(window_sim+window_rec)-5, np.max(window_sim+window_rec)+5)

                        ax[ich][0].plot(timingdata[channel_id][0], tracrec[channel_id][0], label = 'reconstruction', lw = linewidth, color = 'green')

                        if sim_trace != None: ax[ich][2].plot( np.fft.rfftfreq(len(sim_trace.get_trace()), 1/sampling_rate), abs(fft.time2freq(sim_trace.get_trace(), sampling_rate)), lw = linewidth, color = 'red')
                        ax[ich][2].plot( np.fft.rfftfreq(len(tracsim[channel_id][0]), 1/sampling_rate), abs(fft.time2freq(tracsim[channel_id][0], sampling_rate)), lw = linewidth, color = 'orange')

                        ax[ich][2].plot( np.fft.rfftfreq(len(tracrec[channel_id][0]), 1/sampling_rate), abs(fft.time2freq(tracrec[channel_id][0], sampling_rate)), color = 'green', lw = linewidth)
                        ax[ich][2].set_xlim((0, 1))
                        ax[ich][2].set_xlabel("frequency [GHz]", )

                        ax[ich][0].legend()

                    if len(tracdata[channel_id]) > 1:
                        ax[ich][1].grid()
                        ax[ich][1].set_xlabel("timing [ns]", )
                        ax[ich][1].plot(channel.get_times(), channel.get_trace(), label = 'data', lw = linewidth, color = 'black')
                        ax[ich][2].plot(np.fft.rfftfreq(len(timingsim[channel_id][1]), 1/sampling_rate), abs(fft.time2freq(tracsim[channel_id][1], sampling_rate)), lw = linewidth, color = 'red')
                        ax[ich][2].plot( np.fft.rfftfreq(len(tracdata[channel_id][1]), 1/sampling_rate), abs(fft.time2freq(tracdata[channel_id][1], sampling_rate)), color = 'black', lw = linewidth)
                        ax[ich][1].plot(timingsim[channel_id][1], tracsim[channel_id][1], label = 'simulation', color = 'orange', lw = linewidth)
                        if sim_trace != None: ax[ich][1].plot(sim_trace.get_times(), sim_trace.get_trace(), label = 'sim channel', color = 'red', lw = linewidth)
                        if 1:#channel_id in [6]:#,7,8,9]:
                            ax[ich][1].plot(timingdata[channel_id][1], tracrec[channel_id][1], label = 'reconstruction', color = 'green', lw = linewidth)
                            #ax[ich][1].fill_between(timingdata[channel_id][1], tracrec[channel_id][1] - tracrec[channel_id][1], tracrec[channel_id][1] +  tracrec[channel_id][1], color = 'green', alpha = 0.2)

                        ax[ich][2].plot( np.fft.rfftfreq(len(tracsim[channel_id][1]), 1/sampling_rate), abs(fft.time2freq(tracsim[channel_id][1], sampling_rate)), lw = linewidth, color = 'orange')
                        ax[ich][1].plot(timingsim_recvertex[channel_id][1], tracsim_recvertex[channel_id][1], label = 'simulation rec vertex', color = 'lightblue', lw = linewidth, ls = '--')

                        # show data / simulation time windows
                        window_sim = timingsim[channel_id][1][0], timingsim[channel_id][1][-1]
                        window_rec = timingdata[channel_id][1][0], timingdata[channel_id][1][-1]
                        for t in window_sim:
                            ax[ich][1].axvline(t, color='orange', ls=':')
                        for t in window_rec:
                            ax[ich][1].axvline(t, color='green', ls=':')
                        ax[ich][1].set_xlim(np.min(window_sim+window_rec)-5, np.max(window_sim+window_rec)+5)



                        ax[ich][2].plot( np.fft.rfftfreq(len(tracrec[channel_id][1]), 1/sampling_rate), abs(fft.time2freq(tracrec[channel_id][1], sampling_rate)), color = 'green', lw = linewidth, label = 'channel id {}'.format(channel_id))
                        ax[ich][2].legend()
                    for ii in range(2):
                        chi2 = chi2_dict[channel_id][ii]
                        if chi2 > 0:
                            ax[ich][ii].set_title(f'$\chi^2={chi2:.2f}$')
                        else:
                            ax[ich][ii].set_fc('grey')

                    ich += 1
            ax[0][0].legend()


            fig.tight_layout()
            fig_path = "{}/{}_{}_fit".format(debug_path, self._event.get_run_number(), self._event.get_id())
            logger.debug(f"output path for stored figure: {fig_path}")
            # print("output path for stored figure","{}/fit_{}.pdf".format(debugplots_path, filenumber))
            save_fig(fig, fig_path, self._debug_formats)
            plt.close('all')

            ### chi squared grid from opt.brute:
            # plt.rc('xtick',)
            # plt.rc('ytick', labelsize = 10)
            if self._brute_force:
                min_energy_index = np.unravel_index(np.argmin(chi2_grid), vw_grid.shape)[-1]
                extent = (
                    vw_grid[0,0,0,0] / units.deg,
                    vw_grid[0,-1,0,0] / units.deg,
                    vw_grid[1,0,0,0] / units.deg,
                    vw_grid[1,0,-1,0] / units.deg,
                )

                fig = plt.figure(figsize=(6,6))
                chi2_grid = chi2_grid - total_chi2
                chi2_grid_min_energy = chi2_grid[:,:,min_energy_index]
                try:
                    vmax = np.percentile(chi2_grid_min_energy[np.where(chi2_grid_min_energy < np.inf)], 20)
                except IndexError:
                    # we probably don't have any valid results... but let's not throw an error
                    # because of the debug plot
                    vmax = None
                plt.imshow(
                    (np.nanmin(chi2_grid, axis=2).T),
                    extent=extent,
                    aspect='auto',
                    vmax=vmax,
                    origin='lower'
                )
                if self._restricted_input: # we did the minimization in azimuth/zenith, so should plot this
                    x_sim, y_sim = simulated_zenith / units.deg, simulated_azimuth / units.deg % 360
                    x_rec, y_rec = rec_zenith / units.deg, rec_azimuth / units.deg % 360
                    xlabel, ylabel = 'zenith [deg]', 'azimuth [deg]'
                else: # minimization in viewing angle & polarization
                    x_sim, y_sim = vw_sim / units.deg, np.arctan2(pol_sim[2], pol_sim[1]) / units.deg
                    x_rec, y_rec = viewingangle_rec / units.deg, np.arctan2(pol_rec[2], pol_rec[1]) / units.deg
                    xlabel, ylabel = 'Viewing angle [deg]', 'Polarization angle [deg]'

                plt.plot(
                    x_sim, y_sim,
                    marker='o', label='{:.1f}, {:.1f}, 1e{:.2f} (simulated)'.format(
                        x_sim, y_sim, np.log10(simulated_energy)
                    ), color='red', ls='none'
                )
                plt.plot(
                    x_rec, y_rec,
                    marker='x', label='{:.1f}, {:.1f}, 1e{:.2f} (reconstructed)'.format(
                        x_rec, y_rec, np.log10(rec_energy)
                    ), color='magenta', ms=8, mfc='magenta', ls='none'
                )
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.legend()
                # plt.title("E=1e{:.2f} eV".format(vw_grid[2,0,0,min_energy_index]))
                plt.title(f'$\chi^2_\mathrm{{min}} = {total_chi2:.0f} / {self.__dof}$')
                cbar = plt.colorbar(label=r"$\chi^2-\chi^2_\mathrm{min}$")
                # vmax = cbar.vmax
                # vmin = cbar.vmin
                # cbar_ticks = cbar.get_ticks()
                # cbar_ticks = cbar_ticks[(cbar_ticks < vmax) & (cbar_ticks > vmin)]
                # cbar_ticks[0] = vmin
                # tick_precision = int(np.max([0, np.min([-(np.log10(vmax-vmin)-1) // 1, 2])]))
                # cbar_ticklabels = [f'{tick:.{tick_precision}f}' for tick in cbar_ticks]
                # cbar_ticklabels[0] = f'{vmin:.2f} / {self.__dof}'
            #    cbar.set_ticks(cbar_ticks, labels=cbar_ticklabels)
                plt.tight_layout()
                save_fig(
                    fig, "{}/{}_{}_chi_squared".format(
                    debug_path, self._event.get_run_number(), self._event.get_id()), 
                    self._debug_formats)
                plt.close()
            #exit()


        ###### STORE PARAMTERS AND PRINT PARAMTERS #########
        station.set_parameter(stnp.extra_channels, extra_channel)
        station.set_parameter(stnp.over_rec, channels_overreconstructed)
        station.set_parameter(stnp.direction_fit_pulses, included_channels)
        station.set_parameter(stnp.nu_zenith, rec_zenith)
        station.set_parameter(stnp.nu_azimuth, self.transform_azimuth(rec_azimuth))
        station.set_parameter(stnp.nu_energy, rec_energy)
        station.set_parameter(stnp.chi2, [fsim, fminfit, fsimsim, self.__dof, sim_reduced_chi2_Vpol, sim_reduced_chi2_Hpol, fit_reduced_chi2_Vpol, fit_reduced_chi2_Hpol, fmin_simdir_recvertex])
        station.set_parameter(stnp.launch_vector, [lv_sim, launch_vector_rec])
        station.set_parameter(stnp.polarization, [pol_sim, pol_rec])
        station.set_parameter(stnp.viewing_angle, [vw_sim, viewingangle_rec])
        if station.has_sim_station(): 
            logger.debug("chi2 for simulated rec vertex {}, simulated sim vertex {} and fit {}".format(fsim, fsimsim, fminfit))#reconstructed vertex
            logger.debug("chi2 for all channels simulated rec vertex {}, simulated sim vertex {} and fit {}".format(all_fsim, all_fsimsim, all_fminfit))#reconstructed vertex
            logger.warning(f"Fit chi squared: {total_chi2:.2f} / {self.__dof}")
            logger.debug("launch vector for simulated {} and fit {}".format(lv_sim, launch_vector_rec))
            zen_sim = hp.cartesian_to_spherical(*lv_sim)[0]
            zen_rec = hp.cartesian_to_spherical(*launch_vector_rec)[0]
            logger.debug("launch zenith for simulated {} and fit {}".format(np.rad2deg(zen_sim), np.rad2deg(zen_rec)))
            logger.debug("polarization for simulated {} and fit {}".format(pol_sim, pol_rec))
            logger.debug("polarization angle for simulated {} and fit{}".format(np.rad2deg(np.arctan2(pol_sim[2], pol_sim[1])), np.rad2deg(np.arctan2(pol_rec[2], pol_rec[1]))))
            logger.debug("viewing angle for simulated {} and fit {}".format(np.rad2deg(vw_sim), np.rad2deg(viewingangle_rec)))
            logger.debug("reduced chi2 Vpol for simulated {} and fit {}".format(sim_reduced_chi2_Vpol, fit_reduced_chi2_Vpol))
            logger.debug("reduced chi2 Hpol for simulated {} and fit {}".format(sim_reduced_chi2_Hpol, fit_reduced_chi2_Hpol))
            logger.debug(f"over reconstructed channels {channels_overreconstructed}")
            logger.debug(f"extra channels {extra_channel}")
            logger.debug(f"L for rec vertex sim direction rec energy: {fmin_simdir_recvertex}")
            logger.debug(f"L for reconstructed vertexy directin and energy: {fminfit}")

    def transform_azimuth(self, azimuth): ## from [-180, 180] to [0, 360]
        azimuth = np.rad2deg(azimuth)
        if azimuth < 0:
            azimuth = 360 + azimuth
        return np.deg2rad(azimuth)

    def _transform_angles(self, viewing_angle, polarization_angle):
        lv = self._launch_vector
        pol = np.array([0, np.cos(polarization_angle), np.sin(polarization_angle)])
        cs = cstrans.cstrafo(*hp.cartesian_to_spherical(*lv))
        pol_cartesian = cs.transform_from_onsky_to_ground(pol)
        rotation_axis = np.cross(lv, pol_cartesian)
        rot = Rotation.from_rotvec(viewing_angle * rotation_axis / np.linalg.norm(rotation_axis))
        nu_direction = -rot.apply(lv) # using the convention that nu_direction points to its origin
        zenith, azimuth = hp.cartesian_to_spherical(*nu_direction)
        return zenith, azimuth

    def minimizer(
            self, params, vertex_x, vertex_y, vertex_z, minimize = True, timing_k = False,
            first_iter = False, banana = False,  direction = [0, 0], ch_Vpol = 6, ch_Hpol = False,
            full_station = False, single_pulse =False, fixed_timing = False,
            starting_values = False, penalty = False, sim = False
        ):
        """

        Parameters
        ----------
        params: list
            input paramters for viewing angle / direction
        vertex_x, vertex_y, vertex_z: float
            input vertex
        minimize: Boolean
            If true, minimization output is given (chi2). If False, parameters are returned. Default minimize = True.
        first_iter: Boolean
            If true, raytracing is performed. If false, raytracing is not performed. Default first_iter = False.
        banana: Boolean
            If true, input values are viewing angle and energy. If false, input values should be theta and phi. Default banana = False.
        direction: list
            List with phi and theta direction. This is only for determining contours. Default direction = [0,0].
        ch_Vpol: int
            channel id for the Vpol of the reference pulse. Must be upper Vpol in phased array. Default ch_Vpol = 6.
        ch_Hpol: int
            channel id for the Hpol which is closest by the ch_Vpol
        full_station:
            if True, all raytype solutions for all channels are used, regardless of SNR of pulse. Default full_station = True.
        single_pulse: Boolean
            if True, only 1 pulse is used from the reference Vpol. Default single_pulse = False.
        fixed_timing: Boolean
            if True, the positions of the pulses using the simulated timing is used. This only works for the simulated vertex and for Alvarez2009 reconstruction and simulation. Default fixed_timing = False.
        starting_values: Boolean
            if True, the phased array cluster is used to obtain starting values for the viewing angle and the energy to limit the timing for the brute force approach. Default starting_values = False.
        penalty: Boolean
            if True, a penalty is included such that the reconstruction is not allowed to overshoot the traces with snr< 3.5. Default penalty = False.

        """



        if banana: ## if input is viewing angle and energy, they need to be transformed to zenith and azimuth
            if len(params) == 3:
                cherenkov_angle, angle, log_energy = params
                # print("viewing angle and energy and angle ", [np.rad2deg(cherenkov_angle), log_energy, np.rad2deg(angle)])
            if len(params) == 2:
                cherenkov_angle, log_energy = params
                angle = self._angle
                #print("viewing angle and energy and angle ", [np.rad2deg(cherenkov_angle), log_energy, np.rad2deg(angle)])
            if len(params) == 1:
                cherenkov_angle = self._viewing_angle
                self._pol_angle = params
                log_energy = self._log_energy
                angle = self._angle
            energy = 10**log_energy

            zenith, azimuth = self._transform_angles(cherenkov_angle, angle)

            if self._brute_force: # minimizers don't like this kind of behaviour, so we only use it for the brute force method
                if np.rad2deg(zenith) > 120:
                    return np.inf ## not in field of view

        else:
            if len(params) == 3:
                zenith, azimuth, log_energy = params
                energy = 10**log_energy
            if len(params) == 1:
                log_energy = params

                energy = 10**log_energy[0]
                zenith, azimuth = direction

        azimuth = self.transform_azimuth(azimuth)

        # pol_angle = 0
        # if self._single_pulse_fit:
        #     pol_angle = self._pol_angle
        traces, timing, launch_vector, viewingangles, raytypes, pol = self._simulation.simulation(
            self._detector, self._station, vertex_x, vertex_y, vertex_z, zenith, azimuth, energy,
            self._use_channels, first_iteration = first_iter, starting_values = starting_values,) ## get traces due to neutrino direction and vertex position
        chi2 = 0
        all_chi2 = dict()
        over_reconstructed = [] ## list for channel ids where reconstruction is larger than data
        # extra_channel = 0 ## count number of pulses besides triggering pulse in Vpol + Hpol


        rec_traces = {} ## to store reconstructed traces
        data_traces = {} ## to store data traces
        data_timing = {} ## to store timing


        #get timing and pulse position for raytype of triggered pulse
        solution_number = None
        if sim or self._sim_vertex:
            raytype = self._station[stnp.raytype_sim]
        else:
            raytype = self._station[stnp.raytype]
        for iS in raytypes[ch_Vpol]:
            if iS == raytype:
                solution_number = iS
        if solution_number is None:
            logger.warning(f"No solution for reference ch_Vpol ({ch_Vpol}) with ray type {raytype}!")
            return np.inf
        T_ref = timing[ch_Vpol][solution_number]
        trace_start_time_ref = self._station.get_channel(ch_Vpol).get_trace_start_time()

        if sim or self._sim_vertex: 
            k_ref = self._station[stnp.pulse_position_sim]# get pulse position for triggered pulse
        if not sim and not self._sim_vertex:  
            k_ref = self._station[stnp.pulse_position]

        reduced_chi2_Vpol = 0
        reduced_chi2_Hpol = 0
        dict_dt = {}
        dict_snr = {}
        chi2 = 0

        ### Loop over all channels. For each pulse, we first determine the arrival time,
        ### and then compute and add the chi squared.
        for channel_id in self._use_channels:
            channel = self._station.get_channel(channel_id)
            data_trace_full = channel.get_trace()
            data_timing_timing = channel.get_times()

            rec_traces[channel_id] = {}
            data_traces[channel_id] = {}
            data_timing[channel_id] = {}
            dict_dt[channel_id] = {}
            dict_snr[channel_id] = {}
            all_chi2[channel_id] = {}
            
            keys = sorted(traces[channel_id].keys(), key=lambda iS: iS!=solution_number) # start with the reference trace
            for key in keys:
                rec_trace = traces[channel_id][key]

                delta_T =  timing[channel_id][key] - T_ref

                ## before correlating, set values around maximum voltage trace data to zero
                delta_toffset = delta_T * self._sampling_rate
                # take into account unequal trace start times
                delta_toffset -= (channel.get_trace_start_time() - trace_start_time_ref) * self._sampling_rate

                ### figuring out the time offset for specfic trace
                dk = int(k_ref + delta_toffset ) # where do we expect the pulse to be wrt channel 6 main pulse and rec vertex position

                ## DETERMIINE PULSE REGION DUE TO REFERENCE TIMING
                if channel_id in self._Hpol_channels:
                    window = self._window_Hpol
                    Vrms = self._Vrms_Hpol
                else:
                    Vrms = self._Vrms
                    window = self._window_Vpol
                
                data_samples = np.arange(int(window[0] * self._sampling_rate), int(window[1]*self._sampling_rate)) + dk
                mask = (data_samples > 0) & (data_samples < len(data_timing_timing))
                if not np.any(mask):
                    continue # pulse not inside recorded trace, skipping...
                
                data_samples = data_samples[mask]
                start_index = data_samples[0] - dk # we need this to keep track of the expected pulse position in different data windows
                data_trace = data_trace_full[data_samples]
                data_times = data_samples / self._sampling_rate + channel.get_trace_start_time()
                snr = (np.max(data_trace) - np.min(data_trace)) / (2 * Vrms)
                dict_snr[channel_id][key] = snr

                # decide whether to use the timing from the reference channel, or use correlation
                if fixed_timing and not (channel_id == ch_Vpol & key == solution_number):
                    dt = dict_dt[ch_Vpol][solution_number]
                elif snr < 3.5 and channel_id in self._PA_cluster_channels and key == solution_number and channel_id != ch_Vpol:
                    dt = dict_dt[ch_Vpol][solution_number]
                elif snr < 3.5 and channel_id in self._fallback_channels.keys() and key == solution_number and self._use_fallback_timing:
                    dt = dict_dt[self._fallback_channels[channel_id]][solution_number]
                else:
                    corr = signal.correlate(data_trace, rec_trace)
                    lags = signal.correlation_lags(len(data_times), len(rec_trace)) + start_index # adding start_index ensures the same lags for different data windows

                    corr_window_start = 0#int(len(corr)/2 - 30 * self._sampling_rate)
                    corr_window_end = len(corr)#int(len(corr)/2 + 30 * self._sampling_rate)
                    
                    # for the PA cluster, we constrain the pulse position to be close to the
                    # pulse position of the reference Vpol TODO - extend to other channels?
                    if channel_id in self._PA_cluster_channels and not channel_id == ch_Vpol:
                        if key == solution_number:
                            corr_window_start = np.max([0, dict_dt[ch_Vpol][solution_number] - np.min(lags) - 5])
                            corr_window_end = np.min([len(corr), dict_dt[ch_Vpol][solution_number] - np.min(lags) + 5])

                    max_cor = np.arange(corr_window_start,corr_window_end, 1)[np.argmax(corr[corr_window_start:corr_window_end])]
                    dt = lags[max_cor] #max_cor - len(corr)
                    # rec_trace_1 = np.roll(rec_trace, math.ceil(-dt))[:len(data_trace_timing_1)]
                    # chi2_dt1 = np.sum((rec_trace_1  - data_trace_timing_1)**2 )/ ((self._Vrms)**2)/len(rec_trace)
                    # rec_trace_2 = np.roll(rec_trace, math.ceil(-dt - 1))[:len(data_trace_timing_1)]
                    # chi2_dt2 = np.sum((rec_trace_2 - data_trace_timing_1)**2) / ((self._Vrms)**2)/len(rec_trace)
                    # if chi2_dt2 < chi2_dt1:
                    #     dt = dt + 1
                    # else:
                    #     dt = dt

                    dict_dt[channel_id][key] = dt

                
                rec_trace = np.roll(rec_trace, dt - start_index)[:len(data_trace)]
                rec_traces[channel_id][key] = rec_trace
                data_traces[channel_id][key] = data_trace
                data_timing[channel_id][key] = data_times

                # Now we compute chi squared:
                chi2_for_channel_and_trace = np.sum((rec_trace - data_trace)**2 / ((Vrms)**2))

                # check if multiple pulses in the same channel overlap
                pulse_overlap = np.any([
                    np.abs(timing[channel_id][key] - timing[channel_id][other_key]) < (window[1] - window[0])
                    for other_key in keys if other_key != key])

                if pulse_overlap: # if they overlap, we exclude them from the fit, unless they are the reference pulse
                    if (channel_id == ch_Vpol or channel_id == ch_Hpol) and key == solution_number:
                        all_chi2[channel_id][key] = chi2_for_channel_and_trace
                elif (snr > 3.5) or (channel_id in self._PA_cluster_channels and key == solution_number):
                    all_chi2[channel_id][key] = chi2_for_channel_and_trace
                elif channel_id in self._fallback_channels.keys() and key == solution_number:
                    if dict_snr[self._fallback_channels[channel_id]][key] > 3.5 and self._use_fallback_timing:
                        all_chi2[channel_id][key] = chi2_for_channel_and_trace
                elif penalty:
                    snr_rec_trace = (np.max(rec_trace) - np.min(rec_trace)) / (2 * Vrms)
                    if snr_rec_trace > 4.0:
                        all_chi2[channel_id][key] = np.inf

        chi2_array = np.concatenate([list(d.values()) for d in all_chi2.values()])
        dof = np.sum(chi2_array.astype(bool)) # number of channels/traces included in the fit
        chi2 = np.sum(chi2_array)
        self.__dof = dof
        if not minimize: #TODO - make this a dict?
            full_output = [
                rec_traces, data_traces, data_timing, all_chi2, 
                [reduced_chi2_Vpol, reduced_chi2_Hpol], 
                over_reconstructed, dof, all_chi2] #all_chi2 used to be included_channels
            return full_output

        return chi2

    def end(self):
        pass


def save_fig(fig, fname, format='.png'):
    """
    Save a matplotlib Figure instance

    Parameters
    ----------
    fig : matplotlib Fig instance
    fname : string
        location / name
    format : string | list (default: '.png')
        format(s) to save to save the figure to.
        If a list, save the figure to multiple formats.
        Can also include '.pickle'/'.pkl' to enable the Fig to be
        imported and edited in the future

    """
    formats = np.atleast_1d(format)
    for fmt in formats:
        if ('pickle' in fmt) or ('pkl' in fmt):
            with open(fname+'.pkl', 'wb') as file:
                pickle.dump(fig, file)
        else:
            if not fmt[0] == '.':
                fmt = '.' + fmt
            fig.savefig(fname+fmt)
