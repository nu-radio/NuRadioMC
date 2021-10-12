import scipy
from scipy import constants
import scipy.stats as stats 
import NuRadioReco.modules.io.eventReader
from radiotools import helper as hp
import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.utilities import fft
from NuRadioReco.framework.parameters import stationParameters as stnp
import h5py
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.utilities import propagated_analytic_pulse_Coincidence
import matplotlib
from scipy import signal
from scipy import optimize as opt
from matplotlib import rc
import datetime
import math
from NuRadioReco.utilities import units
import datetime


class neutrinoDirectionReconstructorCoincidence:
    
    
    def __init__(self):
        pass

    def begin(self, station, det, event, shower_ids, use_channels=[6, 14], ch_Vpol = 6):
        """
        begin method. This function is executed before the event loop.

        We do not use this function for the reconsturctions. But only to determine uncertainties.
        """

        self._station = station
        self._use_channels = use_channels
        self._det = det
        self._sampling_rate = station.get_channel(0).get_sampling_rate()
        simulated_energy = 0
        for i in np.unique(shower_ids):
            simulated_energy += event.get_sim_shower(i)[shp.energy]

        #simulated_energy = event.get_sim_shower(shower_id)[shp.energy]
       # print("shower ids", shower_ids)
        shower_id = shower_ids
        self._simulated_azimuth = event.get_sim_shower(shower_id)[shp.azimuth]
        self._simulated_zenith = event.get_sim_shower(shower_id)[shp.zenith]
        sim_vertex = False
        if sim_vertex:
            vertex =event.get_sim_shower(shower_id)[shp.vertex] #station[stnp.nu_vertex]
        else:
            vertex = station[stnp.nu_vertex]
        simulation = propagated_analytic_pulse.simulation(False, vertex)#event.get_sim_shower(shower_id)[shp.vertex])
        rt = ['direct', 'refracted', 'reflected'].index(self._station[stnp.raytype]) + 1
        simulation.begin(det, station, use_channels, raytypesolution = rt, ch_Vpol = ch_Vpol)#[1, 2, 3] [direct, refracted, reflected]
        print("simulated zenith", np.rad2deg(self._simulated_zenith))
        print("simulatd azimuth", np.rad2deg(self._simulated_azimuth))        
        a, b, self._launch_vector_sim, c, d, e =  simulation.simulation(det, station, vertex[0],vertex[1], vertex[2], self._simulated_zenith, self._simulated_azimuth, simulated_energy, use_channels, first_iter = True)
        print("LAN VECTOR SIM", self._launch_vector_sim)
        print("viewing angle", np.rad2deg(c))
        #print(stop)
        self._simulation = simulation
        return self._launch_vector_sim
    
    def run(self, event, stations, det, shower_ids = None,
            use_channels=[6, 14], filenumber = 1, single_pulse = False, debug_plots = False, debugplots_path = None, template = False, sim_vertex = True, Vrms = 0.0114, only_simulation = False, ch_Vpol = 6, ch_Hpol = 13, full_station = True, brute_force = True, fixed_timing = False, restricted_input = True):

        """
        Module to reconstruct the direction of the event.
        event: Event
            The event to reconstruct the direction
        station: Station
            The station used to reconstruct the direction
        shower_ids: list
            list of shower ids for the event, only used if simulated vertex is used for input. Default shower_ids = None.
        use_channels: list
            list of channel ids used for the reconstruction
        filenumber: int
            This is only to link the debug plots to correct file. Default filenumber = 1.
        debug_plots: Boolean
            if True, debug plots are produced. Default debug_plots = False.
        debugplots_path: str
            Path to store the debug plots. Default = None.
        template: Boolean
            If True, ARZ templates are used for the reconstruction. If False, a parametrization is used. Default template = False.
        sim_vertex: Boolean
            If True, the simulated vertex is used. This is for debugging purposes. Default sim_vertex = False.
        Vrms: float
            Noise root mean squared. Default = 0.0114 V
        only_simulation: Boolean
            if True, the fit is not performed but only the simulated values are compared with the data. This is just for debugging purposes. Default only_simulation = False.
        ch_Vpol: int
            channel id of the Vpol used to determine reference timing. Should be the top Vpol of the phased array. Default ch_Vpol = 6.
        ch_Hpol: int
            channel id of Hpol nearest to the Vpol. Timing of the Hpol is determined using the vertex position, because difference is only 1 m. Default ch_Vpol = 13.
        full_station: Boolean
            If True, all the raytypes in the list use_channels are used. If False, only the triggered pulse is used. Default full_station = True.
        brute_force: Boolean
            If True, brute force method is used. If False, minimization is used. Default brute_force = True.
        fixed_timing: Boolean
            If True, the known positions of the pulses are used calculated using the vertex position. Only allowed when sim_vertex is True. If False, an extra correlation is used to find the exact pulse position. Default fixed_timing = False.
        restricted_input: Boolean
            If True, a reconstruction is performed a few degrees around the MC values. This is (of course) only for simulations. Default restricted_input = False.
        
        """
        ## there are 3 options for analytic models for the fit: 
        # - the Alvarez2009 model
        # - ARZ read in by templates
        # - ARZ average model 
        
       # station.set_is_neutrino()
        self._Vrms = Vrms
        self._stations = stations
        self._use_channels = use_channels
        self._det = det
        self._model_sys = 0.0 ## test amplitude effect of systematics on the model
#        self._single_pulse = False

        if sim_vertex:
            shower_id = shower_ids[0]
            reconstructed_vertex = event.get_sim_shower(shower_id)[shp.vertex]
            print("simulated vertex direction reco", event.get_sim_shower(shower_id)[shp.vertex])
        else:
            reconstructed_vertex = station[stnp.nu_vertex]
        
            print("reconstructed vertex direction reco", reconstructed_vertex)
      
        simulation = propagated_analytic_pulse_Coincidence.simulation(template, reconstructed_vertex) ### if the templates are used, than the templates for the correct distance are loaded
        print('self._stations[0]', self._stations[0])
        rt = ['direct', 'refracted', 'reflected'].index(self._stations[0][stnp.raytype]) + 1 ## raytype from the triggered pulse
        simulation.begin(det, stations[0], use_channels, raytypesolution = rt, ch_Vpol = ch_Vpol)
        self._simulation = simulation

        if self._stations[0].has_sim_station():
           
            sim_station = True
            simulated_zenith = event.get_sim_shower(shower_id)[shp.zenith]
            simulated_azimuth = event.get_sim_shower(shower_id)[shp.azimuth]
            self._simulated_azimuth = simulated_azimuth
            simulated_energy = 0
            for i in np.unique(shower_ids):
                print("flavour", event.get_sim_shower(shower_id)[shp.flavor])
                if 1:#(abs(event.get_sim_shower(shower_id)[shp.flavor]) != 12):
                    simulated_energy += event.get_sim_shower(i)[shp.energy]
                    print("simulated energy", simulated_energy)
            self.__simulated_energy = simulated_energy
            simulated_vertex = event.get_sim_shower(shower_id)[shp.vertex]
            ### values for simulated vertex and simulated direction
            tracsim, timsim, lv_sim, vw_sim, a, pol_sim = simulation.simulation(det, station, event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], simulated_zenith, simulated_azimuth, simulated_energy, use_channels, first_iter = True) 
       
            ## check SNR of channels
            SNR = []
            for ich, channel in enumerate(station.iter_channels()): ## checks SNR of channels
                print("channel {}, SNR {}".format(channel.get_id(),(abs(min(channel.get_trace())) + max(channel.get_trace())) / (2*Vrms) ))
                if channel.get_id() in use_channels:
                    SNR.append((abs(abs(min(channel.get_trace()))) + max(channel.get_trace())) / (2*Vrms))
                    print("SNR", SNR)
        
       
    
        channl = station.get_channel(use_channels[0])
        n_samples = channl.get_number_of_samples()
        self._sampling_rate = channl.get_sampling_rate()
        sampling_rate = self._sampling_rate
        
        if 1:
       
        #add_vertex_uncertainties = False
        #if add_vertex_uncertainties:
        #   for ie, efield in enumerate(station.get_sim_station().get_electric_fields()):
        #        if efield.get_channel_ids()[0] == 1:
        #            vertex_R = np.sqrt((simulated_vertex[0] )**2 + simulated_vertex[1]**2 + (simulated_vertex[2]+100)**2)
        #            vertex_zenith = hp.cartesian_to_spherical(simulated_vertex[0] , simulated_vertex[1], (simulated_vertex[2]+100))[0]
        #            vertex_azimuth = hp.cartesian_to_spherical(simulated_vertex[0] , simulated_vertex[1], (simulated_vertex[2]+100))[1]

                    #### add uncertainties in radians
        #            zenith_uncertainty = 0
        #            azimuth_uncertainty = 0
        #            R_uncertainty = 0
        #            vertex_zenith += zenith_uncertainty
        #            vertex_azimuth += azimuth_uncertainty
        #            vertex_R += R_uncertainty
        #            new_vertex = vertex_R *hp.spherical_to_cartesian(vertex_zenith, vertex_azimuth)
        #            new_vertex = [(new_vertex[0] ), new_vertex[1], new_vertex[2]-100]

        #           vertex_R = np.sqrt((new_vertex[0] )**2 + new_vertex[1]**2 + (new_vertex[2]+100)**2)
        
            print("simulated vertex", simulated_vertex)
            print('reconstructed', reconstructed_vertex)
           
            
            #### values for reconstructed vertex and simulated direction
            if sim_station:
                traces_sim, timing_sim, self._launch_vector, viewingangles_sim, rayptypes, a = simulation.simulation( det, station, event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], simulated_zenith, simulated_azimuth, simulated_energy, use_channels, first_iter = True)

         
                fsimsim = self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], minimize =  True, first_iter = True, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)
                all_fsimsim = self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], minimize =  False, first_iter = True, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[3]
                tracsim = self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], minimize =  False, first_iter = True, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[0]
                tracsim_recvertex = self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize =  False, first_iter = True,ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[0]
 
                fsim = self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], minimize =  True, first_iter = True, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)
              
                all_fsim = self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], minimize =  False, first_iter = True, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[3]
                print("Chi2 values for simulated direction and with/out simulated vertex are {}/{}".format(fsimsim, fsim))
            
                sim_reduced_chi2_Vpol = self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], minimize =  False, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[4][0]
               

                sim_reduced_chi2_Hpol = self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], minimize =  False, first_iter = True, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[4][1]
                self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize =  False, first_iter = True, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)

            signal_zenith, signal_azimuth = hp.cartesian_to_spherical(*self._launch_vector) ## due to
            sig_dir = hp.spherical_to_cartesian(signal_zenith, signal_azimuth)
            self._vertex_azimuth = hp.cartesian_to_spherical(*reconstructed_vertex)[1]


            ###### START MINIMIZATION ################
 
            cherenkov = 56 ## cherenov angle
            viewing_start = vw_sim - np.deg2rad(2)
            viewing_end = vw_sim + np.deg2rad(2)
           # viewing_start = np.deg2rad(cherenkov) - np.deg2rad(15)
           # viewing_end = np.deg2rad(cherenkov) + np.deg2rad(15)
            theta_start = np.deg2rad(-180)
            theta_end =  np.deg2rad(180)

            cop = datetime.datetime.now()
            print("SIMULATED DIRECTION {} {}".format(np.rad2deg(simulated_zenith), np.rad2deg(simulated_azimuth)))

            if only_simulation:
                print("no reconstructed is performed. The script is tested..")
            elif brute_force and not restricted_input:# restricted_input:
                results = opt.brute(self.minimizer, ranges=(slice(viewing_start, viewing_end, np.deg2rad(.5)), slice(theta_start, theta_end, np.deg2rad(1)), slice(np.log10(simulated_energy) - .5, np.log10(simulated_energy) + .5, .1)), full_output = True, finish = opt.fmin , args = (reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True, False, False, True, False, ch_Vpol, ch_Hpol, full_station))
            elif not brute_force and not restricted_input:
                print("not brute force")
                start_zenith = simulated_zenith
                start_azimuth = simulated_azimuth
                start_energy = np.log10(simulated_energy)    
                bounds = ([np.deg2rad(40), np.deg2rad(70)], [14, 20])
                #start_zenith = np.deg2rad(75)
                #start_azimuth = np.deg2rad(60)
                #start_energy = np.log10(simulated_energy)
                #### get fixed polarization, iterate over energy 
                method = 'Powell'
                angleresultsmin = []
                viewang = []
                en = []
                tmp = 100000
                R = np.arange(0, 360, 5)
                for r in R:#np.arange(60, 150, 30):
                    print("R", r)
                    self._r = np.deg2rad(r)
                    angleresults = scipy.optimize.minimize(self.minimizer, [np.deg2rad(57), 14], method = method, args=(reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True, False, False, True, False, ch_Vpol, ch_Hpol, False, True, [3, 10, 13]), bounds= bounds) 
                
                    angleresultsmin.append(angleresults.fun)
                    viewang.append(angleresults.x[0])
                    en.append(angleresults.x[1])
                    if angleresults.fun < tmp:
                        tmp = angleresults.fun
                        tmp_results = angleresults
 
                fig = plt.figure()
                ax = fig.add_subplot(131)
                ax.plot(R, angleresultsmin, 'o')
                ax = fig.add_subplot(132)
                ax.plot(R, np.rad2deg(viewang), 'o')
                ax.axhline(np.rad2deg(vw_sim))
                ax = fig.add_subplot(133)
                ax.plot(R, en, 'o')
                ax.axhline(np.log10(simulated_energy))
                fig.tight_layout()
                fig.savefig("/lustre/fs22/group/radio/plaisier/software/simulations/full_reco/try_{}.pdf".format(filenumber))
                #print("results", results)
                print("viewing angle",np.rad2deg(tmp_results.x[0]))
                print("view angle sim", np.rad2deg(vw_sim)) 
                print("energy", 10**tmp_results.x[1])
                print("simulated energy", simulated_energy)
               # print(stop)
      
                #### map vector settings to starting azimuth and zenith
                vertex_azimuth = hp.cartesian_to_spherical(*simulated_vertex)[1]
                if np.rad2deg(vertex_azimuth) < 0:
                    start_azimuth = vertex_azimuth + 2*np.pi
                else: 
                    start_azimuth= vertex_azimuth
          
                launch_zenith = hp.cartesian_to_spherical(*self._launch_vector)[0]
                popt = [ -0.7390428,  113.74132012]
                start_zenith = np.deg2rad(np.rad2deg(launch_zenith)*popt[0] + popt[1])
#popt [ -0.7390428  113.74132012]
                
                ## determine R due to zenith and azimuth ### which means same polarization
                ### for viewing angle and R [0, 360] calculate zen and az
                rotation_matrix = hp.get_rotation(sig_dir, np.array([0, 0,1]))
                cherenkov_angle = np.deg2rad(53)#results.x[0]#results[0][0]
                
                angles = np.arange(np.deg2rad(0), np.deg2rad(360), np.deg2rad(.1))
                zen = []
                az = []
                for angle in angles:

                    p3 = np.array([np.sin(cherenkov_angle)*np.cos(angle), np.sin(cherenkov_angle)*np.sin(angle), np.cos(cherenkov_angle)])
                    p3 = rotation_matrix.dot(p3)
                    az.append(hp.cartesian_to_spherical(p3[0], p3[1], p3[2])[1])
                    zen.append(hp.cartesian_to_spherical(p3[0], p3[1], p3[2])[0])
                if True:   
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(np.rad2deg(zen), np.rad2deg(az), 'o')
                    ax.plot(np.rad2deg(start_zenith), np.rad2deg(start_azimuth), 'o')
                    ax.plot(np.rad2deg(simulated_zenith), np.rad2deg(simulated_azimuth), 'o')
                    fig.savefig("/lustre/fs22/group/radio/plaisier/software/simulations/full_reco/test.pdf")
                 
                #### starting setttings such that viewing angle matches output previous step, r due to starting values, E due to previous
                
                #### give boundaries for r, theta and energy
                #### run for two sides of cone
                station.set_parameter(stnp.first_step, [tmp_results.x[0], tmp_results.x[1]])
                view = np.rad2deg(tmp_results.x[0])
                diff = np.diff((view, 56))
                view1 = 56 - diff
                view2 = 56 + diff
                view_start1 = np.deg2rad(view1)
                view_start2 = np.deg2rad(view2)
                start_energy = tmp_results.x[1]
                R_start = R[np.argmin(angleresultsmin)]#np.deg2rad(200)
                bounds = ([np.deg2rad(tmp_results.x[0]) - np.deg2rad(1), np.deg2rad(tmp_results.x[0]) + np.deg2rad(1)], [np.deg2rad(0), np.deg2rad(360)], [start_energy - .3, start_energy + .3])
                ## is energies for these the same?
                print("start viewing angle", [view_start1[0], R_start, start_energy])
                print("start viewing angle 2", np.rad2deg(view_start2)) 
                method = 'BFGS'
                #try_results = scipy.optimize.minimize(self.minimizer, [start_zenith, start_azimuth, start_energy], args = (reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True, False, False, False, False, ch_Vpol, ch_Hpol, True, False))
                #print("try results", try_results)
                #print(stop)
                R_possibilities = []
                print("input energy", 10**start_energy)
                for R_start1 in np.arange(0, 360, 2):
                   # print("view start", view_start1)
                   # print("R_start", np.deg2rad(R_start))
                   # print("start energy", start_energy)
                    x = self.minimizer([np.deg2rad(view),  np.deg2rad(R_start1) ,start_energy], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True, False, False, True, False, ch_Vpol, ch_Hpol, True, False, [ch_Vpol])
                   
                    if 1:#x < np.inf: 
                        R_possibilities.append(x)
                print("R values", np.arange(0, 360, 2))
                print("L values, for rec viewing angle and energy and iterating R", R_possibilities)
               # print("R_possibities", R_possibilities)
                #print(stop)
                #R_start = np.deg2rad(R_possibilities[-1])##np.deg2rad(90)
                start_energy = tmp_results.x[1]
                view = tmp_results.x[0]
                R_start = R[np.argmin(angleresultsmin)]
                print("start results 1 >>............... R start = ", R_start)
                results1 = scipy.optimize.minimize(self.minimizer, [view,  np.deg2rad(R_start), start_energy],method = method, args=(reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True, False, False, True, False, ch_Vpol, ch_Hpol, True, False, [ch_Vpol]), bounds= bounds)
                print("results for view, R, E first", results1)
                results = [results1.x[0], results1.x[1], results1.x[2]]
                 #print(stop)
                #bounds = ([np.deg2rad(view_start2) - np.deg2rad(1), np.deg2rad(view_start2) + np.deg2rad(1)], [np.deg2rad(0), np.deg2rad(360)], [start_energy - .5, start_energy + .5])
                #print(stop) 
                ### Fit around fitted viewing angle and energy and find R
                #results2 = scipy.optimize.minimize(self.minimizer, [view_start2, R_start, start_energy],method = method, args=(reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True, False, False, True, False, ch_Vpol, ch_Hpol, True, False), bounds= bounds)
                #results = opt.brute(self.minimizer, ranges=(slice(viewing_start, viewing_end, np.deg2rad(.5)), slice(theta_start, theta_end, np.deg2rad(1)), slice(np.log10(simulated_energy) - .5, np.log10(simulated_energy) + .5, .1)), full_output = True, finish = opt.fmin , args = (reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True, False, False, True, False, ch_Vpol, ch_Hpol, full_station))
                #print("results1.x", results1)
                #print("start energy", start_energy)
                #print("stop", stop)
                #y1=np.zeros([3],dtype=np.float)
                #for i in range(len(results1.x)):
                #    y1[i] = results1.x[i]
                #    print("results1.x", results1.x)
               #     print("results1.x[i]", results1.x[i])
               # print("y1 (should be same sas resulst1.x", y1)

                #L1 = self.minimizer([y1[0], y1[1], y1[2]], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True, False, False, True, False, ch_Vpol, ch_Hpol, True, False)
               # print(stop)
               # print("y", y)
                #y2=np.zeros([3],dtype=np.float)
                #for i in range(len(results2.x)):
                #    y2[i] = results2.x[i]

                #L2 = self.minimizer([y2[0], y2[1], y2[2]], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True, False, False, True, False, ch_Vpol, ch_Hpol, True, False)
                #print("L1", L1)
                #print("L2", L2)
                #if L1 < L2:
                #    print("L1 < L2")
                #    results = y1
                #elif L2 < L1:
                #    print("L2 < L1")
                #    results = y2
                ##else: #both find same minima
                 #   results = y1#print('results 1', results1)
                #print('results 2', results2)
                #print("results", results)
                #print("viewin gangle", np.rad2deg(results[0]))
                #print("energy", results[2])
               # print(stop)
            elif restricted_input:
                zenith_start =  simulated_zenith - np.deg2rad(5)
                zenith_end =simulated_zenith +  np.deg2rad(5)
                azimuth_start =simulated_azimuth - np.deg2rad(5)
                azimuth_end = simulated_azimuth + np.deg2rad(5)
                energy_start = np.log10(simulated_energy) - 1
                energy_end = np.log10(simulated_energy) + 1
                results = opt.brute(self.minimizer, ranges=(slice(zenith_start, zenith_end, np.deg2rad(1)), slice(azimuth_start, azimuth_end, np.deg2rad(1)), slice(energy_start, energy_end, .1)), finish = opt.fmin, full_output = True, args = (reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True, False, False, False, False, ch_Vpol, ch_Hpol, full_station))
                
            print('start datetime', cop)
            print("end datetime", datetime.datetime.now() - cop)
                
    
            ###### GET PARAMETERS #########
            
            if only_simulation:
                rec_zenith = simulated_zenith
                rec_azimuth = simulated_azimuth
                rec_energy = simulated_energy
            
            elif brute_force and not restricted_input:
                rotation_matrix = hp.get_rotation(sig_dir, np.array([0, 0,1]))
                cherenkov_angle = results[0][0]
                angle = results[0][1]

                p3 = np.array([np.sin(cherenkov_angle)*np.cos(angle), np.sin(cherenkov_angle)*np.sin(angle), np.cos(cherenkov_angle)])
                p3 = rotation_matrix.dot(p3)
                global_az = hp.cartesian_to_spherical(p3[0], p3[1], p3[2])[1]
                global_zen = hp.cartesian_to_spherical(p3[0], p3[1], p3[2])[0]
                global_zen = np.deg2rad(180) - global_zen
                

                rec_zenith = global_zen
                rec_azimuth = global_az
                rec_energy = 10**results[0][2]
            elif not brute_force and not restricted_input:
                rotation_matrix = hp.get_rotation(sig_dir, np.array([0, 0,1]))
                cherenkov_angle = results[0]
                angle = results[1]

                p3 = np.array([np.sin(cherenkov_angle)*np.cos(angle), np.sin(cherenkov_angle)*np.sin(angle), np.cos(cherenkov_angle)])
                p3 = rotation_matrix.dot(p3)
                global_az = hp.cartesian_to_spherical(p3[0], p3[1], p3[2])[1]
                global_zen = hp.cartesian_to_spherical(p3[0], p3[1], p3[2])[0]
                rec_zenith = np.deg2rad(180) - global_zen



                #rec_zenith = results.x[0][0]
                rec_azimuth = global_az#results.x[1][0]
                rec_energy = 10**results[2]
                ## fit around rec azimuth and zenith?
                bounds = ((np.deg2rad(rec_zenith) - np.deg2rad(10), np.deg2rad(rec_zenith) + np.deg2rad(10)), (np.deg2rad(rec_azimuth) - np.deg2rad(10), np.deg2rad(rec_azimuth) + np.deg2rad(10)), (np.log10(rec_energy) -  .5, np.log10(rec_energy) + .5))
                print("input zenith {} input azimuth {} input energy {}".format(np.rad2deg(rec_zenith), np.rad2deg(rec_azimuth), rec_energy))
                #results = scipy.optimize.minimize(self.minimizer, [rec_zenith, rec_azimuth,np.log10(rec_energy)],method = method, args=(reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True, False, False,False, False, ch_Vpol, ch_Hpol, True, False), bounds= bounds)
                #rec_zenith = results.x[0]
                #rec_azimuth = results.x[1] 
                #rec_energy = 10**results.x[2]
                #print(stop)
            elif restricted_input:
                rec_zenith = results[0][0]
                rec_azimuth = results[0][1]
                rec_energy = 10**results[0][2]

            ###### PRINT RESULTS ###############
            print("reconstructed energy {}".format(rec_energy))
            print("reconstructed zenith {} and reconstructed azimuth {}".format(np.rad2deg(rec_zenith), np.rad2deg(self.transform_azimuth(rec_azimuth))))
            print("         simualted zenith {}".format(np.rad2deg(simulated_zenith)))
            print("         simualted azimuth {}".format(np.rad2deg(simulated_azimuth)))
            
            print("     reconstructed zenith = {}".format(np.rad2deg(rec_zenith)))
            print("     reconstructed azimuth = {}".format(np.rad2deg(self.transform_azimuth(rec_azimuth))))
            
            print("###### seperate fit reconstructed valus")
            print("         zenith = {}".format(np.rad2deg(rec_zenith)))
            print("         azimuth = {}".format(np.rad2deg(self.transform_azimuth(rec_azimuth))))
            print("        energy = {}".format(rec_energy))
            print("         simualted zenith {}".format(np.rad2deg(simulated_zenith)))
            print("         simualted azimuth {}".format(np.rad2deg(simulated_azimuth)))
            print("         simulated energy {}".format(simulated_energy))
        
 
            
            print("RECONSTRUCTED DIRECTION ZENITH {} AZIMUTH {}".format(np.rad2deg(rec_zenith), np.rad2deg(self.transform_azimuth(rec_azimuth))))
            print("RECONSTRUCTED ENERGY", rec_energy)
            
           
            
            ## get the traces for the reconstructed energy and direction
            #print("test 1")
            tracrec = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = False, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[0]
            #print("test 2")
            fit_reduced_chi2_Vpol = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = False, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[4][0]
            fit_reduced_chi2_Hpol = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = False, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[4][1]
            channels_overreconstructed = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = False, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[5]
            extra_channel = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = False, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[6]

            fminfit = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize =  True, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)
           
            all_fminfit = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize =  False, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[3]
            bounds = ((14, 20))
            method = 'BFGS'
            results = scipy.optimize.minimize(self.minimizer, [14],method = method, args=(reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True, False, False,False, [simulated_zenith, simulated_azimuth], ch_Vpol, ch_Hpol, True, False), bounds= bounds)
           # method = 'BSFR'
            fmin_simdir_recvertex = self.minimizer([simulated_zenith, simulated_azimuth, results.x[0]], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = True, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)
            print("energy reconstructed for simulated direction", 10**results.x[0])

            print("FMIN SIMULATED direction with simulated vertex", fsim)
            print("FMIN RECONSTRUCTED VALUE FIT", fminfit)
            

          
            if debug_plots:
                linewidth = 5
                tracdata = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = False, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[1]
                #print(stop)
                timingdata = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = False, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[2]
                timingsim = self.minimizer([simulated_zenith, simulated_azimuth, np.log10(simulated_energy)], event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], first_iter = True, minimize = False, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[2]
                              
                timingsim_recvertex = self.minimizer([simulated_zenith, simulated_azimuth, np.log10(simulated_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], first_iter = True, minimize = False, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[2]
                #fig, ax = plt.subplots(len(use_channels), 3, sharex=False, figsize=(40, 20))
                plt.rc('xtick', labelsize = 25)
                plt.rc('ytick', labelsize = 25)
                fig, ax = plt.subplots(len(use_channels), 3, sharex=False, figsize=(40, 10*len(use_channels)))

                ich = 0
                SNRs = np.zeros((len(use_channels), 2)) 
                fig_sim, ax_sim = plt.subplots(len(use_channels), 1, sharex = True, figsize = (20, 10))

                for channel in station.iter_channels():
                    if channel.get_id() in use_channels: # use channels needs to be sorted
                        isch = 0
                        for sim_channel in self._stations[0].get_sim_station().get_channels_by_channel_id(channel.get_id()):
                            if isch == 0:
                                        
                                sim_trace = sim_channel
                                ax_sim[ich].plot( sim_channel.get_trace())
                            if isch == 1:
                                ax_sim[ich].plot( sim_channel.get_trace())
                                sim_trace += sim_channel
                            isch += 1
                       
                        ax_sim[ich].plot(channel.get_times(), channel.get_trace())
                        ax_sim[ich].plot(sim_trace.get_times(), sim_trace.get_trace())
                       
                        if len(tracdata[channel.get_id()]) > 0:
                            ax[ich][0].grid()
                            ax[ich][2].grid()
                            ax[ich][0].set_xlabel("timing [ns]", fontsize = 30)
                            ax[ich][0].plot(channel.get_times(), channel.get_trace(), lw = linewidth, label = 'data', color = 'black')
                            #ax[ich][0].fill_between(timingsim[channel.get_id()][0],tracsim[channel.get_id()][0]- sigma, tracsim[channel.get_id()][0] + sigma, color = 'red', alpha = 0.2 )
                            ax[ich][0].fill_between(timingdata[channel.get_id()][0], tracrec[channel.get_id()][0] - self._model_sys*tracrec[channel.get_id()][0], tracrec[channel.get_id()][0] + self._model_sys * tracrec[channel.get_id()][0], color = 'green', alpha = 0.2)
                            ax[ich][2].plot( np.fft.rfftfreq(len(tracdata[channel.get_id()][0]), 1/sampling_rate), abs(fft.time2freq( tracdata[channel.get_id()][0], sampling_rate)), color = 'black', lw = linewidth)
                            ax[ich][0].plot(timingsim[channel.get_id()][0], tracsim[channel.get_id()][0], label = 'simulation', color = 'orange', lw = linewidth)
                            ax[ich][0].plot(sim_trace.get_times(), sim_trace.get_trace(), label = 'sim channel', color = 'red', lw = linewidth)

                            #ax[ich][0].plot(timingsim_recvertex[channel.get_id()][0], tracsim_recvertex[channel.get_id()][0], label = 'simulation rec vertex', color = 'lightblue' , lw = linewidth, ls = '--')

                            ax[ich][0].set_xlim((timingsim[channel.get_id()][0][0], timingsim[channel.get_id()][0][-1]))
            
                            if 1:
                                 ax[ich][0].plot(timingdata[channel.get_id()][0], tracrec[channel.get_id()][0], label = 'reconstruction', lw = linewidth, color = 'green')
                                # ax[ich][0].plot(timingdata[channel.get_id()][0], tracrec[channel.get_id()][0], label = 'reconstruction', color = 'green')

                            ax[ich][2].plot( np.fft.rfftfreq(len(sim_trace.get_trace()), 1/sampling_rate), abs(fft.time2freq(sim_trace.get_trace(), sampling_rate)), lw = linewidth, color = 'red')
                            ax[ich][2].plot( np.fft.rfftfreq(len(tracsim[channel.get_id()][0]), 1/sampling_rate), abs(fft.time2freq(tracsim[channel.get_id()][0], sampling_rate)), lw = linewidth, color = 'orange')
                            if 1:
                                 ax[ich][2].plot( np.fft.rfftfreq(len(tracrec[channel.get_id()][0]), 1/sampling_rate), abs(fft.time2freq(tracrec[channel.get_id()][0], sampling_rate)), color = 'green', lw = linewidth)
                                 ax[ich][2].set_xlim((0, 1))
                                 ax[ich][2].set_xlabel("frequency [GHz]", fontsize = 10)        
            #                ax[ich][0].legend(fontsize = 30)
                           
                        if len(tracdata[channel.get_id()]) > 1:
                            ax[ich][1].grid()
                            ax[ich][1].set_xlabel("timing [ns]")
                            ax[ich][1].plot(channel.get_times(), channel.get_trace(), label = 'data', lw = linewidth, color = 'black')
                            ax[ich][1].fill_between(timingsim[channel.get_id()][1],tracsim[channel.get_id()][1]- Vrms, tracsim[channel.get_id()][1] + Vrms, color = 'red', alpha = 0.2 )
                            ax[ich][2].plot(np.fft.rfftfreq(len(timingsim[channel.get_id()][1]), 1/sampling_rate), abs(fft.time2freq(tracsim[channel.get_id()][1], sampling_rate)), lw = linewidth, color = 'red')
                            ax[ich][2].plot( np.fft.rfftfreq(len(tracdata[channel.get_id()][1]), 1/sampling_rate), abs(fft.time2freq(tracdata[channel.get_id()][1], sampling_rate)), color = 'black', lw = linewidth)
                            ax[ich][1].plot(timingsim[channel.get_id()][1], tracsim[channel.get_id()][1], label = 'simulation', color = 'orange', lw = linewidth)
                            ax[ich][1].plot(sim_trace.get_times(), sim_trace.get_trace(), label = 'sim channel', color = 'red', lw = linewidth)
                            if 1:#channel.get_id() in [6]:#,7,8,9]: 
                                ax[ich][1].plot(timingdata[channel.get_id()][1], tracrec[channel.get_id()][1], label = 'reconstruction', color = 'green', lw = linewidth)
                                ax[ich][1].fill_between(timingdata[channel.get_id()][1], tracrec[channel.get_id()][1] - self._model_sys*tracrec[channel.get_id()][1], tracrec[channel.get_id()][1] + self._model_sys * tracrec[channel.get_id()][1], color = 'green', alpha = 0.2)
                        
                            ax[ich][2].plot( np.fft.rfftfreq(len(tracsim[channel.get_id()][1]), 1/sampling_rate), abs(fft.time2freq(tracsim[channel.get_id()][1], sampling_rate)), lw = linewidth, color = 'orange')
                            #ax[ich][1].plot(timingsim_recvertex[channel.get_id()][1], tracsim_recvertex[channel.get_id()][1], label = 'simulation rec vertex', color = 'lightblue', lw = linewidth, ls = '--')
                            ax[ich][1].set_xlim((timingsim[channel.get_id()][1][0], timingsim[channel.get_id()][1][-1]))
                            if 1:#channel.get_id() in [6]:
                                 ax[ich][2].plot( np.fft.rfftfreq(len(tracrec[channel.get_id()][1]), 1/sampling_rate), abs(fft.time2freq(tracrec[channel.get_id()][1], sampling_rate)), color = 'green', lw = linewidth)
                            
                        
                        
                        ich += 1
                ax[0][0].legend(fontsize = 30)

                
                fig.tight_layout()
                print("output path for stored figure","{}/fit_{}.pdf".format(debugplots_path, filenumber))
                fig.savefig("{}/fit_{}.pdf".format(debugplots_path, filenumber, shower_id))
                fig_sim.savefig('{}/sim_{}.pdf'.format(debugplots_path, filenumber, shower_id))




             ### values for reconstructed vertex and reconstructed direction
            traces_rec, timing_rec, launch_vector_rec, viewingangle_rec, a, pol_rec =  simulation.simulation( det, station, reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], rec_zenith, rec_azimuth, rec_energy, use_channels, first_iter = True)

            ###### STORE PARAMTERS AND PRINT PARAMTERS #########
            station.set_parameter(stnp.extra_channels, extra_channel)
            station.set_parameter(stnp.over_rec, channels_overreconstructed)
            station.set_parameter(stnp.nu_zenith, rec_zenith)
            station.set_parameter(stnp.nu_azimuth, self.transform_azimuth(rec_azimuth))
            station.set_parameter(stnp.nu_energy, rec_energy)
            station.set_parameter(stnp.chi2, [fsim, fminfit, fsimsim, self.__dof, sim_reduced_chi2_Vpol, sim_reduced_chi2_Hpol, fit_reduced_chi2_Vpol, fit_reduced_chi2_Hpol, fmin_simdir_recvertex])
            station.set_parameter(stnp.launch_vector, [lv_sim, launch_vector_rec])
            station.set_parameter(stnp.polarization, [pol_sim, pol_rec])
            station.set_parameter(stnp.viewing_angle, [vw_sim, viewingangle_rec])
            print("chi2 for simulated rec vertex {}, simulated sim vertex {} and fit {}".format(fsim, fsimsim, fminfit))#reconstructed vertex
            print("chi2 for all channels simulated rec vertex {}, simulated sim vertex {} and fit {}".format(all_fsim, all_fsimsim, all_fminfit))#reconstructed vertex
            print("launch vector for simulated {} and fit {}".format(lv_sim, launch_vector_rec))
            zen_sim = hp.cartesian_to_spherical(*lv_sim)[0]
            zen_rec = hp.cartesian_to_spherical(*launch_vector_rec)[0]
            print("launch zenith for simulated {} and fit {}".format(np.rad2deg(zen_sim), np.rad2deg(zen_rec)))
            print("polarization for simulated {} and fit {}".format(pol_sim, pol_rec))
            print("polarization angle for simulated {} and fit{}".format(np.rad2deg(np.arctan2(pol_sim[2], pol_sim[1])), np.rad2deg(np.arctan2(pol_rec[2], pol_rec[1]))))
            print("viewing angle for simulated {} and fit {}".format(np.rad2deg(vw_sim), np.rad2deg(viewingangle_rec)))
            print("reduced chi2 Vpol for simulated {} and fit {}".format(sim_reduced_chi2_Vpol, fit_reduced_chi2_Vpol))
            print("reduced chi2 Hpol for simulated {} and fit {}".format(sim_reduced_chi2_Hpol, fit_reduced_chi2_Hpol))
            print("over reconstructed channels", channels_overreconstructed)
            print("extra channels", extra_channel)
            print("L for rec vertex sim direction rec energy:", fmin_simdir_recvertex)
            print("L for reconstructed vertexy directin and energy:", fminfit)

    def transform_azimuth(self, azimuth): ## from [-180, 180] to [0, 360]
        azimuth = np.rad2deg(azimuth)
        if azimuth < 0:
            azimuth = 360 + azimuth
        return np.deg2rad(azimuth)


                  
    def minimizer(self, params, vertex_x, vertex_y, vertex_z, minimize = True, timing_k = False, first_iter = False, banana = False,  direction = [0, 0], ch_Vpol = 6, ch_Hpol = False, full_station = True, single_pulse =False, channels_step = [3], fixed_timing = False):
            """""""""""
            params: list
                input paramters for viewing angle / direction
            vertex_x, vertex_y, vertex_z: float
                input vertex
            minimize: Boolean
                If true, minimization output is given (chi2). If False, parameters are returned. Default minimize = True.
            first_iter: Boolean
                If true, raytracing is performed. If false, raytracing is not perfomred. Default first_iter = False.
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
            single_pulse:
                if True, only 1 pulse is used from the reference Vpol. Default single_pulse = False. 

            
            """""""""""""""
    
    
    
            #print("params", params)
            model_sys = 0
            
            ### add filter
            #model_sys = 0
            #ff = np.fft.rfftfreq(600, .1)
            #mask = ff > 0
            #order = 8
            #passband = [300* units.MHz, 400* units.MHz]
            #b, a = signal.butter(order, passband, 'bandpass', analog=True)
            #w, ha = signal.freqs(b, a, ff[mask])
            #fa = np.zeros_like(ff, dtype=np.complex)
            #fa[mask] = ha
            #pol_filt = fa
            ######


            if banana: ## if input is viewing angle and energy, they need to be transformed to zenith and azimuth
                if len(params) ==3:
                    cherenkov_angle, angle, log_energy = params 
             
                if len(params) == 2:
                    cherenkov_angle, log_energy = params
                    angle = self._r#np.deg2rad(0)
                energy = 10**log_energy
                

                print("viewing angle and energy and angle ", [np.rad2deg(cherenkov_angle), log_energy, np.rad2deg(angle)])
                signal_zenith, signal_azimuth = hp.cartesian_to_spherical(*self._launch_vector)

                sig_dir = hp.spherical_to_cartesian(signal_zenith, signal_azimuth)
                rotation_matrix = hp.get_rotation(sig_dir, np.array([0, 0,1]))


                p3 = np.array([np.sin(cherenkov_angle)*np.cos(angle), np.sin(cherenkov_angle)*np.sin(angle), np.cos(cherenkov_angle)])
                p3 = rotation_matrix.dot(p3)
                azimuth = hp.cartesian_to_spherical(p3[0], p3[1], p3[2])[1]
                zenith = hp.cartesian_to_spherical(p3[0], p3[1], p3[2])[0]
                zenith = np.deg2rad(180) - zenith
               


                if np.rad2deg(zenith) > 100:
                    return np.inf ## not in field of view
                if np.rad2deg(zenith) < 20: 
                    return np.inf
             
            else: 
                if len(params) ==3:
                    zenith, azimuth, log_energy = params 
                    energy = 10**log_energy
             #       print("parameters zen {} az {} energy {}".format(np.rad2deg(zenith), np.rad2deg(azimuth), energy))
                if len(params) == 1:
                    log_energy = params
                 
                    energy = 10**log_energy[0]
                    zenith, azimuth = direction
            
            azimuth = self.transform_azimuth(azimuth)
           # print("input simulation azimuth {}, zenith {}".format(np.rad2deg(azimuth), np.rad2deg(zenith)))
            traces, timing, launch_vector, viewingangles, raytypes, pol = self._simulation.simulation(self._det, self._stations, vertex_x, vertex_y, vertex_z, zenith, azimuth, energy, self._use_channels, first_iter = first_iter) ## get traces due to neutrino direction and vertex position
           # print("viewing angles", np.rad2deg(viewingangles))
            chi2 = 0
            all_chi2 = []
            over_reconstructed = [] ## list for channel ids where reconstruction is larger than data
            extra_channel = 0 ## count number of pulses besides triggering pulse in Vpol + Hpol


            rec_traces = {} ## to store reconstructed traces
            data_traces = {} ## to store data traces
            data_timing = {} ## to store timing
        

            #get timing and pulse position for raytype of triggered pulse
            for iS in raytypes[ch_Vpol]:
                if raytypes[ch_Vpol][iS] == ['direct', 'refracted', 'reflected'].index(self._station[stnp.raytype]) + 1:
                    solution_number = iS#for reconstructed vertex it can happen that the ray solution does not exist 
            T_ref = timing[ch_Vpol][solution_number]

            k_ref = self._station[stnp.pulse_position]# get pulse position for triggered pulse
          
            ks = {}
            

            ich = -1
            reduced_chi2_Vpol = 0
            reduced_chi2_Hpol = 0
            channels_Vpol = self._use_channels#[ch_Vpol]#[1,4,6,10, 11, 12, 13]
            channels_Hpol = [ch_Hpol] 
            dict_dt = {}
            for station in self._stations:
                rec_traces[station.get_id()] = {}
                data_traces[station.get_id()] = {}
                data_timing[station.get_id()] = {}
                dict_dt[station.get_id()] = {}
                for ch in self._use_channels:
                    dict_dt[station.get_id()][ch] = {}
                for channel in station.iter_channels(): ### FIRST SET TIMINGS
                    if (channel.get_id() in channels_Vpol) and (channel.get_id() in self._use_channels):
                    #    print("CHANNLE sampling rate", channel.get_sampling_rate())
                        ich += 1 ## number of channel
                        data_trace = np.copy(channel.get_trace())
                        rec_traces[station.get_id()][channel.get_id()] = {}
                        data_traces[station.get_id()][channel.get_id()] = {}
                        data_timing[station.get_id()][channel.get_id()] = {}

                        ### if no solution exist, than analytic voltage is zero
                        rec_trace = np.zeros(len(data_trace))# if there is no raytracing solution, the trace is only zeros

                        delta_k = [] ## if no solution type exist then channel is not included
                        num = 0
                        chi2s = np.zeros(2)
                        max_timing_index = np.zeros(2)
                        max_data = []
                        for i_trace, key in enumerate(traces[station.get_id()][channel.get_id()]):#get dt for phased array pulse
    #                        print("i trace",i_trace)
                         #   dict_dt[channel.get_id()][i_trace] = {}
                            rec_trace_i = traces[station.get_id()][channel.get_id()][key]
                            rec_trace = rec_trace_i

                            max_trace = max(abs(rec_trace_i))
                            delta_T =  timing[station.get_id()][channel.get_id()][key] - T_ref
                            if int(delta_T) == 0:
                                trace_ref = i_trace
       
                            ## before correlating, set values around maximum voltage trace data to zero
                            delta_toffset = delta_T * self._sampling_rate

                            ### figuring out the time offset for specfic trace
                            dk = int(k_ref + delta_toffset )# where do we expect the pulse to be wrt channel 6 main pulse and rec vertex position
                            rec_trace1 = rec_trace
                            #template_spectrum = fft.time2freq(rec_trace1, 5)
                            #template_trace = fft.freq2time(template_spectrum, self._sampling_rate)
                            #print("len template trace", len(template_trace))
                            #rec_trace1 = template_trace
                              
                            if ((dk > self._sampling_rate * 60)&(dk < len(np.copy(data_trace)) - self._sampling_rate * 100)):#channel.get_id() == 9:#ARZ:#(channel.get_id() == 10 and i_trace == 1):
                                data_trace_timing = np.copy(data_trace) ## cut data around timing
                                ## DETERMIINE PULSE REGION DUE TO REFERENCE TIMING

                                data_timing_timing = np.copy(channel.get_times())#np.arange(0, len(channel.get_trace()), 1)#
                                dk_1 = data_timing_timing[dk]
                                #print("sampling rate", self._sampling_rate)
                                data_timing_timing = data_timing_timing[dk - 5*60 : dk + 5*100]
                                data_trace_timing = data_trace_timing[dk - 5*60 : dk + 5* 100]
                                data_trace_timing_1 = np.copy(data_trace_timing)
                                ### cut data trace timing to make window to search for pulse smaller
                                data_trace_timing_1[data_timing_timing < (dk_1 - 150)] = 0
                                data_trace_timing_1[data_timing_timing > (dk_1 + 5 *70)] = 0
                                ##### dt is the same for a channel+ corresponding Hpol. Dt will be determined by largest trace. We determine dt for each channel and trace seperately (except for low SNR channel 13).
                                max_data_2 = 0
                                if (i_trace ==0):

                                 #   print("trace 1")
                                    max_data_1 =  max(data_trace_timing_1)
                                if i_trace ==1:
                                    max_data_2 = max(data_trace_timing_1)
                                if 1:#((i_trace ==0) or (i_trace ==1 & (max_data_2 > max_data_1))):
                                    #print("######################## CHANNLE ID",i_trace)
                                    library_channels ={}
                                    library_channels[station.get_id()] = {}
                   #                 if channel.get_id() == 13:
                   #                     if (((max(channel.get_trace()) - min(channel.get_trace())) / (2*sigma)) > 4):#rec_trace_i
                                    #library_channels[6] = [1,2,4,5,6,10,11,12,13]
                                    for i_ch in self._use_channels:
                                        library_channels[station.get_id()][i_ch] = [i_ch]
                           #         library_channels[ch_Vpol] = [ch_Vpol, ch_Hpol]
                                    #library_channels[13] = [13]
          #                          else:
          #                              library_channels[6] = [6, 13]
                              
                                    
                                    #library_channels[1] = [1,2]
                                    #library_channels[4] = [4,5]
                                    #library_channels[10] = [10]
                                    #library_channels[11] = [11]
                                    #library_channels[12] = [12]
                                    if 1:
                                        corr = signal.hilbert(signal.correlate(rec_trace1, data_trace_timing_1))
                                    dt1 = np.argmax(corr) - (len(corr)/2) + 1
                        #            print("rec trace 1", len(rec_trace1))
                        #            print("data_trace_timing", len(data_trace_timing_1))
                                    chi2_dt1 = np.sum((np.roll(rec_trace1, math.ceil(-1*dt1)) - data_trace_timing_1)**2 / ((self._Vrms)**2))/len(rec_trace1)
                                    dt2 = np.argmax(corr) - (len(corr)/2)
                                    chi2_dt2 = np.sum((np.roll(rec_trace1, math.ceil(-1*dt2)) - data_trace_timing_1)**2 / ((self._Vrms)**2))/len(rec_trace1)
                                    if chi2_dt2 < chi2_dt1:
                                        dt = dt2
                                    else:
                                        dt = dt1
       

                                    corresponding_channels = library_channels[station.get_id()][channel.get_id()]
                                    for ch in corresponding_channels:
                                        dict_dt[station.get_id()][ch][i_trace] = dt
                                       
                                    if channel.get_id() == ch_Hpol: ## if SNR Hpol < 4, timing due to vertex is used
                                       if (((max(channel.get_trace()) - min(channel.get_trace())) / (2*self._Vrms)) < 4):#
                                           dict_dt[station.get_id()][ch][i_trace] = dict_dt[station.get_id()][ch_Vpol][i_trace]
              #                      print("dict dt", dict_dt)
            

                #dict_dt[1] = dict_dt[6]
                #dict_dt[2] = dict_dt[6]
                #dict_dt[4] = dict_dt[6]
                #dict_dt[5] = dict_dt[6]
                #dict_dt[10] = dict_dt[6]
                #dict_dt[11] = dict_dt[6]
                #dict_dt[12] = dict_dt[6]
                #dict_dt[13] = dict_dt[6]
                if fixed_timing:
                    for i_ch in self._use_channels:
                    
                        dict_dt[i_ch][0] = dict_dt[station.get_id()][ch_Vpol][trace_ref]
                        dict_dt[i_ch][1] = dict_dt[station.get_id()][ch_Vpol][trace_ref]



                dof = 0
                for channel in station.iter_channels():
                    rec_trace[station.get_id()] = {}
                    data_traces[station.get_id()] = {}
                    data_timing[station.get_id()] = {}
                    if channel.get_id() in self._use_channels:
                        chi2s = np.zeros(2)
                        echannel = np.zeros(2)
                        dof_channel = 0
                        rec_traces[station.get_id()][channel.get_id()] = {}
                        data_traces[station.get_id()][channel.get_id()] = {}
                        data_timing[station.get_id()][channel.get_id()] = {}
                        weight = 1
                        data_trace = np.copy(channel.get_trace())
                        data_trace_timing = data_trace
                        if traces[station.get_id()][channel.get_id()]:
                            for i_trace, key in enumerate(traces[station.get_id()][channel.get_id()]): ## iterate over ray type solutions
                                rec_trace_i = traces[station.get_id()][channel.get_id()][key]
                                rec_trace = rec_trace_i

                                max_trace = max(abs(rec_trace_i))
                                delta_T =  timing[station.get_id()][channel.get_id()][key] - T_ref

                                ## before correlating, set values around maximum voltage trace data to zero
                                delta_toffset = delta_T * self._sampling_rate

                                ### figuring out the time offset for specfic trace
                                dk = int(k_ref + delta_toffset )
                                rec_trace1 = rec_trace
                               
                                if ((dk > self._sampling_rate *60)&(dk < len(np.copy(data_trace)) - self._sampling_rate * 100)):
                                    data_trace_timing = np.copy(data_trace) ## cut data around timing
                                    ## DETERMIINE PULSE REGION DUE TO REFERENCE TIMING
                                    
                                    data_timing_timing = np.copy(channel.get_times())#np.arange(0, len(channel.get_trace()), 1)#
                                    dk_1 = data_timing_timing[dk]
                                    data_timing_timing = data_timing_timing[dk - 60*5 : dk + 100*5]
                                    data_trace_timing = data_trace_timing[dk -60*5 : dk + 100*5]
                                    data_trace_timing_1 = np.copy(data_trace_timing)
                                    data_trace_timing_1[data_timing_timing < (dk_1 - 30 * 5)] = 0
                                    data_trace_timing_1[data_timing_timing > (dk_1 + 70* 5)] = 0
                                    
                                    dt = dict_dt[channel.get_id()][i_trace]
                                
                                    rec_trace1 = np.roll(rec_trace1, math.ceil(-1*dt))
                                     
                                    rec_trace1 = rec_trace1[30 *5  : 130 *5]
                                    data_trace_timing = data_trace_timing[30*5 :  130 * 5]
                                    data_timing_timing = data_timing_timing[30 *5 :  130* 5]
                                    delta_k.append(int(k_ref + delta_toffset + dt )) ## for overlapping pulses this does not work
                                    ks[channel.get_id()] = delta_k
                                    rec_traces[channel.get_id()][i_trace] = rec_trace1
                                    data_traces[channel.get_id()][i_trace] = data_trace_timing
                                    data_timing[channel.get_id()][i_trace] = data_timing_timing
                                    SNR = abs(max(data_trace_timing) - min(data_trace_timing) ) / (2*self._Vrms)
                               
                                    #### add filter
                                   # ff = np.fft.rfftfreq(600, .1)
                                   # mask = ff > 0
                                   # order = 8
                                   # passband = [200* units.MHz, 300* units.MHz]
                                   # b, a = signal.butter(order, passband, 'bandpass', analog=True)
                                   # w, ha = signal.freqs(b, a, ff[mask])
                                   # fa = np.zeros_like(ff, dtype=np.complex)
                                   # fa[mask] = ha
                                   # pol_filt = fa
                                # if vertex is wrong reconstructed, than it can be that data_timing_timing does not exist. In that case, set to zero.
                                    try:
                                        max_timing_index[i_trace] = data_timing_timing[np.argmax(data_trace_timing)]
                                    except:
                                        max_timing_index[i_trace] = 0

                           #         print("SNR single pulse full station channelstep", [SNR, single_pulse, full_station, channels_step])
                                    if fixed_timing:
                                        if SNR > 3.5:
                                            echannel[i_trace] = 1
                                    if  ((channel.get_id() == ch_Vpol ) and (i_trace == trace_ref)) or fixed_timing:
                                        trace_trig = i_trace
                           
                                        if not fixed_timing:  echannel[i_trace] = 0
                                        chi2s[i_trace] = np.sum((rec_trace1 - data_trace_timing)**2 / ((self._Vrms+model_sys*abs(data_trace_timing))**2))#/len(rec_trace1)
                                        data_tmp = fft.time2freq(data_trace_timing, self._sampling_rate) #* pol_filt
                                        power_data_6 = np.sum(fft.freq2time(data_tmp, self._sampling_rate)**2)
                                        rec_tmp = fft.time2freq(rec_trace1, self._sampling_rate) #* pol_filt
                                        power_rec_6 = np.sum(fft.freq2time(rec_tmp, self._sampling_rate) **2)
                                        #reduced_chi2_Vpol +=  np.sum((rec_trace1 - data_trace_timing)**2 / ((sigma+model_sys*abs(data_trace_timing))**2))/len(rec_trace1)
                                        dof_channel += 1
                                     
                                        if (i_trace == trace_ref) and (channel.get_id() == 3):
                                            Vpol_ref = np.sum((rec_trace1 - data_trace_timing)**2 / ((self._Vrms+model_sys*abs(data_trace_timing))**2))/len(rec_trace1)
                                            reduced_chi2_Vpol +=  np.sum((rec_trace1 - data_trace_timing)**2 / ((self._Vrms+model_sys*abs(data_trace_timing))**2))/len(rec_trace1)

                                     
                                        
                                    elif ((channel.get_id() == ch_Hpol) and (len(channels_step) < 2) and (i_trace == trace_ref) and (not single_pulse)):
                              #          print("Hpol", channel.get_id())
                                        echannel[i_trace] = 0
                                        mask_start = 0
                                        mask_end = 80 * self._sampling_rate
                                        if 1:#(SNR > 3.5): #if there is a clear pulse, we can limit the timewindow
                                            mask_start = int(np.argmax(rec_trace1) - 10 * self._sampling_rate)
                                            mask_end = int(np.argmax(rec_trace1) + 10 * self._sampling_rate)
                                        if mask_start < 0:
                                            mask_start = 0
                                        if mask_end > len(rec_trace1):
                                            mask_end = len(rec_trace1)
                                        chi2s[i_trace] = np.sum((rec_trace1[mask_start:mask_end] - data_trace_timing[mask_start:mask_end])**2 / ((self._Vrms)**2))#/len(rec_trace1[mask_start:mask_end])
                                        data_tmp = fft.time2freq(data_trace_timing, self._sampling_rate) #* pol_filt
                                        
                                        #####
                                        #power_data_13 = np.sum((fft.freq2time(data_tmp, self._sampling_rate) )**2)
                                        #rec_tmp = fft.time2freq(rec_trace1, self._sampling_rate) #* pol_filt
                                        #power_rec_13 = np.sum((fft.freq2time(rec_tmp, self._sampling_rate) )**2)
                                        #R_rec = power_rec_6/power_rec_13
                                        #R_data = power_data_6/power_data_13
                                        #####
                                        
                                        if i_trace == trace_ref:
                                            Hpol_ref = np.sum((rec_trace1[mask_start:mask_end] - data_trace_timing[mask_start:mask_end])**2 / ((self._Vrms)**2))/len(rec_trace1[mask_start:mask_end])
                                        reduced_chi2_Hpol += np.sum((rec_trace1[mask_start:mask_end] - data_trace_timing[mask_start:mask_end])**2 / ((self._Vrms)**2))/len(rec_trace1[mask_start:mask_end])
                                        add = True
                                        dof_channel += 1
                                        
                                    elif ((SNR > 3.5) and (not single_pulse) and (len(channels_step) < 2)):#and (i_trace ==trace_ref)):
                        #                print("else")
                                        mask_start = 0
                                        echannel[i_trace] = 1
                                        mask_end = 80 * int(self._sampling_rate)
                                        if channel.get_id() in channels_Hpol:
                                            if 1:#(SNR > 3.5):
                                                mask_start = int(np.argmax(rec_trace1) - 10 * self._sampling_rate)
                                                mask_end = int(np.argmax(rec_trace1) + 10 * self._sampling_rate)

                                        if mask_start < 0:
                                            mask_start = 0
                                        if mask_end > len(rec_trace1):
                                            mask_end = len(rec_trace1)
                                                                 
                                        add = True
                                        chi2s[i_trace] = np.sum((rec_trace1[mask_start:mask_end] - data_trace_timing[mask_start:mask_end])**2 / ((self._Vrms)**2))#/len(rec_trace1[mask_start:mask_end])
                                        dof_channel += 1
                                    elif ((full_station) and (not single_pulse) and (len(channels_step) < 2)):
                                        mask_start = 0
                                        mask_end = 80 * int(self._sampling_rate)
                                        echannel[i_trace] = 0
                                     
                                        if abs(max(rec_trace1[mask_start:mask_end]) - min(rec_trace1[mask_start:mask_end]))/(2*self._Vrms) > 4.0:
                                         #   print("over reconstructed", abs(max(rec_trace1[mask_start:mask_end]) - min(rec_trace1[mask_start:mask_end]))/(2*sigma))
                                            chi2s[i_trace] = np.inf
                           

                                else:
                                    #if one of the pulses is not inside the window
                                    rec_traces[station.get_id()][channel.get_id()][i_trace] = np.zeros(80 * self._sampling_rate)
                                    data_traces[station.get_id()][channel.get_id()][i_trace] = np.zeros(80 * self._sampling_rate)
                                    data_timing[station.get_id()][channel.get_id()][i_trace] = np.zeros(80 * self._sampling_rate)
                                       
     
                        else:#if no raytracing solution exist
                            rec_traces[station.get_id()][channel.get_id()][0] = np.zeros(80 * int(self._sampling_rate))
                            data_traces[station.get_id()][channel.get_id()][0] = np.zeros(80 * int(self._sampling_rate))
                            data_timing[station.get_id()][channel.get_id()][0] = np.zeros(80 * int(self._sampling_rate))
                            rec_traces[station.get_id()][channel.get_id()][1] = np.zeros(80 * int(self._sampling_rate))
                            data_traces[station.get_id()][channel.get_id()][1] = np.zeros(80 * int(self._sampling_rate))
                            data_timing[station.get_id()][channel.get_id()][1] = np.zeros(80 * int(self._sampling_rate))

                        #### if the pulses are overlapping, than we don't include them in the fit because the timing is not exactly known. Only for channel 6 and 13 we use 1 single pulse.
                        if max(data_timing[channel.get_id()][0]) > min(data_timing[channel.get_id()][1]):
                            if int(min(data_timing[channel.get_id()][1])) != 0:
            #
                                if (channel.get_id() == ch_Vpol):
                                    chi2 += chi2s[trace_ref]# = np.sum((rec_trace1 - data_trace_timing)**2 / ((sigma+model_sys*abs(data_trace_timing))**2))/len(rec_trace1)
                                    reduced_chi2_Vpol = Vpol_ref
                                    dof += 1
                                if (channel.get_id() == ch_Hpol):
                                    chi2 += chi2s[trace_ref]
                                    reduced_chi2_Hpol = Hpol_ref
                             
                           
            
                        else:
                                extra_channel += echannel[0]
                                extra_channel += echannel[1]
                                chi2 += chi2s[0]
                                chi2 += chi2s[1]
                 #               print("chi2s", chi2s)
                                dof += dof_channel
                                all_chi2.append(chi2s[0])
                                all_chi2.append(chi2s[1])
                
            self.__dof = dof
            if timing_k:
                return ks
            if not minimize:
                return [rec_traces, data_traces, data_timing, all_chi2, [reduced_chi2_Vpol, reduced_chi2_Hpol], over_reconstructed, extra_channel]
            print("parameters zen {} az {} energy {} viewing angle {} chi2 {}".format(np.rad2deg(zenith), np.rad2deg(azimuth), np.log10(energy), np.rad2deg(viewingangles), chi2))

         
            return chi2 ### sum of reduced chi2 per time window
            
            """
                    ### helper functions for plotting
            def mollweide_azimuth(az):
                az -= (simulated_azimuth - np.deg2rad(180)) ## put simulated azimuth at 180 degrees
                az = np.remainder(az, np.deg2rad(360)) ## rotate values such that they are between 0 and 360
                az -= np.deg2rad(180)
                return az
        
            def mollweide_zenith(zen):
                zen -= (simulated_zenith  - np.deg2rad(90)) ## put simulated azimuth at 90 degrees
                zen = np.remainder(zen, np.deg2rad(180)) ## rotate values such that they are between 0 and 180
                zen -= np.deg2rad(90) ## hisft to mollweide projection
                return zen


            def get_normalized_angle(angle, degree=False, interval=np.deg2rad([0, 360])):
                import collections
                if degree:
                    interval = np.rad2deg(interval)
                delta = interval[1] - interval[0]
                if(isinstance(angle, (collections.Sequence, np.ndarray))):
                    angle[angle >= interval[1]] -= delta
                    angle[angle < interval[0]] += delta
                else:
                    while (angle >= interval[1]):
                        angle -= delta
                    while (angle < interval[0]):
                        angle += delta
                return angle
            """



    
        
        
    def end(self):
        pass
