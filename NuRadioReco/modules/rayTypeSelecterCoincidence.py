from radiotools import helper as hp
import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from scipy import signal
from scipy import optimize as opt
import datetime
import math
from NuRadioMC.utilities import medium
from NuRadioMC.SignalProp import propagation
import scipy


class rayTypeSelecter:


    def __init__(self):
        self.begin()

    def begin(self):
        pass

    def run(self, event, station, det, use_channels=[9, 14], template = None, shower_id = None, icemodel = 'greenland_simple', raytracing_method = 'analytic', sim = False, debug_plots = True):
        """
        Finds the pulse position of the triggered pulse in the phased array and returns the raytype of the pulse
        
        Paramters
        __________
        evt: Event
            The event to run the module on
        station: Station
            The station to run the Module on
        det: Detector
            The detector description
        use_channels: list
            List of phased array channels used for the pulse selection
        template: array
            Neutrino voltage template. Default = None
        shower_id: int
            For simulated event, the shower id is needed for vertex selection. Default = None
        icemodel: str
            Icemodel used for the propagation. Default = 'greenland_simple'
        raytracing_method: str
            Method used for the raytracing. Default = 'analytic'
        sim: Boolean
            True if simulated event is used. Default = False.
        debug_plots: Boolean
            Default = True.
        """
            
        
        ice = medium.get_ice_model(icemodel)
        prop = propagation.get_propagation_module(raytracing_method)
        sampling_rate = station.get_channel(0).get_sampling_rate() ## assume same for all channels
        station_id = station.get_id()
        
        if sim:
            vertex = event.get_sim_shower(shower_id)[shp.vertex]
        else:
            vertex = station[stnp.nu_vertex]
    
       
       
        if debug_plots: fig, axs = plt.subplots(3, figsize = (10, 10))
        if debug_plots: iax = 0
        #### determine position of pulse 
        T_ref = np.zeros(3)
        max_totaltrace = np.zeros(3)
        position_max_totaltrace = np.zeros(3)
        for raytype in [ 1, 2,3]:
            type_exist = 0
            total_trace = np.zeros(len(station.get_channel(0).get_trace()))
            for channel in station.iter_channels():
                if channel.get_id() in use_channels: # if channel in phased array
                    channel_id = channel.get_id()
                    x2 = det.get_relative_position(station_id, channel_id) + det.get_absolute_position(station_id)
                    r = prop( ice, 'GL1')
                    r.set_start_and_end_point(vertex, x2)
                    simchannel = []
                    r.find_solutions()
                    for iS in range(r.get_number_of_solutions()):
                        print("solution type in raytypeselecter", r.get_solution_type(iS))
                        if r.get_solution_type(iS) == raytype:
                           type_exist= 1#print("raytype", raytype)
                          
                           T = r.get_travel_time(iS)
     #                      print("ref channel", use_channels[0])
                           print("raytype", raytype)
                           if channel_id == use_channels[0]: T_ref[iS] = T
      #
                           dt = T - T_ref[iS]
                           dn_samples = dt * sampling_rate
                           dn_samples = math.ceil(-1*dn_samples)
                           cp_trace = np.copy(channel.get_trace())
                           cp_times= np.copy(channel.get_times())
                          # if sim: #cp_trace = np.copy(station.get_sim_station.get_channel_id(channel_id).get_trace())
                              
                          #     for sim_channel in station.get_sim_station().get_channels_by_channel_id(channel_id):
                          #         try:
                          #             simchannel += sim_channel
                          #         except:
                          #             simchannel = sim_channel
                               #sim_trace = simchannel.get_trace()
                               ### get timing for maximum for simchannel
                          #     timing_max_sim = simchannel.get_times()[np.argmax(simchannel.get_trace())]
                          # print("len cp trace", len(cp_trace))
                           if template is not None:
                                #### Run template through channel 6
                                #channel = station.get_channel(6)
                #                cp_trace = np.copy(channel.get_trace())
                                if len(cp_trace) != len(template):
                                    #add_zeros = np.zeros(abs(len(channel.get_trace()) - len(template)))
                                    if (len(template) < len(cp_trace)): template = np.pad(template, (0, abs(len(cp_trace) - len(template))))
                          # if sim:
                               #print("timing max sim", timing_max_sim)
                               #print("channel get times()", channel.get_times())
                               #cp_times_float = np.around(cp_times, decimals = 1)#cp_times.astype(np.float)
                               #print("channel times", channel.get_times())
                             #  pos_pulse = abs(cp_times - timing_max_sim).argmin()
                               #pos_pulse = np.where(cp_times_float ==  float(timing_max_sim))
                               #print("pos pulse", pos_pulse)
                             #  cp_trace[np.arange(len(cp_trace)) > (pos_pulse + 30 * sampling_rate)] = 0
                             #  cp_trace[np.arange(len(cp_trace)) < (pos_pulse - 20 * sampling_rate)] = 0
                           corr = scipy.signal.correlate(cp_trace*(1/(max(cp_trace))), template*(1/(max(template))))
                           dt = np.argmax(corr) - (len(corr)/2) +1
                           template_roll = np.roll(template, int(dt))
                           pos_max = np.argmax(template_roll) ## position of pulse
                           if channel_id == use_channels[0]: ## position for reference pulse
                                position_max_totaltrace = pos_max
                           cp_trace[np.arange(len(cp_trace)) < (position_max_totaltrace - 20 * sampling_rate)] = 0
                           cp_trace[np.arange(len(cp_trace)) > (position_max_totaltrace + 30 * sampling_rate)] = 0
                           trace = np.roll(cp_trace, dn_samples)
                           total_trace += trace
                           if debug_plots: axs[iax].plot(trace, color = 'darkgrey', lw = 5)
                           if debug_plots: 
                               if channel_id == use_channels[-1]:
                                   axs[iax].plot(total_trace, lw = 5, color = 'darkgreen', label= 'combined trace')
                                   axs[iax].legend(loc = 1, fontsize= 20)
                                   for tick in axs[iax].yaxis.get_majorticklabels():
                                       tick.set_fontsize(20)
                                   for tick in axs[iax].xaxis.get_majorticklabels():
                                       tick.set_fontsize(20)
                                   for tick in axs[2].yaxis.get_majorticklabels():
                                       tick.set_fontsize(20)
                                   for tick in axs[2].xaxis.get_majorticklabels():
                                       tick.set_fontsize(20)
            if debug_plots and type_exist:
               
                axs[iax].set_title("raytype: {}".format(['direct', 'refracted', 'reflected'][raytype-1]), fontsize = 30)
                axs[iax].grid()
                axs[iax].set_xlim((position_max_totaltrace - 40*sampling_rate, position_max_totaltrace + 40*sampling_rate))
                axs[iax].set_xlabel("samples", fontsize = 25)
                iax += 1
            if type_exist and debug_plots:
                if 1:
                    #axs[raytype-1].set_title("raytype {}".format(['direct', 'refracted', 'reflected'][raytype-1]))
                    axs[2].plot(abs(scipy.signal.correlate(total_trace, template_roll/max(template))), label = '{}'.format(['direct', 'refracted', 'reflected'][raytype-1])) #hp.get_normalized_xcorr(total_trace, template_roll/(max(template)))), label = 'total {}'.format(raytype))
                    axs[2].legend(fontsize = 20)
                    #axs[iax].legend(fontsize = 30)
                    axs[2].grid()
                    axs[2].set_title("correlation", fontsize = 30)
                    #tick in axs2.yaxis.get_majorticklabels():
   # tick.set_fontsize(20)
        #        if debug_plots: axs[raytype-1].set_xlim((position_max_totaltrace - 200, position_max_totaltrace+200))
                max_totaltrace[raytype-1] = max(scipy.signal.correlate(total_trace, template_roll/max(template)))#abs(hp.get_normalized_xcorr(total_trace/max(total_trace), template_roll/max(template))))
                where_are_NaNs = np.isnan(max_totaltrace)
                max_totaltrace[where_are_NaNs] = 0 
            #position_max_totaltrace[raytype-1] = pos_max#np.argmax((abs(total_trace)))
        #print("max total trace", max_totaltrace)
        if debug_plots:
            fig.tight_layout()
            fig.savefig("/lustre/fs22/group/radio/plaisier/software/simulations/full_reco/pulse_selection.png")
        
        ### store parameters
        reconstructed_raytype = ['direct', 'refracted', 'reflected'][np.argmax(max_totaltrace)]

        station.set_parameter(stnp.raytype, reconstructed_raytype)
        #print("position max totaltrace", position_max_totaltrace)
        station.set_parameter(stnp.pulse_position, position_max_totaltrace)
        #print("reconstructed raytype", reconstructed_raytype)
    def end(self):
        pass
