import NuRadioReco.modules.io.eventReader
from radiotools import helper as hp
import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.utilities import fft
from NuRadioReco.framework.parameters import stationParameters as stnp
import h5py
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.utilities import propagated_analytic_pulse
import matplotlib
from scipy import signal
from scipy import optimize as opt
from matplotlib import rc
from matplotlib.lines import Line2D
#from lmfit import Minimizer, Parameters, fit_report
import datetime
import math
from NuRadioMC.utilities import medium
from NuRadioMC.SignalProp import propagation
import scipy


class rayTypeSelecter:


    def __init__(self):
        self.begin()

    def begin(self):
        """
        begin method. This function is executed before the event loop.

        The antenna pattern provider is initialized here.
        """
        pass

    def run(self, event, shower_id, station, det,
            use_channels=[9, 14], template = None):
        ice = medium.get_ice_model('greenland_simple')
        prop = propagation.get_propagation_module('analytic')
        sim_station = True
        sim = True ### Use pulse position of noiseless trace 

        if sim_station: vertex = event.get_sim_shower(shower_id)[shp.vertex]
        debug_plots = True
        #vertex= station[stnp.nu_vertex] 
        x1 = vertex
        sampling_rate = station.get_channel(0).get_sampling_rate() ## assume same for all channels
        if debug_plots: fig, axs = plt.subplots(4)
        ich = 0
    #    print("template", template)
        #### determine position of pulse 
        T_ref = np.zeros(3)
        max_totaltrace = np.zeros(3)
        position_max_totaltrace = np.zeros(3)
        for raytype in [ 1, 2,3]:#
            k = 0
            total_trace = np.zeros(len(station.get_channel(0).get_trace()))
            for channel in station.iter_channels():
                
                if channel.get_id() in use_channels:
                    channel_id = channel.get_id()
                    x2 = det.get_relative_position(station.get_id(), channel_id) + det.get_absolute_position(station.get_id())
           #         r = prop( ice, 'GL1')
                    r = prop( ice, 'GL1')
                    r.set_start_and_end_point(x1, x2)

                    simchannel = []
                    r.find_solutions()
                    for iS in range(r.get_number_of_solutions()):
                        print("solution type in raytypeselecter", r.get_solution_type(iS))
                        #print("raytype", raytype)
                        if r.get_solution_type(iS) == raytype:
                           k = 1#print("raytype", raytype)   
                           T = r.get_travel_time(iS)
     #                      print("ref channel", use_channels[0])
                           print("raytype", raytype)
                           if channel.get_id() == use_channels[0]: T_ref[iS] = T
      #                     print("T REF", T_ref)
                           dt = T - T_ref[iS]
                           dn_samples = dt * sampling_rate
                           dn_samples = math.ceil(-1*dn_samples)
                           cp_trace = np.copy(channel.get_trace())
                           cp_times= np.copy(channel.get_times())
                           if sim: #cp_trace = np.copy(station.get_sim_station.get_channel_id(channel.get_id()).get_trace())
                              
                               for sim_channel in station.get_sim_station().get_channels_by_channel_id(channel_id):
                                   try:            
                                       simchannel += sim_channel
                                   except:
                                       simchannel = sim_channel
                               #sim_trace = simchannel.get_trace()
                               ### get timing for maximum for simchannel
                               timing_max_sim = simchannel.get_times()[np.argmax(simchannel.get_trace())]
                          # print("len cp trace", len(cp_trace))
                           if template is not None:
                                #### Run template through channel 6
                                #channel = station.get_channel(6)
                #                cp_trace = np.copy(channel.get_trace())
                                if len(cp_trace) != len(template):
                                    #add_zeros = np.zeros(abs(len(channel.get_trace()) - len(template)))
                                    if (len(template) < len(cp_trace)): template = np.pad(template, (0, abs(len(cp_trace) - len(template))))
                           if sim:
                               #print("timing max sim", timing_max_sim)
                               #print("channel get times()", channel.get_times())
                               #cp_times_float = np.around(cp_times, decimals = 1)#cp_times.astype(np.float)
                               #print("channel times", channel.get_times())
                               pos_pulse = abs(cp_times - timing_max_sim).argmin()
                               #pos_pulse = np.where(cp_times_float ==  float(timing_max_sim))
                               #print("pos pulse", pos_pulse)
                               cp_trace[np.arange(len(cp_trace)) > (pos_pulse + 30 * sampling_rate)] = 0
                               cp_trace[np.arange(len(cp_trace)) < (pos_pulse - 20 * sampling_rate)] = 0
                           corr = scipy.signal.correlate(cp_trace*(1/(max(cp_trace))), template*(1/(max(template))))
                            #xcorr = hp.get_normalized_xcorr(channel.get_trace(), template)
                            # print("xcorr", len(xcorr))
                           dt = np.argmax(corr) - (len(corr)/2) +1
       #                    print("dt", dt)
                           template_roll = np.roll(template, int(dt))
                           pos_max = np.argmax(template_roll)
                           if channel.get_id() == use_channels[0]: 
                               pos_max_6 = pos_max
                               position_max_totaltrace = pos_max_6
                           cp_trace[np.arange(len(cp_trace)) < (pos_max - 20 * sampling_rate)] = 0
                           cp_trace[np.arange(len(cp_trace)) > (pos_max + 30 * sampling_rate)] = 0
                           trace = np.roll(cp_trace, dn_samples)
                           total_trace += trace
                           if debug_plots: axs[raytype-1].plot(trace)
       #     print("pos max 6 ", pos_max_6)  ### pos of pulse does not depend on ra tracing solution. 
       #     position_max_totaltrace[raytype-1] = pos_max_6
        #          if debug_plots: axs[raytype-1].plot(trace)
        #    if debug_plots: axs[raytype].set_xlim((1500, 2250))
            if k:
                if debug_plots: axs[raytype-1].set_title("raytype {}".format(['direct', 'refracted', 'reflected'][raytype-1]))
                if debug_plots: axs[3].plot(abs(hp.get_normalized_xcorr(total_trace/max(total_trace), template_roll/(max(template)))), label = 'total {}'.format(raytype))
                if debug_plots: axs[3].legend()
                if debug_plots: axs[raytype-1].legend()
        #        if debug_plots: axs[raytype-1].set_xlim((position_max_totaltrace - 200, position_max_totaltrace+200))
                max_totaltrace[raytype-1] = max(abs(hp.get_normalized_xcorr(total_trace/max(total_trace), template_roll/max(template))))
                where_are_NaNs = np.isnan(max_totaltrace)
                max_totaltrace[where_are_NaNs] = 0 
            #position_max_totaltrace[raytype-1] = pos_max#np.argmax((abs(total_trace)))
        print("max total trace", max_totaltrace)
        reconstructed_raytype = ['direct', 'refracted', 'reflected'][np.argmax(max_totaltrace)]
        if debug_plots: fig.tight_layout()
        if debug_plots: fig.savefig("/lustre/fs22/group/radio/plaisier/software/simulations/full_reco/test.pdf")
        station.set_parameter(stnp.raytype, reconstructed_raytype) 
        print("position max totaltrace", position_max_totaltrace)
        station.set_parameter(stnp.pulse_position, position_max_totaltrace)#pulse location channel 6
        print("reconstructed raytype", reconstructed_raytype)
    def end(self):
        pass
