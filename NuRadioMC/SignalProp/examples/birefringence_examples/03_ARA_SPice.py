from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units
from NuRadioReco.framework import base_trace
import NuRadioReco.framework.electric_field
from NuRadioReco.detector import antennapattern
import matplotlib.pyplot as plt
import numpy as np
from radiotools import helper as hp
from scipy import signal
from NuRadioMC.SignalProp import analyticraytracing
import pickle, lzma

###-----------------------------------------
#   EXAMPLE: Script to simulate the effects of birefringence on the vpol amplitude at the ARA A5 station. 
#            The measured data the simulation is compared to was taken from: DOI 10.1088/1475-7516/2020/12/009
#            A full study of this calculation was published here: DOI https://doi.org/10.1140/epjc/s10052-023-11238-y
###-----------------------------------------

SPice_position = np.array([0 , 0 , -1300 ])* units.m
A5ara_position = np.array([-434, 4125, -200])* units.m

ref_index_model = 'ARA_2022'
ice = medium.get_ice_model(ref_index_model)
rays = analyticraytracing.ray_tracing(ice)
ray_tracing_solution = 0

propagation_pulse = NuRadioReco.framework.electric_field.ElectricField([1], position=None,
                shower_id=None, ray_tracing_id=None)

config = {'propagation': {}}
config['propagation']['attenuate_ice'] = True
config['propagation']['focusing_limit'] = 2
config['propagation']['focusing'] = False
config['propagation']['birefringence'] = True
config['propagation']['birefringence_model'] = 'southpole_A'
config['propagation']['birefringence_propagation'] = 'analytical'

prov = antennapattern.AntennaPatternProvider()
hpol = prov.load_antenna_pattern("XFDTD_Hpol_150mmHole_n1.78")
vpol = prov.load_antenna_pattern("XFDTD_Vpol_CrossFeed_150mmHole_n1.78")


def hilbert(T, th, ph):
    # function to calculate the hilbert envelope of a pulse
 
    h_th = signal.hilbert(th)
    h_ph = signal.hilbert(ph)    
    return(T, th, ph, np.abs(h_th), np.abs(h_ph))


def hilbert_max(T, th, ph):
    # function to calculate the maximum of the hilbert envelope of a pulse
    
    hil = hilbert(T, th, ph)
    return(hil[0][hil[3] == max(hil[3])], hil[0][hil[4] == max(hil[4])], max(hil[3]), max(hil[4]))
       

def fluence_hil(T, th, ph):
    # function to calculate the fluence of the hilbert envelope of a pulse for 70ns around the maximum of the pulse
    
    data = hilbert(T, th, ph)    
    m = hilbert_max(T, th, ph)
    
    if m[2] > m[3]:
        dominant = m[0]
    else:
        dominant = m[1]
    
    t_area = data[0][(data[0] < (dominant + 35)) &  (data[0] > (dominant - 35))]
    th_area = data[1][(data[0] < (dominant + 35)) &  (data[0] > (dominant - 35))]   
    ph_area = data[2][(data[0] < (dominant + 35)) &  (data[0] > (dominant - 35))]        
    dt = t_area[1] - t_area[0]

    F_th = np.sqrt(np.sum(th_area**2))
    F_ph = np.sqrt(np.sum(ph_area**2))    
    return(F_th, F_ph, dt)


def get_antenna_response(vec_r, e_field):
    # function to convert the electric field to voltage traces by folding it with the antenna response

    zenith, azimuth = hp.cartesian_to_spherical(*vec_r)

    t_v = base_trace.BaseTrace()
    t_h = base_trace.BaseTrace()
            
    ff = e_field.get_frequencies()
    efield_spectrum_theta = e_field.get_frequency_spectrum()[1]
    efield_spectrum_phi = e_field.get_frequency_spectrum()[2] 
    
    
    VEL_vpol = vpol.get_antenna_response_vectorized(ff, zenith, azimuth,
                                                    0 * units.deg, 0 * units.deg, 90 * units.deg, 0 * units.deg)  # the last four angles define the orientatino of an antenna (see https://nu-radio.github.io/NuRadioMC/NuRadioReco/pages/detector/detector_database_fields.html for details)
    VEL_hpol = hpol.get_antenna_response_vectorized(ff, zenith, azimuth,
                                                    0 * units.deg, 0 * units.deg, 90 * units.deg, 0 * units.deg)  # the last four angles define the orientatino of an antenna (see https://nu-radio.github.io/NuRadioMC/NuRadioReco/pages/detector/detector_database_fields.html for details)
    
    
    voltage_spectrum_vpol = efield_spectrum_theta * VEL_vpol['theta'] + efield_spectrum_phi * VEL_vpol['phi']
    voltage_spectrum_hpol = efield_spectrum_theta * VEL_hpol['theta'] + efield_spectrum_phi * VEL_hpol['phi']

    t_v.set_frequency_spectrum(voltage_spectrum_vpol, 1/dt) 
    t_h.set_frequency_spectrum(voltage_spectrum_hpol, 1/dt)

    v_pol = t_v.get_trace()
    h_pol = t_h.get_trace()

    return v_pol, h_pol

def birefringence_propagation(e_field, emitter_depth):
    # function to apply the birefringence effects for a specific SPice depth

    propagation_pulse.set_trace(e_field, sr)
    SPice_position = np.array([0 , 0 , emitter_depth ])* units.m
    
    rays.set_start_and_end_point(SPice_position, A5ara_position)
    rays.find_solutions()
    rays.set_config(config)

    rays.apply_propagation_effects(propagation_pulse, ray_tracing_solution)

    vec_r = rays.get_receive_vector(ray_tracing_solution)
    v_pol, h_pol = get_antenna_response(vec_r, propagation_pulse)

    t_v, t_h, v_max, h_max = hilbert_max(propagation_pulse.get_times(), v_pol, h_pol)
    return v_max, h_max


depths = np.arange(-1600, -799, 100)
angle = 15
waveform = 0

v_max = []
h_max = []

with lzma.open("extra_files/SPice_pulses.xz", "r") as f:
    emitter_model = pickle.load(f)
efield = emitter_model['efields'][angle][0]
sr = emitter_model['sampling_rate']
dt = 1/sr

trace = NuRadioReco.framework.electric_field.ElectricField([1], position=None,
                shower_id=None, ray_tracing_id=None)
trace.set_trace(np.array([np.zeros_like(efield[0]), efield[0], efield[1]]), sr)
        
#normalizing waveforms
pulse_fluence = fluence_hil(trace.get_times(), efield[0], efield[1])
norm = pulse_fluence[0] + pulse_fluence[1]
efield[0] = efield[0] / norm
efield[1] = efield[1] / norm

fig, axs = plt.subplots(2, 1, figsize=(6, 8))

axs[0].plot(trace.get_times(), efield[0], 'b', label='starting pulse, theta')
axs[0].plot(trace.get_times(), efield[1], 'r', label='starting pulse, phi')
axs[0].set_ylabel('amplitude [A.U.]')
axs[0].set_xlabel('time [ns]')
axs[0].legend()

r_component = np.zeros(len(efield[0]))
electric_field = np.stack((r_component, efield[0], efield[1]))

for depth in range(len(depths)):
    print('depth: ', depths[depth])

    vpol_max, hpol_max = birefringence_propagation(electric_field, depths[depth])
    v_max.append(vpol_max)
    h_max.append(hpol_max)


#norm_factor = 2150
norm_factor = 1000
loaded_data = np.load('extra_files/ARA_data.npy')

axs[1].errorbar(loaded_data[0], loaded_data[1], loaded_data[2], markersize = 3, linestyle='None', marker='s', label = 'measured vpol amplitude')
axs[1].plot(depths, norm_factor * np.array(v_max), label='pulse theta')
axs[1].plot(depths, norm_factor * np.array(h_max), label='pulse phi')
axs[1].set_xlabel('depth [m]')
axs[1].set_ylabel('amplitude [a.u.]')  
axs[1].legend()
plt.tight_layout() 
plt.savefig('03_ARA_simple_plot.png', dpi=400)