from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units
import NuRadioReco.framework.electric_field
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from NuRadioMC.SignalProp import analyticraytracing
import pickle, lzma

"""
-----------------------------------------
   EXAMPLE: Script to simulate the effects of birefringence on the polarization at the ARIANNA SouthPole station. 
            The measured data the simulation is compared to was taken from: DOI 10.1088/1748-0221/15/09/P09039
            A full study of this calculation was published here: DOI https://doi.org/10.1140/epjc/s10052-023-11238-y
-----------------------------------------
"""

SPice_position = np.array([0 , 0 , -1300 ])* units.m
ARIANNA_position = np.array([564, 37, -1])* units.m

ref_index_model = 'southpole_2015'
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


def pulse_polarization(T, th, ph):
    # function to calculate the polarization of a pulse
    
    F = fluence_hil(T, th, ph)
    return np.rad2deg(np.arctan(F[1]/F[0]))


def birefringence_propagation(e_field, emitter_depth):
    # function to apply the birefringence effects for a specific SPice depth

    propagation_pulse.set_trace(e_field, sr)
    SPice_position = np.array([0 , 0 , emitter_depth ])* units.m
    
    rays.set_start_and_end_point(SPice_position, ARIANNA_position)
    rays.find_solutions()
    rays.set_config(config)

    rays.apply_propagation_effects(propagation_pulse, ray_tracing_solution)

    trace = propagation_pulse.get_trace()

    pol = pulse_polarization(propagation_pulse.get_times(), trace[1], trace[2])
    return pol


depths = np.arange(-1700, -699, 100)
angle = 60 
waveform = 0

polar = []

with lzma.open("extra_files/SPice_pulses.xz", "r") as f:
    emitter_model = pickle.load(f)
efield = emitter_model['efields'][angle][0]
sr = emitter_model['sampling_rate']

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

    polarization = birefringence_propagation(electric_field, depths[depth])
    polar.append(polarization)


syst = np.load('extra_files/ARIANNA_systematics.npy')
ARIANNA_data = np.load('extra_files/ARIANNA_data.npy')

axs[1].fill_between(syst[0], syst[1], syst[2],color='deepskyblue', label='systematic uncertainty', alpha=.25)
axs[1].errorbar(ARIANNA_data[0], ARIANNA_data[1], ARIANNA_data[2],fmt='d',elinewidth=1, color='midnightblue', label='SPICE data')
axs[1].plot(depths, polar, 'r', zorder=10, label='simulated polarization')

comm_a = [-900 , -950]
comm_b = [-1230 , -1270]
comm_c = [-1520 , -1550]
axs[1].fill_between(comm_a, 0, 30, color='black',alpha=0.15,linewidth=0.0, label='Comm. Period')
axs[1].fill_between(comm_b, 0, 30, color='black',alpha=0.15,linewidth=0.0)
axs[1].fill_between(comm_c, 0, 30, color='black',alpha=0.15,linewidth=0.0)

axs[1].set_xlabel('depth [m]')
axs[1].set_ylabel(r'$\arctan\left(\frac{f_\Phi}{f_\theta}\right)[^\circ]$', fontsize=10)
axs[1].set_ylim(0,30)
axs[1].set_xlim(-1750,-750)        
axs[1].legend()

plt.tight_layout() 
plt.savefig('04_ARIANNA_simple_plot.png', dpi=400)