from NuRadioMC.SignalProp import propagation
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units
import NuRadioReco.framework.electric_field
import matplotlib.pyplot as plt
import numpy as np

from NuRadioMC.SignalProp import analyticraytracing

###-----------------------------------------
#   EXAMPLE: Script to simulate the effects of birefringence for a specific geometry and pulse. 
#            The relevant birefringent propaties along the ray path are extracted and plotted. (propagation path, final pulse, refractive indices, polarization vectors)
###-----------------------------------------


simulated_trace = np.load('extra_files/example_pulse.npy')

norm_factor = 1600
sr = 2*10**(9) * units.hertz
zeros = np.zeros(len(simulated_trace))
pulse_efield = np.vstack((zeros, simulated_trace * norm_factor, simulated_trace * norm_factor))

sim_pulse = NuRadioReco.framework.electric_field.ElectricField([1], position=None,
                 shower_id=None, ray_tracing_id=None)

initial_point = np.array([0 , 0 , -1300 ])* units.m
final_point = np.array([300, 1500, -150])* units.m
ray_tracing_solution = 0

fig, axs = plt.subplots(3, 2, figsize=(7.5, 10))
x_limits = [460, 520]
y_limits = [-1, 1.4]


# ---------------   raytracing:            analyticraytracing.py ---------------------
# ---------------   e-field propagation:   analyticraytracing.py ---------------------
# ---------------   biefringence model:    'southpole_A' ---------------------

bire_ice_model = 'southpole_A'

config = {'propagation': {}}
config['propagation']['attenuate_ice'] = False
config['propagation']['module'] = 'direct'
config['propagation']['focusing_limit'] = 2
config['propagation']['focusing'] = False
config['propagation']['birefringence'] = True
config['propagation']['birefringence_model'] = bire_ice_model

sim_pulse.set_trace(pulse_efield, sr)

ref_index_model = 'southpole_2015'
ice = medium.get_ice_model(ref_index_model)

rays = analyticraytracing.ray_tracing(ice)
rays.set_start_and_end_point(initial_point,final_point)
rays.find_solutions()
rays.set_config(config)

#plotting the starting pulse
axs[0, 0].plot(sim_pulse.get_times(), sim_pulse.get_trace()[1], label='starting pulse theta')
axs[0, 0].plot(sim_pulse.get_times(), sim_pulse.get_trace()[2], '--', label='starting pulse phi')
axs[0, 0].set_ylabel('amplitude [A.U.]')
axs[0, 0].set_xlabel('time [ns]')
axs[0, 0].set_ylim(y_limits)
axs[0, 0].set_xlim(x_limits)
axs[0, 0].legend(loc = 1)

rays.apply_propagation_effects(sim_pulse, ray_tracing_solution)
properties_directory = rays.get_path_properties_birefringence(ray_tracing_solution, bire_model = bire_ice_model)

#plotting the refractive indices along the path
xx = properties_directory['path'][:, 0]
yy = properties_directory['path'][:, 1]
zz = properties_directory['path'][:, 2]
rr = np.sqrt(yy**2 + xx**2)

axs[1, 1].plot(rr, properties_directory['refractive_index_x'] - 1.78,label= r'$N_x$')
axs[1, 1].plot(rr, properties_directory['refractive_index_y'] - 1.78,label= r'$N_y$')
axs[1, 1].plot(rr, properties_directory['refractive_index_z'] - 1.78,label= r'$N_z$')
axs[1, 1].plot(rr, properties_directory['first_refractive_index'] - properties_directory['nominal_refractive_index'],'--', label= r'$N_1$')
axs[1, 1].plot(rr, properties_directory['second_refractive_index'] - properties_directory['nominal_refractive_index'], '--', label= r'$N_2$')
axs[1, 1].legend()
axs[1, 1].set_ylabel(r'$N_{1, 2} - \langle  n  \rangle$')
axs[1, 1].set_xlabel('radius [m]')

#first normalized eigenvector along the propagation path
axs[2, 0].plot(rr, properties_directory['first_polarization_vector'][:, 1],label= r'$e_1^\theta$')
axs[2, 0].plot(rr, properties_directory['first_polarization_vector'][:, 2],label= r'$e_1^\phi$')
axs[2, 0].plot(rr, properties_directory['first_polarization_vector'][:, 0], 'r--',label= r'$e_1^r$')
axs[2, 0].legend()
axs[2, 0].set_xlabel('radius [m]')
axs[2, 0].set_ylabel('normalized eigenvector')

#second normalized eigenvector along the propagation path
axs[2, 1].plot(rr, properties_directory['second_polarization_vector'][:, 1],label= r'$e_2^\theta$')
axs[2, 1].plot(rr, properties_directory['second_polarization_vector'][:, 2],label= r'$e_2^\phi$')
axs[2, 1].plot(rr, properties_directory['second_polarization_vector'][:, 0], 'r--',label= r'$e_2^r$')
axs[2, 1].legend()
axs[2, 1].set_xlabel('radius [m]')
axs[2, 1].set_ylabel('normalized eigenvector')

#plotting the resulting pulse
axs[0, 1].plot(sim_pulse.get_times(), sim_pulse.get_trace()[1],label= 'birefringent pulse theta')
axs[0, 1].plot(sim_pulse.get_times(), sim_pulse.get_trace()[2], '--', label= 'birefringent pulse phi')
axs[0, 1].set_ylabel('amplitude [A.U.]')
axs[0, 1].set_ylim(y_limits)
axs[0, 1].set_xlim(x_limits)
axs[0, 1].legend(loc = 1)
axs[0, 1].set_xlabel('time [ns]')

#plotting the propagation path
axs[1, 0].plot(rr, zz, label= 'propagation path')
axs[1, 0].set_ylabel('depth [m]')
axs[1, 0].set_xlabel('radius [m]')
axs[1, 0].legend()

plt.tight_layout()
plt.savefig('02_path_info_plot.png', dpi=400)