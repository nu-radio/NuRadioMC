from NuRadioMC.SignalProp import propagation
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units
import NuRadioReco.framework.electric_field
import matplotlib.pyplot as plt
import numpy as np

from NuRadioMC.SignalProp import radioproparaytracing
from NuRadioMC.SignalProp import analyticraytracing

"""
    EXAMPLE: Script to simulate the effects of birefringence on a simple pulse using different available propagation modules and ice-models. 
"""

simulated_trace = np.load('extra_files/example_pulse.npy')

sr = 2*10**(9) * units.hertz
zeros = np.zeros(len(simulated_trace))
pulse_efield = np.vstack((zeros, simulated_trace, simulated_trace))

sim_pulse = NuRadioReco.framework.electric_field.ElectricField([1], position=None,
                 shower_id=None, ray_tracing_id=None)

initial_point = np.array([0 , 0 , -1200 ])* units.m
final_point = np.array([1500, 1500, -200])* units.m
ray_tracing_solution = 0

fig, axs = plt.subplots(4, 1, figsize=(10, 12))
x_limits = [460, 520]

def birefringence_propagation(config, raytracing, isolution, plot, lim, tracing_module, propagation_module, birefringent_ice_model):

    #function to initialize a new config file, apply propagation effects and plot the results

    sim_pulse.set_trace(pulse_efield, sr)
    ref_index_model = 'southpole_2015'

    ice = medium.get_ice_model(ref_index_model)

    if raytracing == 'analytical':
        rays = analyticraytracing.ray_tracing(ice)
    elif raytracing == 'numerical':
        rays = radioproparaytracing.radiopropa_ray_tracing(ice)

    rays.set_start_and_end_point(initial_point,final_point)
    rays.find_solutions()
    rays.set_config(config)

    axs[plot].plot(sim_pulse.get_times(), sim_pulse.get_trace()[1], label='starting pulse theta')
    axs[plot].plot(sim_pulse.get_times(), sim_pulse.get_trace()[2], '--', label='starting pulse phi')

    rays.apply_propagation_effects(sim_pulse, isolution)

    axs[plot].plot(sim_pulse.get_times(), sim_pulse.get_trace()[1],label= 'birefringent pulse theta')
    axs[plot].plot(sim_pulse.get_times(), sim_pulse.get_trace()[2], label= 'birefringent pulse phi')

    axs[plot].text(490, -0.0003, 'raytracing module:                 ' + tracing_module)
    axs[plot].text(490, -0.0004, 'e-field propagation module:   ' + propagation_module)
    axs[plot].text(490, -0.0005, 'birefringence ice model:         ' + birefringent_ice_model)

    axs[plot].set_ylabel('amplitude [A.U.]')
    axs[plot].set_xlim(lim)

config = {'propagation': {}}
config['propagation']['attenuate_ice'] = True
config['propagation']['focusing_limit'] = 2
config['propagation']['focusing'] = False

# ---------------   raytracing:            analyticraytracing.py ---------------------
# ---------------   e-field propagation:   analyticraytracing.py ---------------------
# ---------------   biefringence model:    'southpole_A' ---------------------

config['propagation']['birefringence'] = True
config['propagation']['birefringence_model'] = 'southpole_A'
config['propagation']['birefringence_propagation'] = 'analytical'

birefringence_propagation(config, 'analytical', ray_tracing_solution, 0, x_limits, 'analyticraytracing.py', 'analyticraytracing.py', 'southpole_A')


# ---------------   raytracing:            analyticraytracing.py ---------------------
# ---------------   e-field propagation:   analyticraytracing.py ---------------------
# ---------------   biefringence model:    'southpole_B' ---------------------

config['propagation']['birefringence'] = True
config['propagation']['birefringence_model'] = 'southpole_B'
config['propagation']['birefringence_propagation'] = 'analytical'

birefringence_propagation(config, 'analytical', ray_tracing_solution, 1, x_limits, 'analyticraytracing.py', 'analyticraytracing.py', 'southpole_B')


# ---------------   raytracing:            analyticraytracing.py ---------------------
# ---------------   e-field propagation:   radioproparaytracing.py ---------------------
# ---------------   biefringence model:    'southpole_A' ---------------------

config['propagation']['birefringence'] = True
config['propagation']['birefringence_model'] = 'southpole_A'
config['propagation']['birefringence_propagation'] = 'numerical'

birefringence_propagation(config, 'analytical', ray_tracing_solution, 2, x_limits, 'analyticraytracing.py', 'radioproparaytracing.py', 'southpole_A')


# ---------------   raytracing:            radioproparaytracing.py ---------------------
# ---------------   e-field propagation:   radioproparaytracing.py ---------------------
# ---------------   biefringence model:    'southpole_A' ---------------------

propa_config = dict()
propa_config['propagation'] = dict(
    attenuate_ice = True,
    focusing_limit = 2,
    focusing = False,
    birefringence = True,
    radiopropa = dict(
        mode = 'iterative',
        iter_steps_channel = [25., 2., .5], #unit is meter
        iter_steps_zenith = [.5, .05, .005], #unit is degree
        auto_step_size = False,
        max_traj_length = 10000) #unit is meter
)
propa_config['speedup'] = dict(
    delta_C_cut = 40 * units.degree
)

birefringence_propagation(propa_config, 'numerical', ray_tracing_solution, 3, x_limits, 'radioproparaytracing.py', 'radioproparaytracing.py', 'southpole_A')

# ---------------   plotting the results ---------------------
axs[0].legend(loc = 1)
axs[-1].set_xlabel('time [ns]')
plt.tight_layout()
plt.savefig('01_simple_propagation_plot.png', dpi=400)